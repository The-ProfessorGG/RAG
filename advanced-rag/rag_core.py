"""Dependency-free Advanced RAG implementation used by the notebook and server."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json
import math
import os
import re
import urllib.error
import urllib.request


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "may", "of", "on", "or",
    "our", "the", "their", "this", "to", "what", "when", "which", "with",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    text: str


@dataclass
class Candidate:
    chunk: Chunk
    tfidf_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    compressed_text: str = ""

    def public(self, citation: int) -> dict:
        return {
            "citation": citation,
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "title": self.chunk.title,
            "tfidf_score": round(self.tfidf_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "rrf_score": round(self.rrf_score, 5),
            "rerank_score": round(self.rerank_score, 4),
            "excerpt": self.compressed_text or self.chunk.text,
        }


def load_documents(documents_dir: str | Path) -> list[Document]:
    root = Path(documents_dir)
    documents: list[Document] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        first_line = text.splitlines()[0] if text else path.stem
        title = first_line.lstrip("# ").strip() or path.stem.replace("-", " ").title()
        documents.append(Document(path.stem, title, path.name, text))
    if not documents:
        raise ValueError(f"No Markdown documents found in {root}")
    return documents


def chunk_documents(documents: Iterable[Document], max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        body = re.sub(r"^#\s+.*?$", "", document.text, count=1, flags=re.MULTILINE).strip()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        part_number = 0
        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) <= max_words:
                part_number += 1
                chunks.append(Chunk(f"{document.id}#{part_number}", document.id, document.title, paragraph))
                continue
            step = max(1, max_words - overlap)
            for start in range(0, len(words), step):
                window = words[start:start + max_words]
                if not window:
                    break
                part_number += 1
                chunks.append(Chunk(f"{document.id}#{part_number}", document.id, document.title, " ".join(window)))
                if start + max_words >= len(words):
                    break
    return chunks


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.term_counts = [Counter(tokenize(chunk.title + " " + chunk.text)) for chunk in chunks]
        self.df = Counter()
        for counts in self.term_counts:
            self.df.update(counts.keys())
        self.size = len(chunks)
        self.vectors = [self._vectorize(counts) for counts in self.term_counts]
        self.norms = [self._norm(vector) for vector in self.vectors]

    def _idf(self, term: str) -> float:
        return math.log((1 + self.size) / (1 + self.df.get(term, 0))) + 1.0

    def _vectorize(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {term: (count / total) * self._idf(term) for term, count in counts.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values())) or 1.0

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        query_vector = self._vectorize(Counter(tokenize(query)))
        query_norm = self._norm(query_vector)
        results = []
        for chunk, vector, norm in zip(self.chunks, self.vectors, self.norms):
            dot = sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
            results.append((chunk, dot / (query_norm * norm)))
        return sorted(results, key=lambda item: (-item[1], item[0].id))


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(chunk.title + " " + chunk.text)) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter()
        for counts in self.term_counts:
            self.df.update(counts.keys())
        self.size = len(chunks)

    def _idf(self, term: str) -> float:
        frequency = self.df.get(term, 0)
        return math.log(1 + (self.size - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        results = []
        for chunk, counts, length in zip(self.chunks, self.term_counts, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / max(1.0, self.avg_length))
                score += self._idf(term) * (tf * (self.k1 + 1)) / denominator
            results.append((chunk, score))
        return sorted(results, key=lambda item: (-item[1], item[0].id))


PHRASE_EXPANSIONS = [
    ({"leave", "vacation", "holiday", "pto"}, "annual leave paid days allowance request notice advance manager"),
    ({"carry", "carried", "unused"}, "annual leave carry-over limit deadline expiry"),
    ({"remote", "home", "hybrid"}, "remote work office anchor days international approval"),
    ({"incident", "breach", "security"}, "security incident report notification deadline"),
    ({"receipt", "expense", "travel"}, "travel expense receipt threshold reimbursement deadline"),
    ({"parental", "caregiver", "parent"}, "parental leave primary secondary caregiver weeks notice"),
    ({"refund", "billing", "charge"}, "customer refund eligibility billing error approval limit"),
    ({"equipment", "laptop", "device"}, "company equipment return loss theft reporting"),
    ({"learning", "course", "certification"}, "learning budget manager director approval exam fee"),
]


def rewrite_query(question: str) -> str:
    """Transparent deterministic rewrite used instead of a hidden model call."""
    original_tokens = content_tokens(question)
    expansions: list[str] = []
    for triggers, expansion in PHRASE_EXPANSIONS:
        if original_tokens & triggers:
            expansions.append(expansion)
    normalized = " ".join(tokenize(question))
    return " ".join([normalized, *expansions]).strip()


def reciprocal_rank_fusion(
    tfidf_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    candidate_k: int = 10,
    rrf_k: int = 60,
) -> list[Candidate]:
    by_id: dict[str, Candidate] = {}
    for rank, (chunk, score) in enumerate(tfidf_results[:candidate_k], start=1):
        candidate = by_id.setdefault(chunk.id, Candidate(chunk))
        candidate.tfidf_score = score
        candidate.rrf_score += 1 / (rrf_k + rank)
    for rank, (chunk, score) in enumerate(bm25_results[:candidate_k], start=1):
        candidate = by_id.setdefault(chunk.id, Candidate(chunk))
        candidate.bm25_score = score
        candidate.rrf_score += 1 / (rrf_k + rank)
    return sorted(by_id.values(), key=lambda item: (-item.rrf_score, item.chunk.id))


def rerank_candidates(question: str, rewritten_query: str, candidates: list[Candidate]) -> list[Candidate]:
    query_terms = content_tokens(question + " " + rewritten_query)
    max_rrf = max((candidate.rrf_score for candidate in candidates), default=1.0) or 1.0
    max_bm25 = max((candidate.bm25_score for candidate in candidates), default=1.0) or 1.0
    for candidate in candidates:
        chunk_terms = content_tokens(candidate.chunk.text)
        title_terms = content_tokens(candidate.chunk.title)
        coverage = len(query_terms & chunk_terms) / max(1, len(query_terms))
        title_coverage = len(query_terms & title_terms) / max(1, len(title_terms))
        lexical_strength = candidate.bm25_score / max_bm25
        fused_strength = candidate.rrf_score / max_rrf
        candidate.rerank_score = (
            0.40 * fused_strength
            + 0.25 * lexical_strength
            + 0.25 * coverage
            + 0.10 * title_coverage
        )
    return sorted(candidates, key=lambda item: (-item.rerank_score, item.chunk.id))


def compress_text(text: str, query: str, max_sentences: int = 2) -> str:
    query_terms = content_tokens(query)
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    scored = []
    for position, sentence in enumerate(sentences):
        terms = content_tokens(sentence)
        overlap = len(query_terms & terms)
        coverage = overlap / max(1, len(query_terms))
        scored.append((coverage, overlap, -position, sentence))
    scored.sort(reverse=True)
    selected = [item[3] for item in scored[:max_sentences] if item[1] > 0]
    if not selected and sentences:
        selected = [sentences[0]]
    selected_set = set(selected)
    return " ".join(sentence for sentence in sentences if sentence in selected_set)


def build_prompt(question: str, rewritten_query: str, candidates: list[Candidate]) -> str:
    evidence = "\n\n".join(
        f"[{index}] {candidate.chunk.title}\n{candidate.compressed_text}"
        for index, candidate in enumerate(candidates, start=1)
    )
    return (
        "Answer using only the compressed evidence. Cite claims with bracketed source numbers. "
        "Prefer the most specific policy and say when evidence is insufficient.\n\n"
        f"ORIGINAL QUESTION\n{question}\n\nRETRIEVAL QUERY\n{rewritten_query}\n\nEVIDENCE\n{evidence}"
    )


def local_extractive_answer(question: str, rewritten_query: str, candidates: list[Candidate]) -> str:
    question_terms = content_tokens(question + " " + rewritten_query)
    sentences = []
    for citation, candidate in enumerate(candidates, start=1):
        for sentence in SENTENCE_RE.split(candidate.compressed_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = content_tokens(sentence)
            overlap = len(question_terms & terms)
            score = overlap / max(1, len(question_terms)) + 0.2 * candidate.rerank_score
            sentences.append((score, citation, sentence))
    sentences.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected, seen = [], set()
    for score, citation, sentence in sentences:
        if score <= 0 or sentence.lower() in seen:
            continue
        selected.append(f"{sentence} [{citation}]")
        seen.add(sentence.lower())
        if len(selected) == 2:
            break
    return " ".join(selected) if selected else "The filtered evidence is insufficient to answer the question."


def openai_answer(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when RAG_GENERATOR=openai")
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
        "instructions": "You are a careful policy assistant. Use only supplied evidence and preserve citation markers.",
        "input": prompt,
        "store": False,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed with HTTP {exc.code}: {detail}") from exc
    parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            parts.extend(
                content["text"]
                for content in item.get("content", [])
                if content.get("type") == "output_text" and content.get("text")
            )
    if not parts:
        raise RuntimeError("The OpenAI response did not contain output text")
    return "\n".join(parts)


class AdvancedRAG:
    def __init__(
        self,
        documents_dir: str | Path,
        top_k: int = 4,
        candidate_k: int = 10,
        score_threshold: float = 0.34,
    ):
        self.documents = load_documents(documents_dir)
        self.chunks = chunk_documents(self.documents)
        self.tfidf = TfidfIndex(self.chunks)
        self.bm25 = BM25Index(self.chunks)
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.score_threshold = score_threshold

    def retrieve(self, question: str) -> tuple[str, list[Candidate], list[Candidate]]:
        rewritten = rewrite_query(question)
        tfidf_results = self.tfidf.search(rewritten)
        bm25_results = self.bm25.search(rewritten)
        fused = reciprocal_rank_fusion(tfidf_results, bm25_results, self.candidate_k)
        reranked = rerank_candidates(question, rewritten, fused)
        filtered = [candidate for candidate in reranked if candidate.rerank_score >= self.score_threshold][: self.top_k]
        if len(filtered) < min(2, self.top_k):
            filtered = reranked[: min(max(2, self.top_k), len(reranked))]
        for candidate in filtered:
            candidate.compressed_text = compress_text(candidate.chunk.text, rewritten)
        return rewritten, reranked, filtered

    def ask(self, question: str, generator: str | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        rewritten, reranked, filtered = self.retrieve(question)
        prompt = build_prompt(question, rewritten, filtered)
        generator = (generator or os.environ.get("RAG_GENERATOR", "local")).lower()
        if generator == "openai":
            answer = openai_answer(prompt)
        elif generator == "local":
            answer = local_extractive_answer(question, rewritten, filtered)
        else:
            raise ValueError("generator must be 'local' or 'openai'")
        sources = [candidate.public(index) for index, candidate in enumerate(filtered, start=1)]
        return {
            "architecture": "advanced-rag",
            "question": question,
            "rewritten_query": rewritten,
            "answer": answer,
            "generator": generator,
            "sources": sources,
            "trace": [
                {"step": "rewrite", "detail": rewritten},
                {"step": "hybrid_retrieve", "detail": f"TF-IDF + BM25 produced {len(reranked)} fused candidates."},
                {"step": "rerank_filter", "detail": f"Kept {len(filtered)} candidates at threshold {self.score_threshold}."},
                {"step": "compress", "detail": "Selected query-relevant sentences from each retained chunk."},
                {"step": "generate", "detail": f"Generated the answer with the {generator} generator."},
            ],
            "debug": {
                "top_reranked": [
                    {"chunk_id": candidate.chunk.id, "title": candidate.chunk.title, "score": round(candidate.rerank_score, 4)}
                    for candidate in reranked[:6]
                ]
            },
        }

    def describe_documents(self) -> list[dict]:
        return [asdict(document) | {"text": document.text[:240] + ("…" if len(document.text) > 240 else "")} for document in self.documents]
