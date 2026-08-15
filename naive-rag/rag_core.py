"""Dependency-free Naive RAG implementation used by the notebook and server."""

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


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float

    def public(self, citation: int) -> dict:
        return {
            "citation": citation,
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "title": self.chunk.title,
            "score": round(self.score, 4),
            "excerpt": self.chunk.text,
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
    """Small educational TF-IDF cosine index implemented with the standard library."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks
        self.term_counts = [Counter(tokenize(chunk.title + " " + chunk.text)) for chunk in chunks]
        self.document_frequency = Counter()
        for counts in self.term_counts:
            self.document_frequency.update(counts.keys())
        self.size = len(chunks)
        self.vectors = [self._vectorize_counts(counts) for counts in self.term_counts]
        self.norms = [self._norm(vector) for vector in self.vectors]

    def _idf(self, term: str) -> float:
        return math.log((1 + self.size) / (1 + self.document_frequency.get(term, 0))) + 1.0

    def _vectorize_counts(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {term: (count / total) * self._idf(term) for term, count in counts.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values())) or 1.0

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_counts = Counter(tokenize(query))
        query_vector = self._vectorize_counts(query_counts)
        query_norm = self._norm(query_vector)
        results: list[SearchResult] = []
        for chunk, vector, norm in zip(self.chunks, self.vectors, self.norms):
            dot = sum(query_vector.get(term, 0.0) * weight for term, weight in vector.items())
            score = dot / (query_norm * norm)
            results.append(SearchResult(chunk, score))
        results.sort(key=lambda result: (-result.score, result.chunk.id))
        return results[: max(1, top_k)]


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[{index}] {result.chunk.title}\n{result.chunk.text}"
        for index, result in enumerate(results, start=1)
    )
    return (
        "Answer the question using only the evidence below. "
        "Cite supporting passages with bracketed numbers such as [1]. "
        "If the evidence is insufficient, say so.\n\n"
        f"EVIDENCE\n{context}\n\nQUESTION\n{question}"
    )


def local_extractive_answer(question: str, results: list[SearchResult]) -> str:
    question_terms = content_tokens(question)
    candidates: list[tuple[float, int, str]] = []
    for citation, result in enumerate(results, start=1):
        for sentence in SENTENCE_RE.split(result.chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = content_tokens(sentence)
            overlap = len(question_terms & terms)
            coverage = overlap / max(1, len(question_terms))
            title_bonus = 0.2 * len(question_terms & content_tokens(result.chunk.title))
            score = coverage + title_bonus + (0.15 * result.score)
            candidates.append((score, citation, sentence))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[str] = []
    seen: set[str] = set()
    for score, citation, sentence in candidates:
        normalized = sentence.lower()
        if score <= 0 or normalized in seen:
            continue
        selected.append(f"{sentence} [{citation}]")
        seen.add(normalized)
        if len(selected) == 2:
            break
    return " ".join(selected) if selected else "The retrieved evidence is insufficient to answer the question."


def openai_answer(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when RAG_GENERATOR=openai")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    payload = {
        "model": model,
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
    text_parts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_parts.append(content["text"])
    if not text_parts:
        raise RuntimeError("The OpenAI response did not contain output text")
    return "\n".join(text_parts)


class NaiveRAG:
    def __init__(self, documents_dir: str | Path, top_k: int = 3):
        self.documents = load_documents(documents_dir)
        self.chunks = chunk_documents(self.documents)
        self.index = TfidfIndex(self.chunks)
        self.top_k = top_k

    def ask(self, question: str, generator: str | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        results = self.index.search(question, self.top_k)
        prompt = build_prompt(question, results)
        generator = (generator or os.environ.get("RAG_GENERATOR", "local")).lower()
        if generator == "openai":
            answer = openai_answer(prompt)
        elif generator == "local":
            answer = local_extractive_answer(question, results)
        else:
            raise ValueError("generator must be 'local' or 'openai'")
        sources = [result.public(index) for index, result in enumerate(results, start=1)]
        return {
            "architecture": "naive-rag",
            "question": question,
            "answer": answer,
            "generator": generator,
            "sources": sources,
            "trace": [
                {"step": "retrieve", "detail": f"One TF-IDF search using the original question; top_k={self.top_k}."},
                {"step": "construct_context", "detail": f"Inserted {len(results)} retrieved chunks into one prompt."},
                {"step": "generate", "detail": f"Generated the answer with the {generator} generator."},
            ],
        }

    def describe_documents(self) -> list[dict]:
        return [asdict(document) | {"text": document.text[:240] + ("…" if len(document.text) > 240 else "")} for document in self.documents]
