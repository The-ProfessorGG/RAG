"""Deployable dependency-free HTTP server for the Advanced RAG example."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import json
import os

from rag_core import AdvancedRAG


ROOT = Path(__file__).resolve().parent
RAG = AdvancedRAG(
    ROOT / "documents",
    top_k=int(os.environ.get("RAG_TOP_K", "4")),
    candidate_k=int(os.environ.get("RAG_CANDIDATE_K", "10")),
    score_threshold=float(os.environ.get("RAG_SCORE_THRESHOLD", "0.34")),
)

INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Advanced RAG Demo</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#10234a;background:#f4f7fb}body{margin:0}.wrap{max-width:1020px;margin:0 auto;padding:36px 20px}.panel{background:white;border:1px solid #d7e0ec;border-radius:18px;padding:24px;box-shadow:0 12px 35px #17355a12}h1{margin:0 0 6px}p{color:#5e6b82}.flow{color:#7256b8;font-weight:700;margin:20px 0}textarea{box-sizing:border-box;width:100%;min-height:96px;padding:14px;border:1px solid #b9c7d9;border-radius:12px;font:inherit}button{margin-top:12px;background:#7256b8;color:white;border:0;border-radius:10px;padding:12px 18px;font-weight:700;cursor:pointer}button:disabled{opacity:.55}.examples button{background:#eee8fb;color:#4e348c;margin:4px;padding:8px 10px}.result{display:none;margin-top:22px}.answer{background:#e8f7ef;border-left:5px solid #16865b;padding:16px;border-radius:10px;white-space:pre-wrap}.rewrite{background:#fff0c8;padding:12px;border-radius:10px}.source{border-top:1px solid #e2e8f0;padding:12px 0}.meta{font-size:.88rem;color:#5e6b82}.error{color:#a52c2c}.trace{font-family:ui-monospace,monospace;font-size:.88rem;background:#f6f8fb;padding:12px;border-radius:10px;white-space:pre-wrap}</style></head>
<body><main class="wrap"><section class="panel"><h1>Advanced RAG</h1><p>Query rewriting, hybrid retrieval, fusion, reranking, filtering, and compression over fictional policies.</p><div class="flow">Rewrite → TF-IDF + BM25 → RRF → rerank/filter → compress → answer</div>
<div class="examples"><button data-q="Can unused leave roll into next year, and is there a deadline?">Leave carry-over</button><button data-q="I think an account was breached. How fast must I tell security?">Security incident</button><button data-q="Do I need a receipt for a R250 taxi during a work trip?">Travel receipts</button></div>
<textarea id="q" aria-label="Question">Can unused leave roll into next year, and is there a deadline?</textarea><br><button id="ask">Run Advanced RAG</button><span id="status" class="meta"></span>
<div id="result" class="result"><h2>Rewritten retrieval query</h2><div id="rewrite" class="rewrite"></div><h2>Answer</h2><div id="answer" class="answer"></div><h2>Reranked and compressed sources</h2><div id="sources"></div><h2>Pipeline trace</h2><div id="trace" class="trace"></div></div></section></main>
<script>const q=document.querySelector('#q'),ask=document.querySelector('#ask'),status=document.querySelector('#status'),result=document.querySelector('#result');document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>q.value=b.dataset.q);ask.onclick=async()=>{ask.disabled=true;status.textContent=' Running pipeline…';result.style.display='none';try{const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})});const d=await r.json();if(!r.ok)throw new Error(d.error||'Request failed');document.querySelector('#rewrite').textContent=d.rewritten_query;document.querySelector('#answer').textContent=d.answer;const s=document.querySelector('#sources');s.replaceChildren(...d.sources.map(x=>{const e=document.createElement('div');e.className='source';const t=document.createElement('strong');t.textContent=`[${x.citation}] ${x.title} · rerank ${x.rerank_score}`;const p=document.createElement('div');p.className='meta';p.textContent=x.excerpt;e.append(t,p);return e}));document.querySelector('#trace').textContent=d.trace.map(x=>`${x.step}: ${x.detail}`).join('\n');result.style.display='block';status.textContent=` ${d.generator} generator`;}catch(e){status.textContent=' '+e.message;status.className='error'}finally{ask.disabled=false}};</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AdvancedRAGDemo/1.0"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict | list) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
        elif path == "/health":
            self._json(200, {"status": "ok", "architecture": "advanced-rag", "documents": len(RAG.documents), "chunks": len(RAG.chunks)})
        elif path == "/documents":
            self._json(200, RAG.describe_documents())
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/query":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Request body must be between 1 and 100000 bytes")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            if not question or len(question) > 2_000:
                raise ValueError("question must contain between 1 and 2000 characters")
            self._json(200, RAG.ask(question, payload.get("generator")))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Advanced RAG demo: http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
