# Advanced RAG – Notebook and Deployable Service

This project improves the baseline pipeline with deliberate pre-retrieval and post-retrieval processing:

```text
Question
  → transparent query rewriting
  → TF-IDF + BM25 hybrid retrieval
  → reciprocal-rank fusion
  → heuristic reranking and filtering
  → sentence-level context compression
  → cited answer
```

The implementation is dependency-free and inspectable. Every score and intermediate stage is available in the API response.

## Included files

- `advanced_rag_demo.ipynb` – executable walkthrough of every pipeline stage.
- `rag_core.py` – query rewriting, two retrievers, fusion, reranking, filtering, compression, prompting, and generation.
- `server.py` – browser interface plus `/query`, `/documents`, and `/health` endpoints.
- `documents/` – eight synthetic Northstar Labs policies.
- `tests/` – standard-library tests.
- `Dockerfile` – portable deployment image.

## Run locally

```bash
python server.py
```

Open `http://localhost:8000`.

To run the notebook:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

Then open `advanced_rag_demo.ipynb` and run all cells.

Example API call:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Can unused leave roll into next year, and is there a deadline?"}'
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Run with OpenAI generation

The service runs offline by default. To use the OpenAI Responses API for the final generation stage:

```text
RAG_GENERATOR=openai
OPENAI_API_KEY=your-key-in-your-environment
OPENAI_MODEL=gpt-5.6-luna
```

Retrieval, fusion, reranking, filtering, and compression remain local. Only the final prompt containing selected evidence is sent to the model. Requests set `store` to `false`.

## Docker

```bash
docker build -t advanced-rag-demo .
docker run --rm -p 8000:8000 advanced-rag-demo
```

The main tuning variables are:

- `RAG_TOP_K` – final number of chunks, default `4`.
- `RAG_CANDIDATE_K` – candidates accepted from each retriever, default `10`.
- `RAG_SCORE_THRESHOLD` – reranking cutoff, default `0.34`.
- `RAG_GENERATOR` – `local` or `openai`.

## Deployment notes

The Docker image can be deployed to any container platform. Store API keys in the platform's secret manager.

For production use, replace deterministic rewriting and heuristic reranking when evaluation demonstrates a benefit. Add authentication, per-document permissions, ingestion jobs, durable index storage, observability, rate limits, and retrieval/answer evaluation.
