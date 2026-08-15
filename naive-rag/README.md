# Naive RAG – Notebook and Deployable Service

This project implements the baseline RAG pipeline:

```text
Question → one TF-IDF search → fixed top-k chunks → prompt → answer
```

It is intentionally simple. The original question is not rewritten, retrieval happens once, and the returned chunks are not reranked or compressed.

## Included files

- `naive_rag_demo.ipynb` – executable walkthrough.
- `rag_core.py` – document loading, chunking, TF-IDF retrieval, prompting, and generation.
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

Then open `naive_rag_demo.ipynb` and run all cells.

Example API call:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How many annual leave days may be carried over?"}'
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Run with OpenAI generation

The offline generator is the default. To use the OpenAI Responses API, set environment variables before starting the service:

```text
RAG_GENERATOR=openai
OPENAI_API_KEY=your-key-in-your-environment
OPENAI_MODEL=gpt-5.6-luna
```

The application performs retrieval itself and sends only the selected evidence to the model. It sets `store` to `false` on the request. Do not place a real key in `.env.example`, source control, the notebook, or the Docker image.

## Docker

```bash
docker build -t naive-rag-demo .
docker run --rm -p 8000:8000 naive-rag-demo
```

With OpenAI generation:

```bash
docker run --rm -p 8000:8000 \
  -e RAG_GENERATOR=openai \
  -e OPENAI_API_KEY \
  -e OPENAI_MODEL=gpt-5.6-luna \
  naive-rag-demo
```

## Deployment notes

The container can be deployed on any platform that accepts Docker images. Set `PORT`, `RAG_GENERATOR`, `OPENAI_MODEL`, and the secret `OPENAI_API_KEY` in the platform's environment configuration.

Before using real company policies, add authentication, authorization, document-level access controls, encrypted storage, ingestion validation, monitoring, rate limits, and a representative evaluation set.
