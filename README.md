# AI-Powered Legal Document Simplification and Guidance (India)

Production-style prototype with API-first and decoupled architecture.

## Architecture

- `backend/`: FastAPI app and HTTP routes.
- `src/core/`: config + logging.
- `src/models/`: Pydantic contracts for API I/O.
- `src/services/`: business logic (PDF parsing, clause segmentation, legal analysis, risk, RAG, LLM abstraction).
- `frontend/`: Streamlit chatbot-style UI.
- `vector_db/`: existing persistent Chroma DB.

## Key Features

- PDF parsing using `pdfplumber`.
- Scanned PDF OCR fallback using `pytesseract` + `PyMuPDF`.
- Clause segmentation by numbering + paragraph boundaries + token-safe chunking.
- Legal clause typing with `law-ai/CustomInLawBERT` (fallback heuristic if model load fails).
- Hybrid risk detection (rule + clause type signal).
- RAG retrieval from persistent Chroma using `BAAI/bge-small-en`.
- Query/document prefixes:
  - `search_query: `
  - `search_document: `
- LLM abstraction:
  - `ollama` via official `ollama` Python client
  - `mock` fallback
- FastAPI endpoints:
  - `POST /upload`
  - `POST /ask`
  - `GET /auth/google/login`
  - `GET /auth/google/callback`
  - `GET /auth/me`
  - `GET /health`
- Streamlit UI:
  - PDF upload + summary + risk view + chat.
  - Google sign-in with user-specific persisted chat history.

## Setup

1. Ensure Python 3.11 or 3.12 is active.
2. Install dependencies:

```bash
uv sync
```

3. Create environment file:

```bash
cp .env.example .env
```

4. Update `.env` values, especially:
- `CHROMA_COLLECTION` to match your existing `vector_db` collection name.
- `LEGAL_MODEL_LOCAL_PATH` if using a local model directory.
- `LLM_PROVIDER` (`ollama` recommended).
- `OLLAMA_BASE_URL` (for API usage, e.g. `https://ollama.com`).
- `OLLAMA_API_KEY` (required for hosted Ollama API).
- `OLLAMA_MODEL` (default: `gpt-oss:120b`).
- `TESSERACT_CMD` for scanned PDF OCR on Windows.
- `CHAT_DB_PATH` for local SQLite chat persistence path.
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `GOOGLE_REDIRECT_URI` for login.
  - Use backend callback URI, e.g. `http://localhost:8000/auth/google/callback`.
- `FRONTEND_BASE_URL` (e.g. `http://localhost:8501`) for post-login redirect.
- `BACKEND_PUBLIC_URL` (e.g. `http://localhost:8000`) for browser login link.
- `AUTH_SECRET_KEY` for signed auth token generation.

## Run

Backend:

```bash
uv run python main.py
```

Frontend:

```bash
uv run streamlit run frontend/app.py
```

## Docker

Build and run both services:

```bash
docker compose up --build
```

Access:
- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`

Notes:
- Source is bind-mounted (`.:/app`) so code changes reflect immediately.
- `vector_db` is intentionally included in Docker build context and mounted via project bind mount.
- In Docker, local Windows model paths are ignored for backend (`EMBEDDING_MODEL_LOCAL_PATH` and `LEGAL_MODEL_LOCAL_PATH` are overridden to empty), so models load by HF repo ID and are cached in Docker volume `hf_cache`.
- In Docker, Streamlit chat/database state is persisted in named volume `chat_db` mounted at `/app/misc`.

## API Contracts

### `POST /upload`
- Input: PDF file (`multipart/form-data`).
- Output:
  - `filename`
  - `total_clauses`
  - `summary`
  - `clauses[]`: `{clause_id, clause_text, clause_type, entities, confidence}`
  - `risks[]`: `{clause_id, clause_text, clause_type, risk_level, triggers}`

### `POST /ask`
- Input:
  - `question` (required)
  - `clause_text` (optional)
  - `top_k` (optional)
- Output:
  - `question`
  - `answer`
  - `retrieved_context[]`
  - `citations[]` (`citation`, `section`, `reference_type`, `score`)

### `POST /ask/stream`
- Input:
  - `question` (required)
  - `clause_text` (optional)
  - `top_k` (optional)
- Output:
  - Server-Sent Events style stream (`text/event-stream`) with event payload types:
    - `context`
    - `token`
    - `done`
    - `error`
