# Semantic StackOverflow Search (FastAPI + FAISS)

A beginner-friendly end-to-end semantic search web app.

Users type a programming question (for example: `how to declare array in python`) and the app returns the most semantically similar StackOverflow-style Q&A.

---

## What is Semantic Search?

Keyword search matches exact words.

Semantic search matches **meaning** using embeddings (vectors), so it can return relevant results even with different wording.

---

## How this project works

1. `app/load_data.py` downloads the Hugging Face dataset and keeps only 3000 rows.
2. For each row we store:
   - `question`
   - `answer`
   - `embeddings`
3. On server startup, FastAPI loads `data/dataset.json` and builds a FAISS index.
4. A user query is embedded with `all-MiniLM-L6-v2`.
5. FAISS returns top 5 nearest neighbors by cosine similarity.
6. Results are rendered as cards in the HTML frontend.

---

## Project structure

```bash
project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── search.py
│   ├── load_data.py
│   └── utils.py
│
├── data/
│   └── dataset.json
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## Local setup and run

> Run all commands from inside `project/`.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Build dataset JSON from Hugging Face (recommended)

```bash
python -m app.load_data
```

This overwrites `data/dataset.json` with the first 3000 rows.

> Note: repository includes a tiny demo `data/dataset.json` so the app can start immediately.

### 3) Start FastAPI server

```bash
uvicorn app.main:app --reload
```

### 4) Open in browser

- http://127.0.0.1:8000

### 5) Optional health check

- http://127.0.0.1:8000/health

---

## API endpoints

### `GET /`
Serves the HTML page.

### `POST /search`
Request:

```json
{
  "query": "how to declare array in python"
}
```

Response:

```json
[
  {
    "question": "...",
    "answer": "...",
    "score": 0.8421
  }
]
```

### `GET /health`
Returns startup status and useful diagnostics.

---

## Deploy for free

## Option A: Render

1. Push repo to GitHub.
2. Create a **Web Service** in Render.
3. Set **Root Directory** to `project`.
4. Build Command:
   ```bash
   pip install -r requirements.txt && python -m app.load_data
   ```
5. Start Command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

## Option B: Railway

1. Create a new Railway project from GitHub.
2. Set root directory to `project`.
3. Start command:
   ```bash
   python -m app.load_data && uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

---

## Notes

- FAISS index is in-memory (no external database).
- Dataset embeddings are precomputed; this project does **not** recompute them.
- Query embedding uses sentence-transformers.
