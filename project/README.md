# Semantic StackOverflow Search (FastAPI + FAISS)

A complete beginner-friendly semantic search web application.

Users can type a programming question (for example: `how to declare array in python`) and the app finds the most semantically similar StackOverflow Q&A from a pre-embedded dataset.

---

## What is Semantic Search?

Traditional keyword search matches exact words.

**Semantic search** matches meaning.

So even if your query uses different wording, this app can still find relevant Q&A by comparing vector embeddings.

---

## How This Project Works

1. We load StackOverflow records from Hugging Face (`question`, `answer`, `embeddings`).
2. We keep only the first 3000 rows and save them to `data/dataset.json`.
3. On app startup:
   - JSON data is loaded.
   - Embeddings are normalized.
   - A FAISS cosine-similarity index is built in memory.
4. User submits a query in the UI.
5. Query is embedded using `sentence-transformers/all-MiniLM-L6-v2`.
6. FAISS retrieves top 5 most similar results.
7. Results are shown as cards in the frontend.

---

## Project Structure

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

## Setup & Run (Local)

> Run commands from inside the `project/` directory.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Build dataset JSON (first 3000 rows)

```bash
python -m app.load_data
```

This creates/overwrites:

- `data/dataset.json`

### 3) Start FastAPI server

```bash
uvicorn app.main:app --reload
```

### 4) Open in browser

- http://127.0.0.1:8000

---

## API Endpoints

### `GET /`
Returns the HTML page.

### `POST /search`
Request body:

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

---

## Deployment (Free)

## Option A: Render

1. Push this project to GitHub.
2. On Render, create **Web Service** from repo.
3. Build Command:
   ```bash
   pip install -r requirements.txt
   python -m app.load_data
   ```
4. Start Command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
5. Root directory: `project`

## Option B: Railway

1. Create new Railway project from GitHub repo.
2. Set root directory to `project`.
3. Railway auto-installs dependencies from `requirements.txt`.
4. Set start command:
   ```bash
   python -m app.load_data && uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

---

## Notes

- This project uses precomputed embeddings from the dataset. No embedding generation for dataset rows is needed.
- Query embedding is generated at runtime using `all-MiniLM-L6-v2`.
- FAISS index is in-memory (no external database).

---

## Screenshot Placeholder

_Add screenshot of the search page here after running locally._
