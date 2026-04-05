# Semantic StackOverflow Search (FastAPI + FAISS)

A complete beginner-friendly semantic search web app.

Users type a programming question (example: `how to declare array in python`) and the app returns the most semantically similar StackOverflow Q&A.

---

## What is Semantic Search?

Normal keyword search checks exact words.

Semantic search checks **meaning** using vector embeddings. That means different wording can still find relevant answers.

---

## How this project works

1. `app/load_data.py` downloads data from:
   - `MartinElMolon/stackoverflow_preguntas_con_embeddings`
2. It keeps only the first **3000 rows**.
3. It saves only required fields to `data/dataset.json`:
   - `question`
   - `answer`
   - `embeddings`
4. FastAPI loads the JSON data and builds an in-memory FAISS index.
5. User query is embedded with `sentence-transformers/all-MiniLM-L6-v2`.
6. FAISS runs cosine similarity search (normalized vectors).
7. Top 5 matches are shown on the frontend.

---

## Project structure

```bash
project/
│
├── app/
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

## Run locally

> Run all commands from inside the `project/` directory.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Build dataset JSON (first 3000 rows)

```bash
python -m app.load_data
```

### 3) Start FastAPI server

```bash
uvicorn app.main:app --reload
```

### 4) Open in browser

- http://127.0.0.1:8000

---

## API endpoints

### `GET /`
Returns the HTML page.

### `POST /search`
Input:

```json
{
  "query": "how to declare array in python"
}
```

Output:

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

## Deploy (free)

## Option A: Render

1. Push code to GitHub.
2. Create a Render **Web Service**.
3. Set root directory to `project`.
4. Build command:
   ```bash
   pip install -r requirements.txt && python -m app.load_data
   ```
5. Start command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

## Option B: Railway

1. Create project from GitHub repo.
2. Set root directory to `project`.
3. Start command:
   ```bash
   python -m app.load_data && uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

---

## Notes

- Vector database is not used; FAISS index is in-memory.
- Dataset embeddings are precomputed; this project does not regenerate them.
- A tiny demo `data/dataset.json` is included so the app can boot immediately.
- For real results, run `python -m app.load_data` to fetch the first 3000 records.

---

## Screenshot placeholder

_Add a screenshot of the running UI here._
