"""FastAPI entrypoint for semantic search web app."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from .search import SemanticSearchEngine


class SearchRequest(BaseModel):
    """Incoming payload for /search endpoint."""

    query: str = Field(..., min_length=2, description="User search query")


# Create app and wire folders.
app = FastAPI(title="Semantic StackOverflow Search")
base_dir = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

# Initialize search engine using exported JSON dataset.
engine = SemanticSearchEngine(data_path=base_dir / "data" / "dataset.json")


@app.on_event("startup")
def startup_event() -> None:
    """Build FAISS index once when server starts."""
    engine.load()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the search UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
def search(payload: SearchRequest) -> list[dict]:
    """Run semantic search and return top results as JSON."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return engine.search(query=query, top_k=5)
