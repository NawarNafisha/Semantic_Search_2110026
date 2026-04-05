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
    """Request body model for /search endpoint."""

    # We require at least 2 characters to avoid empty/noisy requests.
    query: str = Field(..., min_length=2, description="User search query")


# Create the FastAPI app.
app = FastAPI(title="Semantic StackOverflow Search")

# Resolve folders relative to project root.
base_dir = Path(__file__).resolve().parents[1]

# Configure template and static folders.
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

# Create semantic search engine (dataset path can be replaced by load_data.py output).
engine = SemanticSearchEngine(data_path=base_dir / "data" / "dataset.json")

# Store startup errors so the server stays alive and returns clear API messages.
startup_error: str | None = None


@app.on_event("startup")
def startup_event() -> None:
    """Load dataset and build FAISS index once at app startup."""
    global startup_error
    try:
        engine.load()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Serve the HTML frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
def search(payload: SearchRequest) -> list[dict]:
    """Run semantic search and return top 5 results as JSON."""
    if startup_error:
        # If startup failed, return a helpful message to the UI.
        raise HTTPException(status_code=503, detail=startup_error)

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        return engine.search(query=query, top_k=5)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
