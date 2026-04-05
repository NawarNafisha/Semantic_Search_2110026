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


app = FastAPI(title="Semantic StackOverflow Search")
base_dir = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

engine = SemanticSearchEngine(data_path=base_dir / "data" / "dataset.json")
startup_error: str | None = None


@app.on_event("startup")
def startup_event() -> None:
    """Build FAISS index on server startup.

    Startup failures are stored and surfaced through API responses instead of
    crashing the whole server process.
    """
    global startup_error
    try:
        engine.load()
        startup_error = None
    except Exception as exc:  # Keep the server alive with useful error details.
        startup_error = str(exc)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the search UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health endpoint for quick diagnostics."""
    if startup_error:
        return {"status": "degraded", "detail": startup_error}
    return {"status": "ok"}


@app.post("/search")
def search(payload: SearchRequest) -> list[dict]:
    """Run semantic search and return top results as JSON."""
    if startup_error:
        raise HTTPException(status_code=503, detail=startup_error)

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        return engine.search(query=query, top_k=5)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
