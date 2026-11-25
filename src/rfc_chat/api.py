import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from rfc_chat.embed import Embedder, FastEmbedder
from rfc_chat.guard import NO_ANSWER, is_supported
from rfc_chat.llm import answer
from rfc_chat.retrieve import Index
from rfc_chat.store import load


class AskRequest(BaseModel):
    # Request schema for /ask endpoint.
    question: str
    k: int = 3
    method: Literal["dense", "bm25", "hybrid"] = "hybrid"

    @field_validator("question")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        # Ensures question is not empty or whitespace only.
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v

class SourceItem(BaseModel):
    # Citation source chunk representation.
    rfc: str
    page: int
    text: str
    score: float

class AskResponse(BaseModel):
    # Response schema for /ask endpoint.
    answer: str
    sources: list[SourceItem]

def create_app(
    db_path: str | Path | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    # FastAPI application factory for RFC chat.
    actual_db_path = Path(db_path or os.environ.get("RFC_CHAT_DB", "data/index.db"))
    app = FastAPI(title="RFC Chat API")
    active_embedder = embedder if embedder is not None else FastEmbedder()

    if actual_db_path.exists():
        chunks, matrix = load(actual_db_path)
        app.state.index = Index(chunks, matrix, active_embedder)
    else:
        app.state.index = None

    @app.get("/health")
    def health() -> dict[str, str]:
        # Service health check endpoint.
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        # Query the RAG index and return answer with sources.
        if app.state.index is None:
            if actual_db_path.exists():
                chunks, matrix = load(actual_db_path)
                app.state.index = Index(chunks, matrix, active_embedder)
            else:
                raise HTTPException(status_code=503, detail="Index not initialized")

        results = app.state.index.search(query=req.question, k=req.k, method=req.method)
        passages = [chunk for chunk, _ in results]
        if not is_supported(req.question, passages):
            return AskResponse(answer=NO_ANSWER, sources=[])
        ans = answer(req.question, passages)
        sources = [
            SourceItem(rfc=chunk.rfc, page=chunk.page, text=chunk.text, score=score)
            for chunk, score in results
        ]
        return AskResponse(answer=ans, sources=sources)

    return app
