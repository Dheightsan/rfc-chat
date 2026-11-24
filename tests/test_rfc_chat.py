import json
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient

from rfc_chat.api import create_app
from rfc_chat.embed import HashingEmbedder
from rfc_chat.evalset import run
from rfc_chat.llm import answer
from rfc_chat.retrieve import Index
from rfc_chat.store import build_index, load
from rfc_chat.text import Chunk, chunk_pages, norm


def test_norm_ligature_whitespace_lowercase() -> None:
    # Verifies NFKC ligature unfolding, whitespace collapsing, and lowercase.
    raw = "  Deﬁne   SPECIﬁCATION  \n\t  Method\r\n  "
    assert norm(raw) == "define specification method"

def test_chunk_pages_boundaries() -> None:
    # Verifies chunks never cross page boundaries.
    pages = ["Page one content", "Page two content"]
    chunks = chunk_pages("9110", pages, size=100, overlap=20)
    assert len(chunks) == 2
    assert chunks[0].rfc == "9110" and chunks[0].page == 1
    assert chunks[0].text == "page one content"
    assert chunks[1].rfc == "9110" and chunks[1].page == 2
    assert chunks[1].text == "page two content"

def test_chunk_pages_overlap_and_word_boundaries() -> None:
    # Verifies splitting on word boundaries with overlap and skipping empty pages.
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    pages = ["   ", text]
    chunks = chunk_pages("9110", pages, size=35, overlap=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.page == 2
        assert len(chunk.text) <= 35
        # Verify no broken words
        words = chunk.text.split()
        for w in words:
            assert w.startswith("word")

def test_chunk_pages_ligature_normalization() -> None:
    # Verifies ligatures inside pages are normalized during chunking.
    pages = ["A speciﬁc classiﬁcation of header"]
    chunks = chunk_pages("9110", pages)
    assert len(chunks) == 1
    assert "specific classification" in chunks[0].text

def test_bm25_ranks_obvious_chunk_first() -> None:
    # Verifies BM25 ranks document with exact keyword match highest.
    chunks = [
        Chunk("9110", 1, "the 308 permanent redirect status code indicates a new uri"),
        Chunk("9110", 2, "quantum mechanics wave particle duality principle"),
        Chunk("9110", 3, "baking sourdough bread with flour water and yeast"),
    ]
    embedder = HashingEmbedder(dim=256)
    matrix = embedder.embed([c.text for c in chunks])
    index = Index(chunks, matrix, embedder)

    results = index.search("308 permanent redirect", k=1, method="bm25")
    assert len(results) == 1
    assert results[0][0].page == 1
    assert results[0][1] > 0

def test_store_round_trip(tmp_path: Path) -> None:
    # Verifies SQLite storage preserves chunk data and vector embeddings.
    db_file = tmp_path / "test_store.db"
    chunks = [
        Chunk("9110", 1, "first chunk text"),
        Chunk("9111", 5, "second chunk caching rules"),
    ]
    embedder = HashingEmbedder(dim=256)
    build_index(db_file, chunks, embedder)

    loaded_chunks, loaded_matrix = load(db_file)
    assert len(loaded_chunks) == len(chunks)
    assert loaded_chunks[0].rfc == "9110" and loaded_chunks[0].page == 1
    assert loaded_chunks[1].rfc == "9111" and loaded_chunks[1].page == 5
    assert loaded_matrix.shape == (2, 256)

    orig_embeddings = embedder.embed([c.text for c in chunks])
    assert np.allclose(loaded_matrix, orig_embeddings)

def test_hybrid_fusion_order() -> None:
    # Verifies reciprocal rank fusion combines dense and bm25 rankings.
    chunks = [
        Chunk("9110", 1, "alpha target keyword"),
        Chunk("9110", 2, "target keyword beta"),
        Chunk("9110", 3, "completely irrelevant gamma"),
    ]
    embedder = HashingEmbedder(dim=256)
    matrix = embedder.embed([c.text for c in chunks])
    index = Index(chunks, matrix, embedder)

    results = index.search("target keyword", k=3, method="hybrid")
    assert len(results) == 3
    # Both chunk 1 and 2 match keyword and should outrank chunk 3
    top_texts = [r[0].text for r in results[:2]]
    assert "completely irrelevant gamma" not in top_texts

def test_evalset_hit_miss_and_ligature() -> None:
    # Verifies hit/miss logic and ligature matching across questions.
    chunks = [
        Chunk("9110", 1, "a speciﬁc permanent redirect status"),
        Chunk("9110", 2, "caching considerations for intermediaries"),
    ]
    embedder = HashingEmbedder(dim=256)
    matrix = embedder.embed([c.text for c in chunks])
    index = Index(chunks, matrix, embedder)

    questions = [
        {
            "id": 1,
            "rfc": "9110",
            "question": "What is permanent redirect?",
            "must_contain": ["Specific", "Permanent Redirect"],
        },
        {
            "id": 2,
            "rfc": "9110",
            "question": "What is authentication?",
            "must_contain": ["OAuth Bearer Token"],
        },
    ]

    results = run(index, questions, k=1, method="bm25")
    assert results[0]["id"] == 1 and results[0]["hit"] is True
    assert results[1]["id"] == 2 and results[1]["hit"] is False

def test_api_endpoints_and_validation(tmp_path: Path) -> None:
    # Verifies /health, /ask success, and 422 on empty question.
    db_file = tmp_path / "api_test.db"
    chunks = [Chunk("9110", 3, "the 422 unprocessable content status code")]
    embedder = HashingEmbedder(dim=256)
    build_index(db_file, chunks, embedder)

    app = create_app(db_path=db_file, embedder=embedder)
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}

    empty_resp = client.post("/ask", json={"question": "   "})
    assert empty_resp.status_code == 422

    ask_resp = client.post("/ask", json={"question": "422 unprocessable", "k": 1, "method": "bm25"})
    assert ask_resp.status_code == 200
    data = ask_resp.json()
    assert "422 unprocessable" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["rfc"] == "9110"
    assert data["sources"][0]["page"] == 3

def test_llm_extractive_fallback() -> None:
    # Verifies fallback returns best passage text with citation.
    passages = [
        Chunk("9110", 12, "idempotent methods include put and delete"),
        Chunk("9110", 13, "other details"),
    ]
    ans = answer("Which methods are idempotent?", passages)
    assert "idempotent methods include put and delete" in ans
    assert "[RFC 9110 p.12]" in ans

def test_llm_path_with_mock_transport(monkeypatch) -> None:
    # Verifies OpenAI-compatible chat API call via httpx MockTransport.
    monkeypatch.setenv("LLM_BASE_URL", "https://mock.api/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "secret-token")

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://mock.api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret-token"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert "Answer only from the passages" in body["messages"][0]["content"]
        resp_data = {
            "choices": [
                {"message": {"content": "Idempotent methods are PUT and DELETE [9110 12]."}}
            ]
        }
        return httpx.Response(200, json=resp_data)

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as client:
        passages = [Chunk("9110", 12, "idempotent methods include put and delete")]
        ans = answer("Which methods are idempotent?", passages, client=client)
        assert ans == "Idempotent methods are PUT and DELETE [9110 12]."
