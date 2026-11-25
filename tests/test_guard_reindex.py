import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from rfc_chat.api import create_app
from rfc_chat.embed import HashingEmbedder
from rfc_chat.guard import NO_ANSWER, coverage, is_supported
from rfc_chat.store import build_index, load, replace_document
from rfc_chat.text import Chunk

CHUNK = Chunk("9110", 7, "the 308 permanent redirect status code means the target moved for good")


def test_supported_question_passes_and_off_topic_is_refused() -> None:
    assert is_supported("What does the 308 status code mean?", [CHUNK])
    assert not is_supported("How do I bake sourdough bread?", [CHUNK])
    assert not is_supported("anything", [])


def test_coverage_ignores_stopwords_and_plural() -> None:
    assert coverage("What are the status codes?", [CHUNK]) == 1.0
    assert coverage("what is the", [CHUNK]) == 0.0


def test_api_says_it_does_not_know_and_gives_no_sources(tmp_path: Path) -> None:
    db = tmp_path / "i.db"
    emb = HashingEmbedder(dim=256)
    build_index(db, [CHUNK], emb)
    client = TestClient(create_app(db_path=db, embedder=emb))
    r = client.post("/ask", json={"question": "Who won the football world cup?", "k": 1})
    assert r.status_code == 200
    assert r.json() == {"answer": NO_ANSWER, "sources": []}


def test_api_answer_cites_the_source_chunk(tmp_path: Path) -> None:
    db = tmp_path / "i.db"
    emb = HashingEmbedder(dim=256)
    build_index(db, [CHUNK], emb)
    client = TestClient(create_app(db_path=db, embedder=emb))
    r = client.post("/ask", json={"question": "What does the 308 status code mean?", "k": 1})
    body = r.json()
    assert "[RFC 9110 p.7]" in body["answer"]
    assert body["sources"][0]["page"] == 7


def test_replace_document_does_not_duplicate(tmp_path: Path) -> None:
    db = tmp_path / "i.db"
    emb = HashingEmbedder(dim=256)
    other = Chunk("9111", 1, "caches store responses")
    build_index(db, [CHUNK, other], emb)
    new = [Chunk("9110", 1, "first new chunk"), Chunk("9110", 2, "second new chunk")]
    removed = replace_document(db, "9110", new, emb)
    assert removed == 1
    replace_document(db, "9110", new, emb)
    chunks, matrix = load(db)
    assert sorted((c.rfc, c.page) for c in chunks) == [("9110", 1), ("9110", 2), ("9111", 1)]
    assert matrix.shape[0] == 3
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM chunks WHERE rfc='9110'").fetchone()[0] == 2


def test_replace_document_creates_missing_db(tmp_path: Path) -> None:
    db = tmp_path / "new" / "i.db"
    assert replace_document(db, "9110", [CHUNK], HashingEmbedder(dim=256)) == 0
    assert len(load(db)[0]) == 1


def test_cost_table_averages_runs() -> None:
    from rfc_chat.cost import table

    rows = [
        {"question": "q", "seconds": 2.0, "prompt_tokens": 100, "completion_tokens": 10, "cost": 0},
        {"question": "q", "seconds": 4.0, "prompt_tokens": 200, "completion_tokens": 30, "cost": 0},
    ]
    out = table(rows, "m")
    assert "| q | 2 | 150 | 20 | 3.0 | 0.00000 |" in out
    assert "mean of 2 calls" in out


def test_answer_with_usage_returns_reported_usage(monkeypatch) -> None:
    import httpx

    from rfc_chat.llm import answer_with_usage

    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    text, usage = answer_with_usage("q", [CHUNK], client)
    assert text == "hi" and usage["prompt_tokens"] == 5
