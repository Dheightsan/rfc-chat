import json
from pathlib import Path
from typing import Any

from rfc_chat.retrieve import Index
from rfc_chat.text import norm


def load_questions(path: str | Path = "eval/questions.json") -> list[dict[str, Any]]:
    # Loads evaluation questions from JSON file.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run(
    index: Index,
    questions: list[dict[str, Any]],
    k: int = 3,
    method: str = "hybrid",
) -> list[dict[str, Any]]:
    # Runs evaluation against index and tests whether top-k chunks contain must_contain strings.
    results: list[dict[str, Any]] = []
    for q in questions:
        q_text = q["question"]
        must_contain = [norm(s) for s in q.get("must_contain", [])]
        search_res = index.search(query=q_text, k=k, method=method)

        hit = False
        for chunk, _ in search_res:
            chunk_norm = norm(chunk.text)
            if all(target in chunk_norm for target in must_contain):
                hit = True
                break

        results.append({
            "id": q["id"],
            "rfc": q.get("rfc"),
            "question": q_text,
            "hit": hit,
        })
    return results
