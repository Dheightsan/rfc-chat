import json
import os
import statistics
import time
from pathlib import Path

import httpx

from rfc_chat.guard import is_supported
from rfc_chat.llm import answer_with_usage
from rfc_chat.retrieve import Index


def measure(index: Index, questions: list[str], repeats: int = 3, k: int = 3) -> list[dict]:
    # Asks each question `repeats` times through the configured endpoint and records usage.
    rows = []
    with httpx.Client(timeout=90.0) as client:
        for q in questions:
            passages = [c for c, _ in index.search(q, k=k, method="hybrid")]
            if not is_supported(q, passages):
                continue
            for _ in range(repeats):
                start = time.monotonic()
                for attempt in range(5):
                    try:
                        _, usage = answer_with_usage(q, passages, client)
                        break
                    except httpx.HTTPStatusError as err:
                        if err.response.status_code != 429 or attempt == 4:
                            raise
                        time.sleep(20 * (attempt + 1))
                time.sleep(3)
                rows.append(
                    {
                        "question": q,
                        "seconds": round(time.monotonic() - start, 2),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "cost": float(usage.get("cost") or 0),
                    }
                )
    return rows


def table(rows: list[dict], model: str) -> str:
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["question"], []).append(r)
    lines = [
        f"Model: {model}",
        "",
        "| question | runs | input tokens | output tokens | seconds | cost USD |",
        "|---|---|---|---|---|---|",
    ]
    for q, rs in by_q.items():
        lines.append(
            f"| {q} | {len(rs)} | {statistics.mean(r['prompt_tokens'] for r in rs):.0f} "
            f"| {statistics.mean(r['completion_tokens'] for r in rs):.0f} "
            f"| {statistics.mean(r['seconds'] for r in rs):.1f} "
            f"| {statistics.mean(r['cost'] for r in rs):.5f} |"
        )
    if rows:
        lines.append(
            f"| mean of {len(rows)} calls | | "
            f"{statistics.mean(r['prompt_tokens'] for r in rows):.0f} "
            f"| {statistics.mean(r['completion_tokens'] for r in rows):.0f} "
            f"| {statistics.mean(r['seconds'] for r in rows):.1f} "
            f"| {statistics.mean(r['cost'] for r in rows):.5f} |"
        )
    return "\n".join(lines)


def run_cost(index: Index, questions_path: str, out_path: str, n: int = 10) -> str:
    model = os.environ.get("LLM_MODEL", "")
    if not (os.environ.get("LLM_BASE_URL") and model):
        raise SystemExit("Set LLM_BASE_URL and LLM_MODEL first (and LLM_API_KEY if needed).")
    questions = [q["question"] for q in json.loads(Path(questions_path).read_text("utf-8"))][:n]
    text = table(measure(index, questions), model)
    Path(out_path).write_text(text + "\n", encoding="utf-8")
    return text
