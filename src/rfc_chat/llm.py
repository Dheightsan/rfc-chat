import os

import httpx

from rfc_chat.text import Chunk


def answer(
    question: str,
    passages: list[Chunk],
    client: httpx.Client | None = None,
) -> str:
    # Answers question using OpenAI-compatible chat API or extractive fallback.
    if not passages:
        return "I don't know."

    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")

    if base_url and model:
        api_key = os.environ.get("LLM_API_KEY", "")
        clean_base = base_url.rstrip("/")
        endpoint = f"{clean_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        system_prompt = (
            "Answer only from the passages and cite them as [rfc page]. "
            "Say you don't know if they don't contain it."
        )
        context_parts = [f"[RFC {p.rfc} p.{p.page}]: {p.text}" for p in passages]
        context = "\n\n".join(context_parts)
        user_prompt = f"Passages:\n{context}\n\nQuestion: {question}"

        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if client is not None:
            resp = client.post(endpoint, json=payload, headers=headers, timeout=60.0)
        else:
            with httpx.Client(timeout=60.0) as http_client:
                resp = http_client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])

    best = passages[0]
    return f"{best.text}\n\n[RFC {best.rfc} p.{best.page}]"
