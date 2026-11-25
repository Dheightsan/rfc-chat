# rfc-chat

I made this to have an applied AI project with a number I can check. It answers questions about six RFC PDFs (HTTP semantics and caching, HTTP message signatures, DPoP, the OAuth security best practices and the JWT access token profile) and shows the pages it used. I picked RFCs because the answer is in the text, so a question can be graded without asking another model.

Stack is Python, FastAPI, SQLite, fastembed (bge-small-en-v1.5), pypdf and pytest.

## How it works

`ingest` reads the PDFs in `corpus/`, cuts each page into chunks of about 900 characters with 150 of overlap and stores them with their embeddings in SQLite. A question is searched three ways: by embeddings (cosine), by BM25 that I wrote, or by both joined with reciprocal rank fusion. `POST /ask` returns the best passages with RFC and page. If `LLM_BASE_URL` and `LLM_MODEL` are set (any OpenAI compatible endpoint, `LLM_API_KEY` optional) the passages go to that model (temperature 0, told to say it doesn't know if the passages don't cover it) to write the answer. If not, the answer is the best passage itself.

## Run it

```bash
uv sync
uv run rfc-chat ingest
uv run rfc-chat serve
curl -s localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What does the 308 status code mean?"}'
```

The first `ingest` downloads the embedding model (about 100 MB). To change a PDF later, run `uv run rfc-chat ingest --update`: it swaps the chunks of each document and leaves the rest, so running it twice gives the same 1468 chunks.

## The evaluation

`eval/questions.json` has 20 questions, each with a phrase that has to appear in the retrieved text. A question counts as a hit if one of the top 3 chunks contains that phrase.

```bash
uv run rfc-chat eval --method all
```

What I got:

| method | hits out of 20 |
|---|---|
| embeddings | 12 |
| BM25 | 16 |
| hybrid (default) | 16 |

To reproduce: `uv run rfc-chat ingest` and then the command above. I did not tune anything after seeing these numbers, except for one check: chunks of about 2000 characters (roughly 500 tokens) gave 14, 14 and 15, so I kept 900.

BM25 beats the small embedding model because many questions reuse the exact words of the RFC. The hybrid does not beat BM25 here. Fusion gives the weaker embedding ranking the same weight, and on question 7 it pushed out a passage that both methods had found alone, while it rescued question 9. With 20 questions I can't say which is better in general.

### When it doesn't know

Before answering, the chat checks that the retrieved passages contain at least 60 % of the question's content words (stopwords out). If not, it says it doesn't know and returns no sources. Otherwise the answer carries its citation, `[RFC 9110 p.7]` style. `eval/unanswerable.json` has 10 questions the RFCs can't answer, and `rfc-chat eval` prints how many get refused:

- 8 of 10 are refused. The 6 off-topic ones and 2 of the 4 that use RFC vocabulary (TLS 1.3 ciphers, SHA-3).
- The device grant and HTTP/3 frames questions still get an answer, because the words appear in the corpus even though the content isn't there. A word check can't tell those apart. A model that reads the passages would have to.
- None of the 20 real questions is refused. This was checked in the same run, not on separate data.

### Cost per query

`uv run rfc-chat cost` asks the first 10 evaluation questions 3 times each through the endpoint set in `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`, and writes the table to `eval/cost.md`. I ran it on `nvidia/nemotron-3.5-lightning:free` through OpenRouter. Averages over 30 calls: 704 input tokens, 1456 output tokens, 10 seconds, and 0 USD because the model is free. The output is large because that model writes its reasoning before the answer. The token counts are the number to carry over to a paid model, and the price I did not measure. A first run on another free model stopped on rate limits, which is why the script waits and retries.

## What I tested and what I didn't

The 19 unit tests pass. They use a hashing embedder, so they need no network or model. The CI runs only ruff and the tests, because the evaluation needs the embedding model download. The CI has not run yet, so GitHub's first run is its first test. The numbers above come from one run on my machine.

- The eval measures retrieval only. It does not check the written answer, so the optional LLM path has been tested with a mocked HTTP client and once against a free model through the cost command, and the answers it wrote were not graded.
- I wrote the questions and picked the phrases while reading the extracted text, so they favour wording that is in the chunks. It is 20 questions, enough to compare methods roughly and not to claim a general accuracy.
- A chunk is cut at 900 characters, so a phrase that falls across a cut is a miss even if the right passage was found.
- I have not run the API under load, and there is no auth.

## Limits

- pypdf drops the bold keywords of the RFCs (MUST, SHOULD NOT) and some code font terms. Questions that depend on them cannot be answered from this text, so I avoided them in the eval. That is a real gap for a chat about standards.
- English only.
- Tables and figures are not read as such.

## License

The RFCs in `corpus/` are copyright of the IETF Trust and are used under its legal provisions, which allow copying them unmodified. My code is MIT.
