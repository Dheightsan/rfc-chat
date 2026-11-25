from rfc_chat.retrieve import tokenize
from rfc_chat.text import Chunk

NO_ANSWER = "I don't know. The documents I have don't cover this."

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "did", "for", "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "should", "that", "the", "their", "there", "this", "to", "use", "used", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your"]
)


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def content_terms(query: str) -> set[str]:
    return {_stem(t) for t in tokenize(query) if t not in STOPWORDS}


def coverage(query: str, passages: list[Chunk]) -> float:
    # Share of the question's content words that show up in the retrieved passages.
    terms = content_terms(query)
    if not terms:
        return 0.0
    found = {_stem(t) for p in passages for t in tokenize(p.text)}
    return len(terms & found) / len(terms)


def is_supported(query: str, passages: list[Chunk], min_coverage: float = 0.6) -> bool:
    return bool(passages) and coverage(query, passages) >= min_coverage
