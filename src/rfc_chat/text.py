import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Chunk:
    # Chunk metadata and text representation.
    rfc: str
    page: int
    text: str

def norm(s: str) -> str:
    # NFKC normalization, collapsed whitespace, lowercase.
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def chunk_pages(
    rfc: str,
    pages: list[str],
    size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    # Splits pages into overlapping chunks on word boundaries without crossing pages.
    chunks: list[Chunk] = []
    for page_num, raw_page in enumerate(pages, start=1):
        page_text = norm(raw_page)
        if not page_text:
            continue
        if len(page_text) <= size:
            chunks.append(Chunk(rfc=rfc, page=page_num, text=page_text))
            continue

        start = 0
        total_len = len(page_text)
        while start < total_len:
            end = start + size
            if end >= total_len:
                chunk = page_text[start:].strip()
                if chunk:
                    chunks.append(Chunk(rfc=rfc, page=page_num, text=chunk))
                break

            split_idx = page_text.rfind(" ", start, end + 1)
            if split_idx <= start:
                split_idx = end
            chunk = page_text[start:split_idx].strip()
            if chunk:
                chunks.append(Chunk(rfc=rfc, page=page_num, text=chunk))

            target_start = max(start + 1, split_idx - overlap)
            if target_start < split_idx and target_start > 0 and page_text[target_start - 1] != " ":
                next_space = page_text.find(" ", target_start, split_idx)
                target_start = next_space + 1 if next_space != -1 else split_idx
            start = target_start

    return chunks
