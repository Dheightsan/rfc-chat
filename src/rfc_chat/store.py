import sqlite3
from pathlib import Path

import numpy as np

from rfc_chat.embed import Embedder
from rfc_chat.text import Chunk


def build_index(db_path: str | Path, chunks: list[Chunk], embedder: Embedder) -> None:
    # Recreates SQLite database with chunks and their embedding blobs.
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        db_file.unlink()

    conn = sqlite3.connect(str(db_file))
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE chunks(
                    id INTEGER PRIMARY KEY,
                    rfc TEXT,
                    page INT,
                    text TEXT,
                    embedding BLOB
                )
                """
            )
            if chunks:
                embeddings = embedder.embed([c.text for c in chunks])
                rows = [
                    (c.rfc, c.page, c.text, embeddings[i].astype(np.float32).tobytes())
                    for i, c in enumerate(chunks)
                ]
                conn.executemany(
                    "INSERT INTO chunks(rfc, page, text, embedding) VALUES (?, ?, ?, ?)",
                    rows,
                )
    finally:
        conn.close()

def load(db_path: str | Path) -> tuple[list[Chunk], np.ndarray]:
    # Loads chunks and embedding matrix from SQLite database.
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT rfc, page, text, embedding FROM chunks ORDER BY id")
        rows = cursor.fetchall()
        chunks: list[Chunk] = []
        vecs: list[np.ndarray] = []
        for rfc, page, text, blob in rows:
            chunks.append(Chunk(rfc=rfc, page=page, text=text))
            vecs.append(np.frombuffer(blob, dtype=np.float32))
        matrix = np.vstack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
        return chunks, matrix
    finally:
        conn.close()
