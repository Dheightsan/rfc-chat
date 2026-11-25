import argparse
import json
import os
import sys
from pathlib import Path

import uvicorn

from rfc_chat.embed import FastEmbedder
from rfc_chat.evalset import load_questions, run
from rfc_chat.guard import is_supported
from rfc_chat.ingest import read_pdfs
from rfc_chat.retrieve import Index
from rfc_chat.store import build_index, load, replace_document
from rfc_chat.text import chunk_pages


def handle_ingest(args: argparse.Namespace) -> int:
    # Ingests PDF documents from corpus directory and builds index database.
    corpus_dir = Path(args.corpus_dir)
    db_path = Path(args.db_path)
    print(f"Ingesting PDFs from {corpus_dir}...")
    pdf_pages = read_pdfs(corpus_dir)
    embedder = FastEmbedder()
    if args.update:
        for rfc, pages in pdf_pages.items():
            removed = replace_document(db_path, rfc, chunk_pages(rfc, pages), embedder)
            print(f"RFC {rfc}: replaced {removed} chunks")
        return 0
    chunks = []
    for rfc, pages in pdf_pages.items():
        chunks.extend(chunk_pages(rfc, pages))
    build_index(db_path, chunks, embedder)
    print(f"Indexed {len(chunks)} chunks into {db_path}")
    return 0

def handle_eval(args: argparse.Namespace) -> int:
    # Evaluates retrieval accuracy against test questions.
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Database not found at {db_path}", file=sys.stderr)
        return 0

    chunks, matrix = load(db_path)
    embedder = FastEmbedder()
    index = Index(chunks, matrix, embedder)
    questions = load_questions(args.questions)

    methods = ["dense", "bm25", "hybrid"] if args.method == "all" else [args.method]
    for method in methods:
        results = run(index, questions, k=args.k, method=method)
        for r in results:
            status = "OK" if r["hit"] else "MISS"
            print(f"{r['id']} {status} {r['question']}")
        hits = sum(1 for r in results if r["hit"])
        print(f"Aciertos: {hits}/{len(questions)} (method={method}, k={args.k})")
    off_path = Path(args.unanswerable)
    if off_path.exists():
        off = json.loads(off_path.read_text(encoding="utf-8"))
        refused = 0
        for q in off:
            passages = [c for c, _ in index.search(q, k=args.k, method="hybrid")]
            ok = not is_supported(q, passages)
            refused += ok
            print(f"{'REFUSED' if ok else 'ANSWERED'} {q}")
        print(f"Rechazos correctos: {refused}/{len(off)} (preguntas fuera del corpus)")
    return 0

def handle_serve(args: argparse.Namespace) -> int:
    # Starts Uvicorn server for the FastAPI application.
    uvicorn.run("rfc_chat.api:create_app", factory=True, host=args.host, port=args.port)
    return 0

def main() -> None:
    # Main CLI entrypoint for rfc-chat.
    default_db = os.environ.get("RFC_CHAT_DB", "data/index.db")
    parser = argparse.ArgumentParser(prog="rfc-chat", description="RFC RAG Chat CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest RFC PDFs into SQLite index")
    ingest_parser.add_argument("--corpus-dir", default="corpus", help="Path to corpus directory")
    ingest_parser.add_argument("--db-path", default=default_db, help="Path to SQLite index db")
    ingest_parser.add_argument(
        "--update",
        action="store_true",
        help="Replace the chunks of each PDF in place instead of rebuilding the index",
    )
    ingest_parser.set_defaults(func=handle_ingest)

    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval accuracy")
    eval_parser.add_argument("--k", type=int, default=3, help="Top-k retrieval count")
    eval_parser.add_argument(
        "--method",
        choices=["dense", "bm25", "hybrid", "all"],
        default="hybrid",
        help="Retrieval method",
    )
    eval_parser.add_argument(
        "--questions",
        default="eval/questions.json",
        help="Path to questions file",
    )
    eval_parser.add_argument(
        "--unanswerable",
        default="eval/unanswerable.json",
        help="Questions the documents cannot answer",
    )
    eval_parser.add_argument("--db-path", default=default_db, help="Path to SQLite index db")
    eval_parser.set_defaults(func=handle_eval)

    serve_parser = subparsers.add_parser("serve", help="Run API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.set_defaults(func=handle_serve)

    parsed = parser.parse_args()
    code = parsed.func(parsed)
    sys.exit(code if code is not None else 0)

if __name__ == "__main__":
    main()
