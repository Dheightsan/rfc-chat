import re
from pathlib import Path

from pypdf import PdfReader


def read_pdfs(dir_path: str | Path) -> dict[str, list[str]]:
    # Extracts text pages from RFC PDFs named like rfc9110.pdf.
    pdf_dir = Path(dir_path)
    result: dict[str, list[str]] = {}
    for file_path in sorted(pdf_dir.glob("*.pdf")):
        match = re.search(r"(?:rfc)?(\d+)", file_path.name, re.IGNORECASE)
        rfc_id = match.group(1) if match else file_path.stem
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        result[rfc_id] = pages
    return result
