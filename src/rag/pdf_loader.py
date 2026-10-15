import logging
from pathlib import Path

from rag.exceptions import RagError
from rag.models import PdfPage
from rag.text import normalize_text

logger = logging.getLogger(__name__)


def source_label(path: Path) -> str:
    """Path relative to the working directory, so citations stay readable and unique."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def list_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise RagError(f"Path not found: {path}")
    if not path.is_dir():
        raise RagError(f"Not a file or directory: {path}")
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() == ".pdf"
        and not any(part.startswith(".") for part in item.relative_to(path).parts)
    )


def load_pdf(path: Path) -> list[PdfPage]:
    if not path.is_file():
        raise RagError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise RagError(f"Not a PDF: {path}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RagError("pypdf is not installed. Run: pip install -r requirements.txt") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RagError(f"Could not read {path.name}: {exc}") from exc

    if reader.is_encrypted:
        raise RagError(f"{path.name} is encrypted. Decrypt it before ingest.")

    pages: list[PdfPage] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning("skipped unreadable page %s of %s: %s", number, path.name, exc)
            raw = ""
        pages.append(PdfPage(number=number, text=normalize_text(raw)))
    return pages
