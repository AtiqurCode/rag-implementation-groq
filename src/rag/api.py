import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.exceptions import RagError
from rag.factory import build_pipeline
from rag.models import Answer, IngestReport
from rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PDF_DIR = Path("data/pdfs")
MAX_PDF_BYTES = 32 * 1024 * 1024
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=20)
    source: str | None = None
    min_relevance: float | None = Field(default=None, ge=0.0, le=1.0)


class IngestBody(BaseModel):
    path: str = "data/pdfs"


def create_app(
    pipeline: RagPipeline | None = None,
    pdf_dir: Path | None = None,
) -> FastAPI:
    documents = pdf_dir or DEFAULT_PDF_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.getLogger("rag").setLevel(logging.INFO)
        if app.state.pipeline is None:
            app.state.pipeline = build_pipeline()
            app.state.pipeline._embedder.embed_query("ready")
        documents.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(title="vectordb-rag", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.pdf_dir = documents

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/stats")
    def stats() -> dict:
        sources = app.state.pipeline.stats()
        files = [
            {"source": source, "chunks": count}
            for source, count in sources.items()
        ]
        return {
            "files": files,
            "file_count": len(files),
            "chunks": sum(sources.values()),
        }

    @app.post("/api/ask")
    def ask(body: AskBody) -> dict:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is empty.")
        source = _clean_source(body.source)
        try:
            answer = app.state.pipeline.ask(
                question,
                k=body.k,
                source=source,
                min_relevance=body.min_relevance,
            )
        except RagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _answer_payload(answer)

    @app.post("/api/ingest")
    def ingest(body: IngestBody) -> dict:
        target = _resolve_pdf_path(documents, body.path)
        try:
            reports = app.state.pipeline.ingest_path(target)
        except RagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"reports": [_report_payload(report) for report in reports]}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        name = _pdf_filename(file.filename or "")
        payload = await file.read(MAX_PDF_BYTES + 1)
        if len(payload) > MAX_PDF_BYTES:
            raise HTTPException(status_code=400, detail="PDF is larger than 32 MB.")
        if not payload.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="That file is not a PDF.")
        documents.mkdir(parents=True, exist_ok=True)
        destination = documents / name
        destination.write_bytes(payload)
        try:
            report = app.state.pipeline.ingest_pdf(destination)
        except RagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"reports": [_report_payload(report)]}

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="UI is not installed.")
        return FileResponse(page)

    return app


def _answer_payload(answer: Answer) -> dict:
    return {
        "question": answer.question,
        "answer": answer.text,
        "answered_from_context": answer.answered_from_context,
        "citations": [
            {
                "source": citation.source,
                "page": citation.page,
                "score": round(citation.score, 4),
                "text": citation.text,
            }
            for citation in answer.citations
        ],
    }


def _report_payload(report: IngestReport) -> dict:
    return {
        "source": report.source,
        "pages": report.pages,
        "empty_pages": report.empty_pages,
        "chunks": report.chunks,
    }


def _clean_source(source: str | None) -> str | None:
    if source is None:
        return None
    cleaned = source.strip().replace("\\", "/")
    if not cleaned:
        return None
    if cleaned.startswith("/") or ".." in Path(cleaned).parts:
        raise HTTPException(status_code=400, detail="Source must be an indexed relative path.")
    return cleaned


def _pdf_filename(raw: str) -> str:
    name = Path(raw).name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file.")
    stem = _UNSAFE.sub("-", name[:-4]).strip(".-")
    if not stem:
        stem = "upload"
    return f"{stem}.pdf"


def _resolve_pdf_path(root: Path, raw: str) -> Path:
    cleaned = raw.strip().replace("\\", "/")
    if cleaned in {"", ".", "data/pdfs"}:
        return root.resolve()
    candidate = Path(cleaned)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="Path must stay inside the PDF folder.")
    if candidate.parts[:2] == ("data", "pdfs"):
        candidate = Path(*candidate.parts[2:]) if len(candidate.parts) > 2 else Path(".")
        if candidate == Path("."):
            return root.resolve()
    target = (root / candidate).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path must stay inside the PDF folder.") from exc
    return target


app = create_app()
