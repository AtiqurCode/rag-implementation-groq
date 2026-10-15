from pathlib import Path

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.models import Answer, IngestReport, RetrievedChunk


class FakePipeline:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.ingested: list[Path] = []

    def stats(self) -> dict[str, int]:
        return {"data/pdfs/policy.pdf": 4}

    def ask(self, question, *, k=None, source=None, min_relevance=None):
        self.asked.append(question)
        return Answer(
            question=question,
            text="30 days.",
            citations=(
                RetrievedChunk(
                    source=source or "data/pdfs/policy.pdf",
                    page=3,
                    index=0,
                    text="Unused goods may be returned within 30 days.",
                    score=0.81234,
                ),
            ),
            answered_from_context=True,
        )

    def ingest_pdf(self, path: Path) -> IngestReport:
        self.ingested.append(path)
        return IngestReport(source=str(path).replace("\\", "/"), pages=2, empty_pages=0, chunks=3)

    def ingest_path(self, path: Path) -> list[IngestReport]:
        self.ingested.append(path)
        return [IngestReport(source="data/pdfs/policy.pdf", pages=2, empty_pages=0, chunks=3)]


def _client(tmp_path: Path) -> tuple[TestClient, FakePipeline]:
    pipeline = FakePipeline()
    app = create_app(pipeline, tmp_path)
    return TestClient(app), pipeline


def test_ask_returns_citations(tmp_path: Path) -> None:
    client, pipeline = _client(tmp_path)
    response = client.post("/api/ask", json={"question": "  refund window  "})
    assert response.status_code == 200
    body = response.json()
    assert pipeline.asked == ["refund window"]
    assert body["answer"] == "30 days."
    assert body["answered_from_context"] is True
    assert body["citations"][0]["page"] == 3
    assert body["citations"][0]["score"] == 0.8123


def test_empty_question_is_rejected(tmp_path: Path) -> None:
    client, _pipeline = _client(tmp_path)
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 400


def test_stats_and_upload(tmp_path: Path) -> None:
    client, pipeline = _client(tmp_path)
    stats = client.get("/api/stats")
    assert stats.json()["chunks"] == 4

    response = client.post(
        "/api/upload",
        files={"file": ("Q3 Policy.pdf", b"%PDF-1.7\n", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["reports"][0]["chunks"] == 3
    saved = tmp_path / "Q3-Policy.pdf"
    assert saved.is_file()
    assert pipeline.ingested == [saved]


def test_upload_rejects_non_pdf(tmp_path: Path) -> None:
    client, _pipeline = _client(tmp_path)
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_ingest_rejects_paths_outside_the_pdf_folder(tmp_path: Path) -> None:
    client, pipeline = _client(tmp_path)
    response = client.post("/api/ingest", json={"path": "../secrets"})
    assert response.status_code == 400
    assert pipeline.ingested == []


def test_ui_is_served() -> None:
    client, _pipeline = _client(Path("."))
    response = client.get("/")
    assert response.status_code == 200
    assert "Ask the documents" in response.text
