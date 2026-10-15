import pytest

from rag.exceptions import RagError
from rag.models import IndexedChunk, PdfPage, RetrievedChunk
from rag.pipeline import EMPTY_INDEX, RagPipeline
from rag.prompts import NOT_FOUND


class FakeEmbedder:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, hits: list[RetrievedChunk] | None = None, count: int = 1) -> None:
        self.hits = hits or []
        self._count = count
        self.replaced: list[IndexedChunk] = []

    def replace_source(self, source: str, chunks: list[IndexedChunk]) -> None:
        self.replaced = chunks

    def query(self, vector, k, source=None):
        return self.hits[:k]

    def count(self, source=None) -> int:
        return self._count

    def sources(self) -> dict[str, int]:
        return {"policy.pdf": self._count}


class FakeGenerator:
    def __init__(self) -> None:
        self.called = False

    def answer(self, question: str, passages: list[RetrievedChunk]) -> str:
        self.called = True
        return f"Refunds take 30 days. [1] ({len(passages)} passages)"


def _pipeline(store, generator, embedder=None) -> RagPipeline:
    return RagPipeline(
        embedder or FakeEmbedder(),
        store,
        generator,
        chunk_size=40,
        chunk_overlap=10,
        top_k=4,
        min_relevance=0.5,
    )


def test_ingest_embeds_before_replace_and_keeps_pages() -> None:
    events: list[str] = []

    class TrackingEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            events.append("embed")
            return super().embed_documents(texts)

    class TrackingStore(FakeStore):
        def replace_source(self, source: str, chunks: list[IndexedChunk]) -> None:
            events.append("replace")
            super().replace_source(source, chunks)

    store = TrackingStore()
    pipeline = _pipeline(store, FakeGenerator(), TrackingEmbedder())
    report = pipeline.ingest_pages(
        "data/pdfs/policy.pdf",
        [
            PdfPage(1, "alpha " * 30),
            PdfPage(2, ""),
            PdfPage(3, "Refunds are allowed within 30 days."),
        ],
    )

    assert events == ["embed", "replace"]
    assert report.empty_pages == 1
    assert report.chunks == len(store.replaced)
    assert {chunk.page for chunk in store.replaced} == {1, 3}
    assert all(chunk.source == "data/pdfs/policy.pdf" for chunk in store.replaced)


def test_blank_pdf_does_not_touch_the_index() -> None:
    store = FakeStore()
    pipeline = _pipeline(store, FakeGenerator())
    with pytest.raises(RagError, match="no extractable text"):
        pipeline.ingest_pages("scan.pdf", [PdfPage(1, "  ")])
    assert store.replaced == []


def test_ask_skips_the_model_when_nothing_clears_the_cutoff() -> None:
    generator = FakeGenerator()
    store = FakeStore(
        hits=[RetrievedChunk("policy.pdf", 1, 0, "unrelated", 0.2)],
        count=1,
    )
    answer = _pipeline(store, generator).ask("What is the refund window?")
    assert answer.text == NOT_FOUND
    assert answer.answered_from_context is False
    assert generator.called is False
    assert answer.citations[0].score == 0.2


def test_ask_sends_only_passages_above_the_cutoff() -> None:
    generator = FakeGenerator()
    store = FakeStore(
        hits=[
            RetrievedChunk("policy.pdf", 2, 0, "Refunds are allowed within 30 days.", 0.9),
            RetrievedChunk("policy.pdf", 8, 3, "Office closes at five.", 0.2),
        ],
        count=2,
    )
    answer = _pipeline(store, generator).ask("What is the refund window?")
    assert answer.answered_from_context is True
    assert "30 days" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].page == 2


def test_empty_index_does_not_call_groq() -> None:
    generator = FakeGenerator()
    embedder = FakeEmbedder()
    answer = _pipeline(FakeStore(count=0), generator, embedder).ask("anything")
    assert answer.text == EMPTY_INDEX
    assert embedder.queries == []
    assert generator.called is False
