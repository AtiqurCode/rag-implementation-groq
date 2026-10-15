import logging
from pathlib import Path

from rag.exceptions import RagError
from rag.ids import make_chunk_id
from rag.models import (
    Answer,
    Chunk,
    IndexedChunk,
    IngestReport,
    PdfPage,
    RetrievedChunk,
)
from rag.pdf_loader import list_pdfs, load_pdf, source_label
from rag.ports import Embedder, Generator, VectorStore
from rag.prompts import NOT_FOUND
from rag.text import chunk_text

logger = logging.getLogger(__name__)

EMPTY_INDEX = "The index is empty. Ingest a PDF first."


class RagPipeline:
    """Index PDF pages, then answer a question from the nearest chunks.

    Embeddings are computed before the old source is deleted. A failed Groq
    call leaves the previous index in place.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        *,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        min_relevance: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._min_relevance = min_relevance

    def ingest_path(self, path: Path) -> list[IngestReport]:
        pdfs = list_pdfs(path)
        if not pdfs:
            raise RagError(f"No PDF files under {path}")
        return [self.ingest_pdf(pdf) for pdf in pdfs]

    def ingest_pdf(self, path: Path) -> IngestReport:
        return self.ingest_pages(source_label(path), load_pdf(path))

    def ingest_pages(self, source: str, pages: list[PdfPage]) -> IngestReport:
        chunks, empty_pages = self._chunk_pages(source, pages)
        if not chunks:
            raise RagError(
                f"{source} has no extractable text. "
                "This loader does not OCR scanned pages."
            )

        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RagError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks."
            )

        indexed = [
            IndexedChunk(
                id=chunk.id,
                source=chunk.source,
                page=chunk.page,
                index=chunk.index,
                text=chunk.text,
                embedding=tuple(vector),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._store.replace_source(source, indexed)
        report = IngestReport(
            source=source,
            pages=len(pages),
            empty_pages=empty_pages,
            chunks=len(indexed),
        )
        logger.info(
            "indexed %s pages=%s empty_pages=%s chunks=%s",
            report.source,
            report.pages,
            report.empty_pages,
            report.chunks,
        )
        return report

    def ask(
        self,
        question: str,
        *,
        k: int | None = None,
        source: str | None = None,
        min_relevance: float | None = None,
    ) -> Answer:
        question = question.strip()
        if not question:
            raise RagError("Question is empty.")

        limit = k if k is not None else self._top_k
        if limit < 1:
            raise RagError("k must be >= 1")
        cutoff = self._min_relevance if min_relevance is None else min_relevance

        if self._store.count(source) == 0:
            text = (
                f"No indexed chunks for source '{source}'."
                if source
                else EMPTY_INDEX
            )
            return Answer(
                question=question,
                text=text,
                citations=(),
                answered_from_context=False,
            )

        hits = self._store.query(
            self._embedder.embed_query(question),
            limit,
            source,
        )
        strong = select_passages(hits, cutoff)
        if hits and not strong:
            logger.info(
                "best similarity %.2f is below cutoff %.2f; chat model was not called",
                max(hit.score for hit in hits),
                cutoff,
            )
        if not strong:
            return Answer(
                question=question,
                text=NOT_FOUND,
                citations=tuple(hits),
                answered_from_context=False,
            )

        return Answer(
            question=question,
            text=self._generator.answer(question, strong),
            citations=tuple(strong),
            answered_from_context=True,
        )

    def stats(self) -> dict[str, int]:
        return self._store.sources()

    def _chunk_pages(self, source: str, pages: list[PdfPage]) -> tuple[list[Chunk], int]:
        chunks: list[Chunk] = []
        empty_pages = 0
        for page in pages:
            pieces = chunk_text(page.text, self._chunk_size, self._chunk_overlap)
            if not pieces:
                empty_pages += 1
                continue
            for index, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        id=make_chunk_id(source, page.number, index, piece),
                        source=source,
                        page=page.number,
                        index=index,
                        text=piece,
                    )
                )
        return chunks, empty_pages


def select_passages(
    hits: list[RetrievedChunk],
    min_relevance: float,
) -> list[RetrievedChunk]:
    return [hit for hit in hits if hit.score >= min_relevance]
