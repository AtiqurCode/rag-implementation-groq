from typing import Protocol

from rag.models import IndexedChunk, RetrievedChunk


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk text. Implementations apply the document task prefix."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question. Implementations apply the query task prefix."""


class VectorStore(Protocol):
    def replace_source(self, source: str, chunks: list[IndexedChunk]) -> None:
        """Drop previous chunks for this source, then insert the new ones."""

    def query(
        self,
        vector: list[float],
        k: int,
        source: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to k chunks, highest cosine similarity first."""

    def count(self, source: str | None = None) -> int:
        """Count chunks, optionally for one source label."""

    def sources(self) -> dict[str, int]:
        """Map source label to chunk count."""


class Generator(Protocol):
    def answer(self, question: str, passages: list[RetrievedChunk]) -> str:
        """Write an answer that is allowed to use only these passages."""
