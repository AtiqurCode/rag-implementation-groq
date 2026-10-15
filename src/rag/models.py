from dataclasses import dataclass


@dataclass(frozen=True)
class PdfPage:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    index: int
    text: str


@dataclass(frozen=True)
class IndexedChunk:
    id: str
    source: str
    page: int
    index: int
    text: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    page: int
    index: int
    text: str
    score: float


@dataclass(frozen=True)
class IngestReport:
    source: str
    pages: int
    empty_pages: int
    chunks: int


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: tuple[RetrievedChunk, ...]
    answered_from_context: bool
