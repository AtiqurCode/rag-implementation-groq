import chromadb
import pytest
from chromadb.config import Settings

from rag.models import IndexedChunk
from rag.store import ChromaStore, cosine_distance_to_similarity


def test_cosine_distance_conversion() -> None:
    assert cosine_distance_to_similarity(0.0) == 1.0
    assert cosine_distance_to_similarity(0.25) == 0.75
    assert cosine_distance_to_similarity(2.0) == 0.0


def test_chroma_returns_the_closer_chunk_and_replaces_a_source(tmp_path) -> None:
    pytest.importorskip("chromadb")
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    store = ChromaStore(tmp_path, "pdf_rag", "nomic-embed-text-v1_5", client=client)

    def chunk(text: str, vector: tuple[float, float], index: int) -> IndexedChunk:
        return IndexedChunk(
            id=f"id-{index}",
            source="data/pdfs/policy.pdf",
            page=1,
            index=index,
            text=text,
            embedding=vector,
        )

    store.replace_source(
        "data/pdfs/policy.pdf",
        [
            chunk("Refunds are allowed within 30 days.", (1.0, 0.0), 0),
            chunk("The office closes at five.", (0.0, 1.0), 1),
        ],
    )
    hits = store.query([1.0, 0.0], k=2)
    assert hits[0].text.startswith("Refunds")
    assert hits[0].score > hits[1].score
    assert store.count() == 2
    assert store.sources() == {"data/pdfs/policy.pdf": 2}

    store.replace_source(
        "data/pdfs/policy.pdf",
        [chunk("Only one chunk remains.", (1.0, 0.0), 0)],
    )
    assert store.count("data/pdfs/policy.pdf") == 1
    assert store.query([0.0, 1.0], k=1, source="missing.pdf") == []


def test_embed_model_change_is_refused(tmp_path) -> None:
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    ChromaStore(tmp_path, "pdf_rag", "nomic-embed-text-v1_5", client=client)
    with pytest.raises(Exception, match="embedding model"):
        ChromaStore(tmp_path, "pdf_rag", "some-other-model", client=client)
