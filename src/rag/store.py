import logging
from collections import Counter
from pathlib import Path

from rag.exceptions import RagError
from rag.models import IndexedChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 200
_DELETE_BATCH = 500


def cosine_distance_to_similarity(distance: float) -> float:
    """Chroma cosine space stores distance as 1 - cosine similarity."""
    similarity = 1.0 - float(distance)
    if similarity < 0.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return similarity


class ChromaStore:
    def __init__(
        self,
        path: Path,
        collection_name: str,
        embed_model: str,
        client=None,
    ) -> None:
        self._path = path
        self._collection_name = collection_name
        self._embed_model = embed_model
        self._client = client if client is not None else _persistent_client(path)
        self._collection = self._open_collection()

    def replace_source(self, source: str, chunks: list[IndexedChunk]) -> None:
        if not chunks:
            raise RagError(f"Refusing to replace {source} with zero chunks.")
        self._delete_source(source)

        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            self._collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=[list(chunk.embedding) for chunk in batch],
                metadatas=[
                    {
                        "source": chunk.source,
                        "page": chunk.page,
                        "chunk_index": chunk.index,
                    }
                    for chunk in batch
                ],
            )
        logger.info("stored %s chunks for %s", len(chunks), source)

    def query(
        self,
        vector: list[float],
        k: int,
        source: str | None = None,
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be >= 1")
        available = self.count(source)
        if available == 0:
            return []

        where = {"source": source} if source else None
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(k, available),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not document or not metadata:
                continue
            hits.append(
                RetrievedChunk(
                    source=str(metadata["source"]),
                    page=int(metadata["page"]),
                    index=int(metadata["chunk_index"]),
                    text=document,
                    score=cosine_distance_to_similarity(distance),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def count(self, source: str | None = None) -> int:
        if source is None:
            return int(self._collection.count())
        found = self._collection.get(where={"source": source}, include=[])
        return len(found["ids"])

    def sources(self) -> dict[str, int]:
        found = self._collection.get(include=["metadatas"])
        counts: Counter[str] = Counter()
        for metadata in found.get("metadatas") or []:
            if metadata and metadata.get("source"):
                counts[str(metadata["source"])] += 1
        return dict(sorted(counts.items()))

    def _delete_source(self, source: str) -> None:
        while True:
            found = self._collection.get(
                where={"source": source},
                limit=_DELETE_BATCH,
                include=[],
            )
            ids = found["ids"]
            if not ids:
                return
            self._collection.delete(ids=ids)

    def _open_collection(self):
        metadata = {"hnsw:space": "cosine", "embed_model": self._embed_model}
        names = _collection_names(self._client)
        if self._collection_name not in names:
            return self._client.create_collection(
                name=self._collection_name,
                metadata=metadata,
            )

        collection = self._client.get_collection(name=self._collection_name)
        recorded = (collection.metadata or {}).get("embed_model")
        space = (collection.metadata or {}).get("hnsw:space")
        if recorded and recorded != self._embed_model:
            raise RagError(
                f"Collection '{self._collection_name}' was built with embedding model "
                f"'{recorded}', but this process is using '{self._embed_model}'. "
                f"Delete {self._path} and ingest again. Mixed vectors are not comparable."
            )
        if space and space != "cosine":
            raise RagError(
                f"Collection '{self._collection_name}' uses '{space}' distance, not cosine. "
                f"Delete {self._path} and ingest again."
            )
        return collection


def _persistent_client(path: Path):
    import chromadb
    from chromadb.config import Settings

    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_names(client) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        name = item.name if hasattr(item, "name") else str(item)
        names.add(name)
    return names
