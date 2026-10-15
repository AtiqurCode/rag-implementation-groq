import logging

from groq import APIStatusError, Groq

from rag.exceptions import RagError
from rag.groq_errors import explain_groq_error

logger = logging.getLogger(__name__)

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class GroqEmbedder:
    """Nomic embeddings on Groq.

    nomic-embed-text-v1.5 is trained with task prefixes. Documents and queries
    must use different prefixes or retrieval quality collapses.
    """

    def __init__(self, client: Groq, model: str, batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, DOCUMENT_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], QUERY_PREFIX)[0]

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        if any(not item.strip() for item in texts):
            raise RagError("Refusing to embed an empty string.")

        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self._batch_size):
            batch = texts[start : start + self._batch_size]
            payload = [_prefix(prefix, item) for item in batch]
            vectors.extend(self._embed_batch(payload))
            logger.info("embedded %s/%s", min(start + len(batch), total), total)
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self._model,
                encoding_format="float",
            )
        except APIStatusError as exc:
            raise explain_groq_error(exc, model=self._model, operation="embeddings") from exc
        except Exception as exc:
            raise RagError(f"Groq embeddings failed: {exc}") from exc

        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise RagError(
                f"Groq returned {len(ordered)} embeddings for {len(texts)} inputs."
            )

        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.embedding
            if not isinstance(vector, list) or not vector:
                raise RagError("Groq returned a non-float embedding. encoding_format=float is required.")
            vectors.append([float(value) for value in vector])
        return vectors


class LocalEmbedder:
    """On-disk embeddings. This Groq account has chat models and no embedding model."""

    def __init__(self, model: str, batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model
        self._batch_size = batch_size
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run(texts, passages=True)

    def embed_query(self, text: str) -> list[float]:
        return self._run([text], passages=False)[0]

    def _run(self, texts: list[str], *, passages: bool) -> list[list[float]]:
        if not texts:
            return []
        if any(not item.strip() for item in texts):
            raise RagError("Refusing to embed an empty string.")

        model = self._load()
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = model.passage_embed(batch) if passages else model.query_embed(batch)
            vectors.extend(_rows(encoded))
            logger.info("embedded %s/%s", min(start + len(batch), total), total)
        return vectors

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RagError("fastembed is not installed. Run: uv pip install fastembed") from exc
            logger.info("loading local embedding model %s", self._model_name)
            self._model = TextEmbedding(self._model_name)
        return self._model


def _rows(encoded) -> list[list[float]]:
    rows: list[list[float]] = []
    for vector in encoded:
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if not values:
            raise RagError("Local embedder returned an empty vector.")
        rows.append([float(value) for value in values])
    return rows


def _prefix(prefix: str, text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(prefix):
        return stripped
    return prefix + stripped
