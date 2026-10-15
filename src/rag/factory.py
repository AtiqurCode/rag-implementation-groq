from groq import Groq

from rag.config import Settings
from rag.embeddings import GroqEmbedder, LocalEmbedder
from rag.exceptions import RagError
from rag.generator import GroqGenerator
from rag.pipeline import RagPipeline
from rag.store import ChromaStore


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    settings = settings or Settings()
    client = Groq(
        api_key=settings.groq_api_key.get_secret_value(),
        timeout=120.0,
        max_retries=4,
    )
    if settings.embed_provider == "local":
        embedder = LocalEmbedder(settings.local_embed_model, settings.embed_batch_size)
        embed_model = settings.local_embed_model
    elif settings.embed_provider == "groq":
        embedder = GroqEmbedder(client, settings.groq_embed_model, settings.embed_batch_size)
        embed_model = settings.groq_embed_model
    else:
        raise RagError(f"Unknown EMBED_PROVIDER '{settings.embed_provider}'")
    return RagPipeline(
        embedder=embedder,
        store=ChromaStore(
            settings.chroma_path,
            settings.collection_name,
            embed_model,
        ),
        generator=GroqGenerator(
            client,
            settings.groq_chat_model,
            temperature=settings.groq_temperature,
            max_completion_tokens=settings.groq_max_completion_tokens,
            reasoning_effort=settings.groq_reasoning_effort,
            include_reasoning=settings.include_reasoning,
        ),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
    )
