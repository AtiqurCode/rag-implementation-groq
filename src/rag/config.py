import re
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_COLLECTION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: SecretStr
    groq_chat_model: str = "openai/gpt-oss-120b"
    embed_provider: str = "local"
    local_embed_model: str = "BAAI/bge-small-en-v1.5"
    groq_embed_model: str = "nomic-embed-text-v1_5"
    groq_reasoning_effort: str = "low"
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    groq_max_completion_tokens: int = Field(default=2048, ge=64, le=32768)
    include_reasoning: bool = False

    chroma_path: Path = Path("storage/chroma")
    collection_name: str = "pdf_rag"

    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0)
    embed_batch_size: int = Field(default=32, ge=1, le=128)

    @field_validator("embed_provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"local", "groq"}:
            raise ValueError("EMBED_PROVIDER must be local or groq")
        return normalized

    @field_validator("groq_reasoning_effort")
    @classmethod
    def _effort(cls, value: str) -> str:
        allowed = {"", "low", "medium", "high"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError("GROQ_REASONING_EFFORT must be low, medium, high, or empty")
        return normalized

    @field_validator("collection_name")
    @classmethod
    def _collection(cls, value: str) -> str:
        if not _COLLECTION_NAME.fullmatch(value):
            raise ValueError(
                "COLLECTION_NAME must be 3-63 characters and start and end with a letter or digit"
            )
        return value

    @model_validator(mode="after")
    def _overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self
