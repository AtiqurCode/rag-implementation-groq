import pytest
from pydantic import ValidationError

from rag.config import Settings


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, groq_api_key="test-key", chunk_size=200, chunk_overlap=200)


def test_settings_accept_a_key_without_reading_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = Settings(_env_file=None, groq_api_key="test-key")
    assert settings.groq_api_key.get_secret_value() == "test-key"
    assert settings.groq_chat_model == "openai/gpt-oss-120b"
    assert settings.chunk_overlap < settings.chunk_size
