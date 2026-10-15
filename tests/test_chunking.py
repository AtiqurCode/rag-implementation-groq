import pytest

from rag.text import chunk_text, normalize_text


def test_normalize_collapses_whitespace() -> None:
    assert normalize_text("a \t b\r\n\r\n\r\nc") == "a b\n\nc"


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("Refunds are allowed within 30 days.", 100, 10) == [
        "Refunds are allowed within 30 days."
    ]


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("   \n\n  ", 100, 10) == []


def test_chunks_respect_size_and_overlap() -> None:
    text = " ".join(f"w{i:03d}" for i in range(80))
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
    for left, right in zip(chunks, chunks[1:]):
        tail = left.split()[-1]
        assert tail in right


def test_hard_split_steps_by_overlap() -> None:
    text = "x" * 25
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert chunks == ["x" * 10, "x" * 10, "x" * 9]


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, chunk_overlap=10)
