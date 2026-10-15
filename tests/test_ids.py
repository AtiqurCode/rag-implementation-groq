from rag.ids import make_chunk_id


def test_chunk_id_is_stable() -> None:
    first = make_chunk_id("data/pdfs/policy.pdf", 3, 1, "same text")
    second = make_chunk_id("data/pdfs/policy.pdf", 3, 1, "same text")
    changed = make_chunk_id("data/pdfs/policy.pdf", 3, 1, "other text")
    assert first == second
    assert first != changed
    assert " " not in first
