import hashlib
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def make_chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Stable id. Re-ingesting the same page text does not mint a new identity."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    label = _UNSAFE.sub("_", source).strip("_")[:80] or "document"
    return f"{label}-p{page}-c{index}-{digest}"
