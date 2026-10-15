import re

_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\f", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks that never cross the given size.

    Splits on paragraph, line, sentence, then word boundaries before falling
    back to a hard character window. Chunks stay page-local because callers
    pass one page at a time.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    return _recursive(cleaned, chunk_size, chunk_overlap, 0)


def _recursive(text: str, chunk_size: int, chunk_overlap: int, sep_index: int) -> list[str]:
    if sep_index >= len(_SEPARATORS) - 1:
        return _hard_split(text, chunk_size, chunk_overlap)

    units = _split_keep(text, _SEPARATORS[sep_index])
    if len(units) <= 1:
        return _recursive(text, chunk_size, chunk_overlap, sep_index + 1)

    expanded: list[str] = []
    for unit in units:
        if len(unit) > chunk_size:
            expanded.extend(_recursive(unit, chunk_size, chunk_overlap, sep_index + 1))
        else:
            expanded.append(unit)
    return _merge(expanded, chunk_size, chunk_overlap)


def _split_keep(text: str, separator: str) -> list[str]:
    parts = text.split(separator)
    if len(parts) == 1:
        return [text] if text else []

    units: list[str] = []
    last = len(parts) - 1
    for index, part in enumerate(parts):
        unit = part + separator if index < last else part
        if unit:
            units.append(unit)
    return units


def _merge(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append("".join(current).strip())
                current, current_len = [], 0
            chunks.extend(_hard_split(unit, chunk_size, chunk_overlap))
            continue

        if current and current_len + len(unit) > chunk_size:
            chunks.append("".join(current).strip())
            while current and (
                current_len > chunk_overlap or current_len + len(unit) > chunk_size
            ):
                current_len -= len(current.pop(0))
            if current_len < 0:
                current_len = 0

        current.append(unit)
        current_len += len(unit)

    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks
