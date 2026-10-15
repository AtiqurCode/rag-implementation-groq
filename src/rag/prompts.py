from rag.models import RetrievedChunk

NOT_FOUND = "I cannot find that in the provided documents."

_RULES = """\
Answer the question using only the numbered excerpts.
If the excerpts do not contain the answer, reply exactly: {not_found}
Cite excerpt numbers like [1] or [2] immediately after each claim they support.
Do not use outside knowledge. Do not guess numbers, dates, or names that are not in an excerpt.\
""".format(not_found=NOT_FOUND)


def build_user_prompt(question: str, passages: list[RetrievedChunk]) -> str:
    blocks = []
    for number, passage in enumerate(passages, start=1):
        blocks.append(
            f"[{number}] source={passage.source} page={passage.page}\n{passage.text}"
        )
    excerpts = "\n\n".join(blocks)
    return f"{_RULES}\n\nExcerpts:\n{excerpts}\n\nQuestion: {question.strip()}"
