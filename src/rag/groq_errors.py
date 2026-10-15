from groq import APIStatusError

from rag.exceptions import RagError


def explain_groq_error(exc: APIStatusError, *, model: str, operation: str) -> RagError:
    if exc.status_code == 401:
        return RagError("Groq rejected GROQ_API_KEY. Create one at https://console.groq.com/keys")
    if exc.status_code in {400, 404}:
        return RagError(
            f"Groq rejected model '{model}' during {operation} "
            f"({exc.status_code}). Check the id at https://console.groq.com/docs/models"
        )
    return RagError(f"Groq {operation} failed ({exc.status_code}): {exc.message}")
