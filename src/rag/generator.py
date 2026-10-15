import logging

from groq import APIStatusError, Groq

from rag.exceptions import RagError
from rag.groq_errors import explain_groq_error
from rag.models import RetrievedChunk
from rag.prompts import build_user_prompt

logger = logging.getLogger(__name__)


class GroqGenerator:
    """Grounded answer from retrieved excerpts.

    Instructions live in the user message. Groq's reasoning models follow that
    layout more reliably than a separate system prompt. Reasoning stays off the
    printed answer via include_reasoning=false.
    """

    def __init__(
        self,
        client: Groq,
        model: str,
        *,
        temperature: float,
        max_completion_tokens: int,
        reasoning_effort: str = "low",
        include_reasoning: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_completion_tokens = max_completion_tokens
        self._reasoning_effort = reasoning_effort
        self._include_reasoning = include_reasoning

    def answer(self, question: str, passages: list[RetrievedChunk]) -> str:
        if not passages:
            raise RagError("Refusing to generate without retrieved passages.")

        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": build_user_prompt(question, passages)}
            ],
            "temperature": self._temperature,
            "max_completion_tokens": self._max_completion_tokens,
            "include_reasoning": self._include_reasoning,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        try:
            response = self._client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            raise explain_groq_error(exc, model=self._model, operation="chat") from exc
        except Exception as exc:
            raise RagError(f"Groq chat failed: {exc}") from exc

        message = response.choices[0].message
        content = (message.content or "").strip()
        if not content:
            logger.warning("Groq returned an empty answer for model %s", self._model)
            raise RagError("Groq returned an empty answer. Raise GROQ_MAX_COMPLETION_TOKENS.")
        return content
