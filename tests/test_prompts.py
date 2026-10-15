from rag.models import RetrievedChunk
from rag.prompts import NOT_FOUND, build_user_prompt


def test_prompt_numbers_passages_and_states_the_refusal() -> None:
    prompt = build_user_prompt(
        "What is the refund window?",
        [
            RetrievedChunk("policy.pdf", 2, 0, "Refunds are allowed within 30 days.", 0.8),
            RetrievedChunk("policy.pdf", 4, 1, "Shipping takes five business days.", 0.4),
        ],
    )
    assert NOT_FOUND in prompt
    assert "[1] source=policy.pdf page=2" in prompt
    assert "[2] source=policy.pdf page=4" in prompt
    assert prompt.endswith("Question: What is the refund window?")
