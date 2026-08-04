import pytest

from agent import classify_intent
from schemas import UserIntent


class FakeStructuredLLM:
    """Stands in for llm.with_structured_output(...).invoke(...) with a canned response."""

    def __init__(self, canned_response):
        self._canned = canned_response

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return self._canned


def _run_classify(intent_type):
    canned = UserIntent(intent_type=intent_type, confidence=0.9, reasoning="test reasoning")
    config = {"configurable": {"llm": FakeStructuredLLM(canned)}}
    state = {"user_input": "irrelevant for this test", "messages": []}
    return classify_intent(state, config)


@pytest.mark.parametrize(
    "intent_type,expected_next_step",
    [
        ("qa", "qa_agent"),
        ("summarization", "summarization_agent"),
        ("calculation", "calculation_agent"),
        ("unknown", "qa_agent"),
    ],
)
def test_classify_intent_routes_to_expected_worker(intent_type, expected_next_step):
    result = _run_classify(intent_type)
    assert result["next_step"] == expected_next_step
    assert result["intent"].intent_type == intent_type
    assert result["actions_taken"] == ["classify_intent"]
