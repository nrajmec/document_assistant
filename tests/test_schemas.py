import pytest
from pydantic import ValidationError

from schemas import (
    AnswerResponse,
    SummarizationResponse,
    CalculationResponse,
    UpdateMemoryResponse,
    UserIntent,
    SessionState,
)


def test_user_intent_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        UserIntent(intent_type="qa", confidence=1.5, reasoning="too high")


def test_user_intent_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        UserIntent(intent_type="qa", confidence=-0.1, reasoning="too low")


def test_user_intent_rejects_invalid_intent_type():
    with pytest.raises(ValidationError):
        UserIntent(intent_type="not_a_real_intent", confidence=0.5, reasoning="bad label")


def test_user_intent_accepts_valid_labels():
    for label in ["qa", "summarization", "calculation", "unknown"]:
        intent = UserIntent(intent_type=label, confidence=0.8, reasoning="ok")
        assert intent.intent_type == label


def test_answer_response_rejects_confidence_out_of_bounds():
    with pytest.raises(ValidationError):
        AnswerResponse(question="q", answer="a", confidence=2.0)


def test_answer_response_defaults():
    resp = AnswerResponse(question="What is X?", answer="X is Y")
    assert resp.sources == []
    assert resp.confidence == 0.0


def test_summarization_response_defaults():
    resp = SummarizationResponse(original_length=100, summary="short summary", key_points=["a"])
    assert resp.document_ids == []


def test_calculation_response_requires_fields():
    resp = CalculationResponse(expression="2+2", result=4.0, explanation="added two numbers")
    assert resp.units is None


def test_update_memory_response_defaults():
    resp = UpdateMemoryResponse(summary="conversation so far")
    assert resp.document_ids == []


def test_session_state_defaults():
    session = SessionState(session_id="s1", user_id="u1")
    assert session.conversation_history == []
    assert session.document_context == []
    assert session.created_at is not None
    assert session.last_updated is not None
