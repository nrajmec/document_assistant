import pytest
from langchain_core.messages import AIMessage

import agent as agent_module
from agent import create_workflow
from schemas import (
    UserIntent,
    UpdateMemoryResponse,
    AnswerResponse,
    SummarizationResponse,
    CalculationResponse,
)


class _CannedInvoker:
    def __init__(self, canned):
        self._canned = canned

    def invoke(self, prompt):
        return self._canned


class FakeLLM:
    """Stands in for a real chat model. with_structured_output() returns a canned
    response keyed on the requested schema, so the same fake can serve both
    classify_intent (UserIntent) and update_memory (UpdateMemoryResponse)."""

    def __init__(self, intent_type):
        self._intent_type = intent_type

    def with_structured_output(self, schema):
        if schema is UserIntent:
            canned = UserIntent(intent_type=self._intent_type, confidence=0.9, reasoning="test")
        elif schema is UpdateMemoryResponse:
            canned = UpdateMemoryResponse(summary="a short summary", document_ids=["DOC-1"])
        else:
            raise AssertionError(f"unexpected schema requested: {schema}")
        return _CannedInvoker(canned)


_SCHEMA_TOOL_MAP = {
    AnswerResponse: ["document_reader"],
    SummarizationResponse: ["document_search"],
    CalculationResponse: ["calculator"],
}


def _fake_invoke_react_agent(response_schema, messages, llm, tools):
    """Stands in for the real ReAct agent loop so tests don't need a live LLM
    or real tool-calling machinery. Returns a distinct tool per schema so tests
    can verify tools_used reflects only the node that actually ran."""
    tools_used = _SCHEMA_TOOL_MAP[response_schema]
    result = {"messages": [AIMessage(content=f"fake response for {response_schema.__name__}")]}
    return result, tools_used


@pytest.fixture(autouse=True)
def patch_invoke_react_agent(monkeypatch):
    monkeypatch.setattr(agent_module, "invoke_react_agent", _fake_invoke_react_agent)


def _initial_state(user_input):
    return {
        "messages": [],
        "user_input": user_input,
        "intent": None,
        "next_step": "classify_intent",
        "conversation_summary": "",
        "active_documents": [],
        "current_response": None,
        "tools_used": [],
        "session_id": None,
        "user_id": None,
        "actions_taken": [],
    }


@pytest.mark.parametrize(
    "intent_type,expected_worker,expected_tools",
    [
        ("qa", "qa_agent", ["document_reader"]),
        ("summarization", "summarization_agent", ["document_search"]),
        ("calculation", "calculation_agent", ["calculator"]),
    ],
)
def test_workflow_routes_and_scopes_state_to_current_turn(intent_type, expected_worker, expected_tools):
    workflow = create_workflow(None, [])
    config = {"configurable": {"thread_id": f"thread-{intent_type}", "llm": FakeLLM(intent_type), "tools": []}}

    final_state = workflow.invoke(_initial_state("some user input"), config=config)

    assert final_state["next_step"] == "end"
    assert final_state["actions_taken"] == ["classify_intent", expected_worker]
    assert final_state["tools_used"] == expected_tools
    assert final_state["conversation_summary"] == "a short summary"
    assert final_state["active_documents"] == ["DOC-1"]


def test_unknown_intent_falls_back_to_qa_worker():
    workflow = create_workflow(None, [])
    config = {"configurable": {"thread_id": "thread-unknown", "llm": FakeLLM("unknown"), "tools": []}}

    final_state = workflow.invoke(_initial_state("gibberish input"), config=config)

    assert final_state["actions_taken"] == ["classify_intent", "qa_agent"]
    assert final_state["tools_used"] == ["document_reader"]


def test_tools_used_does_not_leak_across_turns_on_same_thread():
    """Regression test: a QA turn on invoice INV-001 should not report the
    calculator tool from an earlier, unrelated calculation turn."""
    workflow = create_workflow(None, [])
    thread_id = "shared-thread"

    config1 = {"configurable": {"thread_id": thread_id, "llm": FakeLLM("calculation"), "tools": []}}
    turn1 = workflow.invoke(_initial_state("calculate something"), config=config1)
    assert turn1["tools_used"] == ["calculator"]

    config2 = {"configurable": {"thread_id": thread_id, "llm": FakeLLM("qa"), "tools": []}}
    turn2 = workflow.invoke(_initial_state("what is the answer"), config=config2)

    assert turn2["tools_used"] == ["document_reader"]
    assert "calculator" not in turn2["tools_used"]
