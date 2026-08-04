from langchain_core.messages import SystemMessage

from prompts import (
    get_intent_classification_prompt,
    get_chat_prompt_template,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    CALCULATION_SYSTEM_PROMPT,
)


def test_classification_prompt_has_expected_input_variables():
    prompt = get_intent_classification_prompt()
    assert set(prompt.input_variables) == {"user_input", "conversation_history"}


def test_classification_prompt_formats_inputs():
    prompt = get_intent_classification_prompt()
    rendered = prompt.format(user_input="calculate the sum", conversation_history="none")
    assert "calculate the sum" in rendered
    assert "none" in rendered


def test_classification_prompt_lists_all_categories():
    rendered = get_intent_classification_prompt().format(
        user_input="x", conversation_history="y"
    )
    for label in ["qa", "summarization", "calculation", "unknown"]:
        assert label in rendered


def test_classification_prompt_asks_for_confidence_and_reasoning():
    rendered = get_intent_classification_prompt().format(
        user_input="x", conversation_history="y"
    )
    assert "confidence" in rendered.lower()
    assert "reasoning" in rendered.lower()


def test_classification_prompt_includes_example_per_category():
    rendered = get_intent_classification_prompt().format(
        user_input="x", conversation_history="y"
    )
    assert "Intent: qa" in rendered
    assert "Intent: summarization" in rendered
    assert "Intent: calculation" in rendered
    assert "Intent: unknown" in rendered


def test_classification_prompt_distinguishes_stated_total_from_calculation():
    rendered = get_intent_classification_prompt().format(
        user_input="x", conversation_history="y"
    )
    assert "already stated" in rendered.lower()


def test_chat_prompt_template_has_three_messages_for_each_intent():
    for intent_type in ["qa", "summarization", "calculation", "unknown"]:
        template = get_chat_prompt_template(intent_type)
        assert len(template.messages) == 3


def test_chat_prompt_template_uses_correct_system_prompt():
    qa_template = get_chat_prompt_template("qa").invoke(
        {"input": "hi", "chat_history": []}
    )
    summarization_template = get_chat_prompt_template("summarization").invoke(
        {"input": "hi", "chat_history": []}
    )
    calculation_template = get_chat_prompt_template("calculation").invoke(
        {"input": "hi", "chat_history": []}
    )

    assert qa_template.messages[0].content == QA_SYSTEM_PROMPT
    assert summarization_template.messages[0].content == SUMMARIZATION_SYSTEM_PROMPT
    assert calculation_template.messages[0].content == CALCULATION_SYSTEM_PROMPT


def test_chat_prompt_template_falls_back_to_qa_for_unknown_intent():
    template = get_chat_prompt_template("unknown").invoke(
        {"input": "hi", "chat_history": []}
    )
    assert isinstance(template.messages[0], SystemMessage)
    assert template.messages[0].content == QA_SYSTEM_PROMPT
