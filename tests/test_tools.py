import os
import shutil
import pytest

from tools import (
    ToolLogger,
    create_calculator_tool,
    create_document_reader_tool,
    create_document_search_tool,
)
from retrieval import SimulatedRetriever

LOGS_DIR = os.path.join(os.path.dirname(__file__), "_tmp_logs")


@pytest.fixture
def logger():
    shutil.rmtree(LOGS_DIR, ignore_errors=True)
    log = ToolLogger(logs_dir=LOGS_DIR, session_id="test-session")
    yield log
    shutil.rmtree(LOGS_DIR, ignore_errors=True)


@pytest.fixture
def retriever():
    return SimulatedRetriever()


def test_calculator_evaluates_simple_expression(logger):
    calculator = create_calculator_tool(logger)
    result = calculator.invoke({"expression": "2 + 2"})
    assert "4" in result


def test_calculator_evaluates_parentheses_and_operators(logger):
    calculator = create_calculator_tool(logger)
    result = calculator.invoke({"expression": "(100 - 20) * 3"})
    assert "240" in result


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "2+2; import os",
        "open('/etc/passwd')",
        "[x for x in range(10)]",
    ],
)
def test_calculator_rejects_unsafe_expressions(logger, expression):
    calculator = create_calculator_tool(logger)
    result = calculator.invoke({"expression": expression})
    assert "Invalid expression" in result


def test_calculator_handles_division_by_zero_without_raising(logger):
    calculator = create_calculator_tool(logger)
    result = calculator.invoke({"expression": "1 / 0"})
    assert "Error" in result


def test_calculator_logs_calls(logger):
    calculator = create_calculator_tool(logger)
    calculator.invoke({"expression": "3 * 3"})
    logs = logger.get_logs()
    assert len(logs) == 1
    assert logs[0]["tool_name"] == "calculator"
    assert logs[0]["input"] == {"expression": "3 * 3"}


def test_calculator_logs_invalid_expression_as_error(logger):
    calculator = create_calculator_tool(logger)
    calculator.invoke({"expression": "rm -rf /"})
    logs = logger.get_logs()
    assert len(logs) == 1
    assert "error" in logs[0]["output"].lower()


def test_document_reader_returns_known_document(logger, retriever):
    reader = create_document_reader_tool(retriever, logger)
    result = reader.invoke({"doc_id": "INV-001"})
    assert "INV-001" in result
    assert "Acme Corporation" in result


def test_document_reader_reports_missing_document(logger, retriever):
    reader = create_document_reader_tool(retriever, logger)
    result = reader.invoke({"doc_id": "DOES-NOT-EXIST"})
    assert "not found" in result.lower()


def test_document_search_finds_keyword_match(logger, retriever):
    search = create_document_search_tool(retriever, logger)
    result = search.invoke({"query": "invoice", "search_type": "keyword"})
    assert "Found" in result


def test_document_search_reports_no_matches(logger, retriever):
    search = create_document_search_tool(retriever, logger)
    result = search.invoke({"query": "zzz_no_such_keyword_zzz", "search_type": "keyword"})
    assert "No documents found" in result
