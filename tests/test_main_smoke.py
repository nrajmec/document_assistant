import main as main_module
from src.assistant import DocumentAssistant


def test_document_assistant_importable():
    """Regression test: main.py previously failed with
    ImportError: cannot import name 'DocumentAssistant' from 'src.assistant'."""
    assert DocumentAssistant is not None


def test_main_module_exposes_entrypoint():
    assert hasattr(main_module, "main")
    assert callable(main_module.main)


def test_main_exits_gracefully_without_api_key(monkeypatch, capsys):
    # Prevent a real .env (found by walking up from cwd) from repopulating the key.
    monkeypatch.setattr(main_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _fail_if_constructed(*args, **kwargs):
        raise AssertionError("DocumentAssistant should not be constructed without an API key")

    monkeypatch.setattr(main_module, "DocumentAssistant", _fail_if_constructed)

    result = main_module.main()

    assert result is None
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY not found" in captured.out
