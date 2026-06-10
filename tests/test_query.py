"""test_query.py — unit and integration tests for query.py.

Unit tests: mock retrieve + mock anthropic client. No network, no DB, no API key required.
Integration tests: real Ollama + real Claude API. Skip if either is unavailable.
  Assumes bottle, tinyrenderer, and minds-platform are already indexed in the DB.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import urllib.request
from unittest.mock import MagicMock, patch, call

import pytest

from query import _format_chunks, ChatSession
from embedder import OLLAMA_URL


def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL.replace("/api/embed", ""), timeout=3)
        return True
    except Exception:
        return False


def _claude_api_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _fake_chunk(
    repo_name="bottle",
    file_path="bottle.py",
    name="route",
    content="def route(): pass",
    similarity=0.90,
):
    return {
        "id": 1,
        "repo_name": repo_name,
        "file_path": file_path,
        "chunk_type": "code",
        "content": content,
        "metadata": {"name": name},
        "similarity": similarity,
    }


def _mock_claude_response(reply_text="Here is the answer."):
    """Build a MagicMock shaped like an anthropic Messages response."""
    block = MagicMock()
    block.type = "text"
    block.text = reply_text

    usage = MagicMock()
    usage.input_tokens = 100
    usage.cache_creation_input_tokens = 50
    usage.cache_read_input_tokens = 0
    usage.output_tokens = 80

    response = MagicMock()
    response.content = [block]
    response.usage = usage
    return response


def _make_session(retrieve_return=None, reply_text="Here is the answer.", repo_name=None):
    """Return a (session, mock_client, mock_retrieve_patch) tuple ready for use in tests."""
    if retrieve_return is None:
        retrieve_return = [_fake_chunk()]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(reply_text)
    conn = MagicMock()
    session = ChatSession(mock_client, conn, repo_name=repo_name)
    return session, mock_client, conn


# ── _format_chunks unit tests ──────────────────────────────────────────────────

def test_format_chunks_empty_list():
    assert _format_chunks([]) == ""


def test_format_chunks_header_line():
    output = _format_chunks([_fake_chunk()])
    first_line = output.splitlines()[0]
    assert "Relevant source sections" in first_line


def test_format_chunks_contains_repo_and_path():
    output = _format_chunks([_fake_chunk(repo_name="bottle", file_path="bottle.py")])
    assert "bottle/bottle.py" in output


def test_format_chunks_contains_similarity_score():
    output = _format_chunks([_fake_chunk(similarity=0.9)])
    assert "similarity 0.900" in output


def test_format_chunks_wraps_content_in_code_fence():
    output = _format_chunks([_fake_chunk(content="def route(): pass")])
    assert "```" in output
    assert "def route(): pass" in output


def test_format_chunks_includes_function_name_from_metadata():
    output = _format_chunks([_fake_chunk(name="route")])
    assert "· route" in output


def test_format_chunks_omits_dot_when_name_empty():
    chunk = _fake_chunk()
    chunk["metadata"] = {"name": ""}
    output = _format_chunks([chunk])
    assert "·" not in output


def test_format_chunks_numbers_multiple_chunks():
    chunks = [_fake_chunk(name=f"fn{i}") for i in range(3)]
    output = _format_chunks(chunks)
    assert "[1]" in output
    assert "[2]" in output
    assert "[3]" in output


# ── ChatSession unit tests ─────────────────────────────────────────────────────

def test_chat_session_initializes_with_empty_history():
    session, _, _ = _make_session()
    assert session.messages == []


@patch("query.retrieve")
def test_send_appends_two_messages_to_history(mock_retrieve):
    mock_retrieve.return_value = [_fake_chunk()]
    session, mock_client, _ = _make_session()
    mock_client.messages.create.return_value = _mock_claude_response()
    session.send("how does routing work?")
    assert len(session.messages) == 2


@patch("query.retrieve")
def test_send_first_message_is_user_role(mock_retrieve):
    mock_retrieve.return_value = [_fake_chunk()]
    session, _, _ = _make_session()
    session.send("q")
    assert session.messages[0]["role"] == "user"


@patch("query.retrieve")
def test_send_second_message_is_assistant_role(mock_retrieve):
    mock_retrieve.return_value = [_fake_chunk()]
    session, _, _ = _make_session()
    session.send("q")
    assert session.messages[1]["role"] == "assistant"


@patch("query.retrieve")
def test_send_with_chunks_user_content_is_list(mock_retrieve):
    mock_retrieve.return_value = [_fake_chunk()]
    session, _, _ = _make_session()
    session.send("how does routing work?")
    assert isinstance(session.messages[0]["content"], list)


@patch("query.retrieve")
def test_send_without_chunks_user_content_is_string(mock_retrieve):
    mock_retrieve.return_value = []
    session, _, _ = _make_session()
    session.send("unrelated query")
    assert isinstance(session.messages[0]["content"], str)


@patch("query.retrieve")
def test_send_chunks_block_has_cache_control(mock_retrieve):
    mock_retrieve.return_value = [_fake_chunk()]
    session, _, _ = _make_session()
    session.send("q")
    content = session.messages[0]["content"]
    # First block is the context block
    assert content[0]["cache_control"] == {"type": "ephemeral"}


@patch("query.retrieve")
def test_send_returns_reply_text(mock_retrieve):
    mock_retrieve.return_value = []
    session, mock_client, _ = _make_session()
    mock_client.messages.create.return_value = _mock_claude_response("Great question!")
    reply = session.send("q")
    assert reply == "Great question!"


@patch("query.retrieve")
def test_multi_turn_accumulates_history(mock_retrieve):
    mock_retrieve.return_value = []
    session, mock_client, _ = _make_session()
    mock_client.messages.create.return_value = _mock_claude_response()
    session.send("first question")
    session.send("follow-up question")
    assert len(session.messages) == 4


@patch("query.retrieve")
def test_system_prompt_has_cache_control(mock_retrieve):
    mock_retrieve.return_value = []
    session, mock_client, _ = _make_session()
    mock_client.messages.create.return_value = _mock_claude_response()
    session.send("q")
    call_kwargs = mock_client.messages.create.call_args[1]
    system = call_kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


@patch("query.retrieve")
def test_send_passes_full_history_to_api_on_second_turn(mock_retrieve):
    mock_retrieve.return_value = []
    session, mock_client, _ = _make_session()

    # session.messages is a mutable list passed by reference, so call_args would reflect
    # its final state (4 entries) by the time we inspect it after both sends.
    # Capture a snapshot copy on each call instead.
    snapshots: list[list] = []
    def capture(*args, **kwargs):
        snapshots.append(list(kwargs["messages"]))
        return _mock_claude_response()

    mock_client.messages.create.side_effect = capture
    session.send("first")
    session.send("second")
    # Second API call should have 3 messages: user1, assistant1, user2
    assert len(snapshots[1]) == 3


@patch("query.retrieve")
def test_send_repo_name_scoped_session(mock_retrieve):
    mock_retrieve.return_value = []
    session, mock_client, conn = _make_session(repo_name="bottle")
    mock_client.messages.create.return_value = _mock_claude_response()
    session.send("how does routing work?")
    mock_retrieve.assert_called_once_with(
        "how does routing work?", conn, repo_name="bottle"
    )


# ── Integration tests ──────────────────────────────────────────────────────────

_skip_integration = pytest.mark.skipif(
    not (_ollama_available() and _claude_api_available()),
    reason="Ollama not running or ANTHROPIC_API_KEY not set — skipping query integration tests",
)


@_skip_integration
def test_e2e_bottle_question_returns_text(db_conn):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    session = ChatSession(client, db_conn, repo_name="bottle")
    reply = session.send("how does URL routing work?")
    assert isinstance(reply, str) and len(reply) > 0, "Expected a non-empty text reply"


@_skip_integration
def test_e2e_tinyrenderer_question_returns_text(db_conn):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    session = ChatSession(client, db_conn, repo_name="tinyrenderer")
    reply = session.send("how is a line drawn on screen?")
    assert isinstance(reply, str) and len(reply) > 0, "Expected a non-empty text reply"


@_skip_integration
def test_e2e_minds_platform_question_returns_text(db_conn):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    session = ChatSession(client, db_conn, repo_name="minds-platform")
    reply = session.send("how is an API call made?")
    assert isinstance(reply, str) and len(reply) > 0, "Expected a non-empty text reply"
