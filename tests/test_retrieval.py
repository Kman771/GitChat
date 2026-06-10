"""test_retrieval.py — unit and integration tests for retrieval.py.

Unit tests: mock embed_query + mock psycopg2 cursor. No Ollama, no DB required.
Integration tests: real Ollama embeddings + real PostgreSQL. Skip if Ollama unreachable.
  Assumes bottle, tinyrenderer, and minds-platform are already indexed in the DB.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import urllib.request
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retrieval import embed_query, retrieve, SIMILARITY_THRESHOLD
from embedder import OLLAMA_URL


def _repo_indexed(db_conn, repo_name: str) -> bool:
    """Return True if the repo has any rows in the chunks table."""
    cur = db_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks WHERE repo_name = %s", (repo_name,))
    count = cur.fetchone()[0]
    cur.close()
    return count > 0


def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL.replace("/api/embed", ""), timeout=3)
        return True
    except Exception:
        return False


FAKE_VEC = [0.0] * 768

FAKE_ROW = (
    1,           # id
    "bottle",    # repo_name
    "bottle.py", # file_path
    "code",      # chunk_type
    "def route(): pass",  # content
    {"name": "route"},    # metadata
    0.85,        # similarity
)

RESULT_KEYS = {"id", "repo_name", "file_path", "chunk_type", "content", "metadata", "similarity"}


def _make_mock_conn(rows=None):
    """Return a mock psycopg2 connection whose cursor returns *rows* from fetchall."""
    if rows is None:
        rows = [FAKE_ROW]
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


# ── Unit tests ─────────────────────────────────────────────────────────────────

@patch("retrieval.get_embeddings")
def test_embed_query_delegates_to_get_embeddings(mock_get_embeddings):
    mock_get_embeddings.return_value = [FAKE_VEC]
    result = embed_query("hello world")
    mock_get_embeddings.assert_called_once_with(["hello world"])
    assert result == FAKE_VEC


@patch("retrieval.embed_query", return_value=FAKE_VEC)
def test_retrieve_result_has_expected_keys(mock_embed):
    conn, _ = _make_mock_conn([FAKE_ROW])
    results = retrieve("routing", conn)
    assert len(results) == 1
    assert set(results[0].keys()) == RESULT_KEYS


@patch("retrieval.embed_query", return_value=FAKE_VEC)
def test_retrieve_empty_list_when_cursor_returns_nothing(mock_embed):
    conn, _ = _make_mock_conn([])
    results = retrieve("anything", conn)
    assert results == []


@patch("retrieval.embed_query", return_value=FAKE_VEC)
def test_retrieve_results_ordered_by_similarity_desc(mock_embed):
    rows = [
        (1, "r", "a.py", "code", "x", {}, 0.95),
        (2, "r", "b.py", "code", "y", {}, 0.80),
        (3, "r", "c.py", "code", "z", {}, 0.72),
    ]
    conn, _ = _make_mock_conn(rows)
    results = retrieve("q", conn)
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)


@patch("retrieval.embed_query", return_value=FAKE_VEC)
def test_retrieve_respects_top_k(mock_embed):
    conn, cursor = _make_mock_conn([FAKE_ROW])
    retrieve("q", conn, top_k=3)
    call_args = cursor.execute.call_args
    params = call_args[0][1]
    # params contains a numpy array (the query vector); filter it out before using `in`
    scalar_params = [p for p in params if not isinstance(p, np.ndarray)]
    assert 3 in scalar_params, f"top_k=3 should appear in SQL params, got: {scalar_params}"


@patch("retrieval.embed_query", return_value=FAKE_VEC)
def test_retrieve_threshold_in_sql_params(mock_embed):
    conn, cursor = _make_mock_conn([])
    retrieve("q", conn, threshold=0.99)
    call_args = cursor.execute.call_args
    params = call_args[0][1]
    scalar_params = [p for p in params if not isinstance(p, np.ndarray)]
    assert 0.99 in scalar_params, f"threshold=0.99 should appear in SQL params, got: {scalar_params}"


@patch("retrieval.embed_query", return_value=FAKE_VEC)
def test_retrieve_with_repo_name_includes_filter_param(mock_embed):
    conn, cursor = _make_mock_conn([])
    retrieve("routing", conn, repo_name="bottle")
    call_args = cursor.execute.call_args
    params = call_args[0][1]
    scalar_params = [p for p in params if not isinstance(p, np.ndarray)]
    assert "bottle" in scalar_params, f"'bottle' should appear in SQL params when repo_name set, got: {scalar_params}"


@patch("retrieval.embed_query", return_value=FAKE_VEC)
def test_retrieve_without_repo_name_omits_string_param(mock_embed):
    conn, cursor = _make_mock_conn([])
    retrieve("routing", conn)
    call_args = cursor.execute.call_args
    params = call_args[0][1]
    string_params = [p for p in params if isinstance(p, str)]
    assert string_params == [], f"No string (repo_name) params expected, got: {string_params}"


# ── Integration tests ──────────────────────────────────────────────────────────

pytestmark_integration = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama is not running — skipping retrieval integration tests",
)


@pytestmark_integration
def test_retrieve_bottle_routing_query(db_conn):
    results = retrieve("how does URL routing work", db_conn, repo_name="bottle")
    assert len(results) >= 1, "Expected at least one chunk about routing from bottle"


@pytestmark_integration
def test_retrieve_tinyrenderer_rendering_query(db_conn):
    if not _repo_indexed(db_conn, "tinyrenderer"):
        pytest.skip("tinyrenderer not indexed in DB — run embedder.py first")
    # tinyrenderer has only ~10 chunks (mostly C++ headers + README); use a lower
    # threshold so the test validates retrieval works, not the similarity cutoff tuning.
    results = retrieve("how is a triangle drawn on screen", db_conn, repo_name="tinyrenderer", threshold=0.5)
    assert len(results) >= 1, "Expected at least one chunk from tinyrenderer (threshold=0.5)"


@pytestmark_integration
def test_retrieve_minds_platform_api_query(db_conn):
    if not _repo_indexed(db_conn, "minds-platform"):
        pytest.skip("minds-platform not indexed in DB — run embedder.py first")
    # minds-platform has only ~39 chunks; lower threshold for same reason as tinyrenderer.
    results = retrieve("how are API calls made", db_conn, repo_name="minds-platform", threshold=0.5)
    assert len(results) >= 1, "Expected at least one chunk from minds-platform (threshold=0.5)"


@pytestmark_integration
def test_retrieve_similarity_scores_in_valid_range(db_conn):
    results = retrieve("function definition", db_conn, threshold=SIMILARITY_THRESHOLD)
    for r in results:
        assert SIMILARITY_THRESHOLD <= r["similarity"] <= 1.0, (
            f"Similarity {r['similarity']:.4f} is outside valid range"
        )


@pytestmark_integration
def test_retrieve_repo_name_isolates_results(db_conn):
    results = retrieve("how does the main entry point work", db_conn, repo_name="bottle")
    for r in results:
        assert r["repo_name"] == "bottle", (
            f"Expected only bottle chunks, got repo_name='{r['repo_name']}'"
        )


@pytestmark_integration
def test_retrieve_all_results_have_required_keys(db_conn):
    results = retrieve("class definition", db_conn)
    for r in results:
        missing = RESULT_KEYS - r.keys()
        assert not missing, f"Result missing keys: {missing}"
