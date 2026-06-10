"""test_repos.py — e2e tests against real GitHub repos.

Clones each repo, chunks it, embeds it, and stores results in the DB.
Existing rows for each repo_name are deleted before each run so the
table always reflects the latest state.

Run: pytest tests/test_repos.py -v -s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import urllib.request
import urllib.error

import pytest

from chunker import chunk_repo, _token_count, MAX_TOKENS
from embedder import embed_and_store, OLLAMA_URL
from tests.conftest import REPO_URLS


def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL.replace("/api/embed", ""), timeout=3)
        return True
    except Exception:
        return False


REQUIRED_KEYS = {"name", "file_path", "repo_name", "content"}


# ── Chunking tests (no Ollama, no DB needed) ───────────────────────────────────

@pytest.mark.parametrize("url", REPO_URLS)
def test_chunk_repo_returns_expected_keys(url):
    chunks = chunk_repo(url)
    assert "code" in chunks
    assert "docs" in chunks


@pytest.mark.parametrize("url", REPO_URLS)
def test_chunk_repo_nonempty(url):
    chunks = chunk_repo(url)
    total = len(chunks["code"]) + len(chunks["docs"])
    assert total > 0, f"No chunks produced for {url}"


@pytest.mark.parametrize("url", REPO_URLS)
def test_all_chunks_have_required_fields(url):
    chunks = chunk_repo(url)
    all_chunks = chunks["code"] + chunks["docs"]
    for chunk in all_chunks:
        missing = REQUIRED_KEYS - chunk.keys()
        assert not missing, f"Chunk missing keys {missing}: {chunk}"
        assert chunk["content"].strip(), "chunk content must not be blank"


@pytest.mark.parametrize("url", REPO_URLS)
def test_no_chunk_exceeds_token_limit(url):
    chunks = chunk_repo(url)
    all_chunks = chunks["code"] + chunks["docs"]
    violations = [
        (c["file_path"], c["name"], _token_count(c["content"]))
        for c in all_chunks
        if _token_count(c["content"]) > MAX_TOKENS
    ]
    assert not violations, (
        f"Chunks exceeding {MAX_TOKENS} tokens in {url}:\n"
        + "\n".join(f"  {fp}:{name} ({n} tokens)" for fp, name, n in violations)
    )


@pytest.mark.parametrize("url", REPO_URLS)
def test_repo_name_extracted_correctly(url):
    expected_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    chunks = chunk_repo(url)
    all_chunks = chunks["code"] + chunks["docs"]
    for chunk in all_chunks:
        assert chunk["repo_name"] == expected_name, (
            f"Expected repo_name '{expected_name}', got '{chunk['repo_name']}'"
        )


# ── E2E embed + store tests (needs Ollama + DB) ────────────────────────────────

@pytest.mark.parametrize("url", REPO_URLS)
def test_embed_and_store_e2e(url, db_conn):
    if not _ollama_available():
        pytest.skip("Ollama is not running")

    repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    conn = db_conn
    cur = conn.cursor()

    print(f"\nChunking {url} ...")
    chunks = chunk_repo(url)
    total_chunks = len(chunks["code"]) + len(chunks["docs"])
    print(f"  {len(chunks['code'])} code chunks, {len(chunks['docs'])} docs chunks")

    # Replace existing data for this repo
    cur.execute("DELETE FROM chunks WHERE repo_name = %s", (repo_name,))
    print(f"  Cleared existing rows for '{repo_name}'")

    code_n, docs_n = embed_and_store(chunks, conn)
    conn.commit()

    # Verify the DB matches what we inserted
    cur.execute("SELECT COUNT(*) FROM chunks WHERE repo_name = %s", (repo_name,))
    db_count = cur.fetchone()[0]
    cur.close()

    assert code_n + docs_n == total_chunks, (
        f"embed_and_store returned ({code_n}, {docs_n}) but expected {total_chunks} total"
    )
    assert db_count == total_chunks, (
        f"DB has {db_count} rows but expected {total_chunks} for '{repo_name}'"
    )
    print(f"  Stored {db_count} rows for '{repo_name}' ✓")
