"""test_embedder.py — integration tests for embedder.py. Requires Ollama running."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import urllib.request
import urllib.error

import numpy as np
import pytest

from embedder import get_embeddings, embed_and_store, OLLAMA_URL, BATCH_SIZE


def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL.replace("/api/embed", ""), timeout=3)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama is not running — skipping embedder integration tests",
)


SAMPLE_TEXTS = [
    "def hello(): return 'hi'",
    "class Dog: pass",
    "## Installation\n\nRun pip install.",
]

SYNTHETIC_CHUNKS = {
    "code": [
        {"name": "hello", "file_path": "a.py", "repo_name": "_test_repo_", "content": "def hello(): return 'hi'"},
        {"name": "world", "file_path": "b.py", "repo_name": "_test_repo_", "content": "def world(): return 'world'"},
    ],
    "docs": [
        {"name": "Intro", "file_path": "README.md", "repo_name": "_test_repo_", "content": "## Intro\n\nWelcome."},
    ],
}


# ── Embedding shape and type ───────────────────────────────────────────────────

def test_embeddings_shape():
    embeddings = get_embeddings(SAMPLE_TEXTS)
    assert len(embeddings) == len(SAMPLE_TEXTS)
    for vec in embeddings:
        assert len(vec) == 768, f"Expected 768-dim vector, got {len(vec)}"


def test_embeddings_are_floats():
    embeddings = get_embeddings(SAMPLE_TEXTS)
    for vec in embeddings:
        for val in vec:
            assert isinstance(val, float), f"Expected float, got {type(val)}"


def test_batch_boundary():
    # Send BATCH_SIZE + 1 texts to exercise the batching loop boundary
    texts = ["test text"] * (BATCH_SIZE + 1)
    embeddings = get_embeddings(texts)
    assert len(embeddings) == BATCH_SIZE + 1


# ── embed_and_store ────────────────────────────────────────────────────────────

def test_embed_and_store_returns_counts(db_conn):
    conn = db_conn
    try:
        code_n, docs_n = embed_and_store(SYNTHETIC_CHUNKS, conn)
        assert code_n == 2
        assert docs_n == 1
    finally:
        conn.rollback()


def test_embed_and_store_inserts_rows(db_conn):
    conn = db_conn
    cur = conn.cursor()
    try:
        # Clean slate for our test repo name
        cur.execute("DELETE FROM chunks WHERE repo_name = '_test_repo_'")
        embed_and_store(SYNTHETIC_CHUNKS, conn)

        cur.execute("SELECT COUNT(*) FROM chunks WHERE repo_name = '_test_repo_'")
        count = cur.fetchone()[0]
        assert count == 3  # 2 code + 1 docs
    finally:
        conn.rollback()
        cur.close()
