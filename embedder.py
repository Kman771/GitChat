"""embedder.py — embed chunks from chunker.py and store them in PostgreSQL."""

import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from chunker import chunk_repo
from config import DATABASE_URL, OLLAMA_URL

OLLAMA_MODEL = "nomic-embed-text"
BATCH_SIZE = 64


def get_embeddings(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Call Ollama's /api/embed endpoint in batches and return all embeddings."""
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({"model": OLLAMA_MODEL, "input": batch}).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama rejected the request (HTTP {e.code}): {e.read().decode()}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_URL}. "
                "Start it with: cd ~/.ollama/bin && OLLAMA_MODELS=~/.ollama/models ./ollama serve &"
            ) from e

        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")

        all_embeddings.extend(data["embeddings"])

    return all_embeddings


def embed_and_store(chunks: list[dict], conn) -> tuple[int, int]:
    """Embed all chunks and bulk-insert them into the chunks table.

    Accepts a flat list of chunk dicts (each with chunk_type, chunk_name, content, etc.).
    Returns (code_rows_inserted, docs_rows_inserted).
    """
    if not chunks:
        return 0, 0

    cur = conn.cursor()
    counts = {"code": 0, "docs": 0}

    # Embed all chunks in one batch for efficiency
    print(f"  Embedding {len(chunks)} chunks...", flush=True)
    texts = [c["content"] for c in chunks]
    embeddings = get_embeddings(texts)

    rows = [
        (
            c["repo_name"],
            c["file_path"],
            c["chunk_type"],
            c["content"],
            json.dumps({"name": c["chunk_name"]}),
            np.array(emb, dtype=np.float32),
        )
        for c, emb in zip(chunks, embeddings)
    ]

    cur.executemany(
        """
        INSERT INTO chunks (repo_name, file_path, chunk_type, content, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        rows,
    )

    for c in chunks:
        counts[c["chunk_type"]] = counts.get(c["chunk_type"], 0) + 1

    conn.commit()
    cur.close()
    return counts.get("code", 0), counts.get("docs", 0)


def main(repo_url: str) -> None:
    db_url = DATABASE_URL

    print(f"Chunking {repo_url} ...")
    chunks = chunk_repo(repo_url)
    code_chunks = sum(1 for c in chunks if c["chunk_type"] == "code")
    docs_chunks = sum(1 for c in chunks if c["chunk_type"] == "docs")
    print(f"  {code_chunks} code chunks, {docs_chunks} docs chunks")

    print(f"Connecting to database ...")
    conn = psycopg2.connect(db_url)
    register_vector(conn)

    print("Embedding and storing ...")
    code_n, docs_n = embed_and_store(chunks, conn)
    conn.close()

    print(f"Done — inserted {code_n} code rows, {docs_n} docs rows into chunks table.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python embedder.py <github-repo-url>")
        sys.exit(1)
    main(sys.argv[1])
