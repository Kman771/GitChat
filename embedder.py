"""embedder.py — embed chunks from chunker.py and store them in PostgreSQL."""

import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from chunker import chunk_repo

OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "nomic-embed-text"
BATCH_SIZE = 64

DEFAULT_DB_URL = "postgresql://kaashishvenkat@localhost:5432/repo_rag"


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


def embed_and_store(chunks: dict[str, list[dict]], conn) -> tuple[int, int]:
    """Embed all chunks and bulk-insert them into the chunks table.

    Returns (code_rows_inserted, docs_rows_inserted).
    """
    cur = conn.cursor()
    counts = {"code": 0, "docs": 0}

    for chunk_type in ("code", "docs"):
        chunk_list = chunks.get(chunk_type, [])
        if not chunk_list:
            continue

        print(f"  Embedding {len(chunk_list)} {chunk_type} chunks...", flush=True)
        texts = [c["content"] for c in chunk_list]
        embeddings = get_embeddings(texts)

        rows = [
            (
                c["repo_name"],
                c["file_path"],
                chunk_type,
                c["content"],
                json.dumps({"name": c["name"]}),
                np.array(emb, dtype=np.float32),
            )
            for c, emb in zip(chunk_list, embeddings)
        ]

        cur.executemany(
            """
            INSERT INTO chunks (repo_name, file_path, chunk_type, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        counts[chunk_type] = len(rows)

    conn.commit()
    cur.close()
    return counts["code"], counts["docs"]


def main(repo_url: str) -> None:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

    print(f"Chunking {repo_url} ...")
    chunks = chunk_repo(repo_url)
    print(f"  {len(chunks['code'])} code chunks, {len(chunks['docs'])} docs chunks")

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
