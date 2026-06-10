"""retrieval.py — embed a query and retrieve the most similar chunks from PostgreSQL."""

import json
import os
import sys

import numpy as np
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from embedder import get_embeddings

SIMILARITY_THRESHOLD = 0.7  # PLACEHOLDER — tune after testing against real queries
TOP_K = 10

DEFAULT_DB_URL = "postgresql://kaashishvenkat@localhost:5432/repo_rag"


def embed_query(query: str) -> list[float]:
    """Embed a single query string via Ollama and return the 768-dim vector."""
    return get_embeddings([query])[0]


def retrieve(
    query: str,
    conn,
    repo_name: str | None = None,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Find chunks whose cosine similarity to *query* is >= *threshold*.

    cosine similarity  = 1 − cosine distance
    pgvector operator  <=>  returns cosine distance, so:
        similarity = 1 − (embedding <=> query_vector)

    Args:
        query:     Natural-language question or code snippet.
        conn:      Open psycopg2 connection with pgvector registered.
        repo_name: If given, restrict search to this repo only.
        top_k:     Hard cap on results even when many chunks beat the threshold.
        threshold: Minimum cosine similarity to include a chunk (0–1 scale).

    Returns:
        List of dicts ordered by similarity descending, each containing:
        id, repo_name, file_path, chunk_type, content, metadata, similarity.
    """
    query_vec = np.array(embed_query(query), dtype=np.float32)

    # Build the query dynamically so the optional repo_name filter is clean.
    # Inner query computes similarity once; outer query filters and sorts.
    if repo_name:
        inner_sql = """
            SELECT id, repo_name, file_path, chunk_type, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE repo_name = %s
        """
        inner_params = (query_vec, repo_name)
    else:
        inner_sql = """
            SELECT id, repo_name, file_path, chunk_type, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
        """
        inner_params = (query_vec,)

    sql = f"""
        SELECT * FROM ({inner_sql}) sub
        WHERE similarity >= %s
        ORDER BY similarity DESC
        LIMIT %s
    """
    params = (*inner_params, threshold, top_k)

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()

    columns = ["id", "repo_name", "file_path", "chunk_type", "content", "metadata", "similarity"]
    return [dict(zip(columns, row)) for row in rows]


def main(query: str, repo_name: str | None = None) -> None:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

    conn = psycopg2.connect(db_url)
    register_vector(conn)

    print(f"\nQuery : {query!r}")
    if repo_name:
        print(f"Repo  : {repo_name}")
    print(f"Threshold : {SIMILARITY_THRESHOLD}  |  top_k : {TOP_K}\n")

    results = retrieve(query, conn, repo_name=repo_name)
    conn.close()

    if not results:
        print("No chunks found above the similarity threshold.")
        return

    print(f"Found {len(results)} chunk(s):\n")
    for r in results:
        meta_name = r["metadata"].get("name", "") if isinstance(r["metadata"], dict) else ""
        print(f"  [{r['similarity']:.4f}]  {r['repo_name']} / {r['file_path']}  ({meta_name})")
        print(f"           type={r['chunk_type']}")
        print(f"           {r['content'][:120].strip()!r}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <query> [repo_name]")
        sys.exit(1)
    q = sys.argv[1]
    rn = sys.argv[2] if len(sys.argv) > 2 else None
    main(q, rn)
