"""retrieval.py — embed a query and retrieve the most similar chunks from PostgreSQL."""

import os
import sys

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from embedder import get_embeddings
from config import DATABASE_URL

TOP_K = 10
MIN_SIMILARITY = 0.3
RESULT_CAP = 5


def embed_query(query: str) -> list[float]:
    """Embed a single query string via Ollama and return the 768-dim vector."""
    return get_embeddings([query])[0]


def retrieve(
    query: str,
    conn,
    repo_name: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Return the top *top_k* chunks most similar to *query*, ordered by cosine similarity.

    Args:
        query:     Natural-language question or code snippet.
        conn:      Open psycopg2 connection with pgvector registered.
        repo_name: If given, restrict search to this repo only.
        top_k:     Number of chunks to return (default 10).

    Returns:
        List of dicts ordered by similarity descending, each containing:
        id, repo_name, file_path, chunk_type, content, metadata, similarity.
    """
    query_vec = np.array(embed_query(query), dtype=np.float32)

    if repo_name:
        sql = """
            SELECT id, repo_name, file_path, chunk_type, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE repo_name = %s
            ORDER BY similarity DESC
            LIMIT %s
        """
        params = (query_vec, repo_name, top_k)
    else:
        sql = """
            SELECT id, repo_name, file_path, chunk_type, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY similarity DESC
            LIMIT %s
        """
        params = (query_vec, top_k)

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()

    columns = ["id", "repo_name", "file_path", "chunk_type", "content", "metadata", "similarity"]
    return [dict(zip(columns, row)) for row in rows]


def retrieve_for_queries(
    queries: list[str],
    conn,
    repo_name: str | None = None,
    top_k_per_query: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    result_cap: int = RESULT_CAP,
) -> list[dict]:
    """Embed each query term, retrieve chunks, deduplicate by id, filter by similarity threshold."""
    seen_ids: set[int] = set()
    results: list[dict] = []
    for q in queries:
        chunks = retrieve(q, conn, repo_name=repo_name, top_k=top_k_per_query)
        for chunk in chunks:
            if chunk["id"] not in seen_ids and chunk["similarity"] >= min_similarity:
                seen_ids.add(chunk["id"])
                results.append(chunk)
    results.sort(key=lambda c: c["similarity"], reverse=True)
    return results[:result_cap]


def main(query: str, repo_name: str | None = None) -> None:
    db_url = DATABASE_URL

    conn = psycopg2.connect(db_url)
    register_vector(conn)

    print(f"\nQuery : {query!r}")
    if repo_name:
        print(f"Repo  : {repo_name}")
    print(f"top_k : {TOP_K}\n")

    results = retrieve(query, conn, repo_name=repo_name)
    conn.close()

    if not results:
        print("No chunks found.")
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
