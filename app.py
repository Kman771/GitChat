"""app.py — Flask web frontend for GitChat RAG system."""

import json
import os
import sys

import anthropic
import psycopg2
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from pgvector.psycopg2 import register_vector

from chunker import chunk_repo
from embedder import embed_and_store
from query import ChatSession, _format_chunks
from retrieval import retrieve

load_dotenv()

app = Flask(__name__)

DEFAULT_DB_URL = "postgresql://kaashishvenkat@localhost:5432/repo_rag"

# Module-level state — single-user dev tool, no concurrency needed.
_state: dict = {
    "conn": None,
    "session": None,
    "repo_name": None,
    "client": None,
}


def _get_conn():
    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    return conn


def _sse(status: str, message: str) -> str:
    return f"data: {json.dumps({'status': status, 'message': message})}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/index", methods=["POST"])
def index_repo():
    data = request.get_json()
    repo_url = (data or {}).get("repo_url", "").strip()
    if not repo_url:
        return jsonify({"error": "No repo URL provided"}), 400

    def generate():
        try:
            conn = _get_conn()

            yield _sse("clearing", "Clearing existing chunks from database...")
            cur = conn.cursor()
            cur.execute("DELETE FROM chunks")
            conn.commit()
            cur.close()

            yield _sse("cloning", f"Cloning {repo_url} (shallow clone)...")
            chunks = chunk_repo(repo_url)
            code_n = len(chunks["code"])
            docs_n = len(chunks["docs"])
            total = code_n + docs_n
            yield _sse("chunking", f"Chunked: {code_n} code + {docs_n} docs = {total} chunks total")

            yield _sse("embedding", f"Embedding {total} chunks via Ollama (this takes a minute)...")
            embed_and_store(chunks, conn)

            repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=api_key)

            _state["conn"] = conn
            _state["client"] = client
            _state["repo_name"] = repo_name
            _state["session"] = ChatSession(client, conn, repo_name=repo_name)

            yield _sse("done", f"Ready! {total} chunks indexed for '{repo_name}'.")

        except Exception as exc:
            yield _sse("error", str(exc))

    return Response(stream_with_context(generate()), content_type="text/event-stream")


@app.route("/api/chat", methods=["POST"])
def chat():
    if _state["session"] is None:
        return jsonify({"error": "No repository loaded. Index a repo first."}), 400

    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    # Retrieve chunks separately so we can return them for the "Show context" panel.
    # ChatSession.send also retrieves internally — the extra embed + DB call is
    # ~100ms and keeps query.py unchanged.
    chunks = retrieve(message, _state["conn"], repo_name=_state["repo_name"])
    context_text = _format_chunks(chunks)

    reply = _state["session"].send(message)

    return jsonify({
        "reply": reply,
        "chunks_used": len(chunks),
        "context": context_text,
    })


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
    app.run(debug=False, port=8080, threaded=False)
