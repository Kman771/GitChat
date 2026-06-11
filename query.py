"""query.py — multi-turn RAG chatbot with prompt caching on retrieved chunks."""

import sys

import anthropic
import psycopg2
from pgvector.psycopg2 import register_vector

from retrieval import retrieve
from config import DATABASE_URL, ANTHROPIC_API_KEY

SYSTEM_PROMPT = (
    "You are a senior engineer helping an intern understand a specific codebase. "
    "Rules:\n"
    "1. Only answer questions about the codebase. For anything else, respond exactly: "
    "\"I can only answer questions about this codebase.\"\n"
    "2. Match response length to question complexity — simple questions get 1-3 sentences, "
    "complex questions get a thorough answer.\n"
    "3. Always reference the relevant code from the provided source chunks.\n"
    "4. Use markdown: backticks for identifiers, fenced blocks for code snippets, "
    "bullets for lists."
)
MODEL = "claude-sonnet-4-6"


def _format_chunks(chunks: list[dict]) -> str:
    """Format chunks as plain text for the UI 'Show context' panel."""
    if not chunks:
        return ""
    lines = ["Relevant source sections (retrieved by semantic similarity):"]
    for i, c in enumerate(chunks, 1):
        name = c["metadata"].get("name", "") if isinstance(c["metadata"], dict) else ""
        header = f"\n[{i}] {c['repo_name']}/{c['file_path']}"
        if name:
            header += f" · {name}"
        header += f" (similarity {c['similarity']:.3f})"
        lines.append(f"{header}\n```\n{c['content']}\n```")
    return "\n".join(lines)


def _chunks_to_documents(chunks: list[dict]) -> list[dict]:
    """Convert retrieved chunks to document content blocks with citations enabled."""
    docs = []
    for c in chunks:
        name = c["metadata"].get("name", "") if isinstance(c["metadata"], dict) else ""
        title = f"{c['repo_name']}/{c['file_path']}"
        if name:
            title += f" · {name}"
        docs.append({
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c["content"]},
            "title": title,
            "citations": {"enabled": True},
        })
    if docs:
        docs[-1]["cache_control"] = {"type": "ephemeral"}
    return docs


def _parse_cited_response(response) -> str:
    """Reconstruct reply text from content blocks, appending a Sources section."""
    parts = []
    refs: list[tuple[str, str]] = []  # (document_title, cited_text) in first-seen order

    for block in response.content:
        if block.type != "text":
            continue
        citations = getattr(block, "citations", None) or []
        parts.append(block.text)
        for cit in citations:
            key = (cit.document_title, cit.cited_text)
            if key not in refs:
                refs.append(key)
            parts.append(f"[{refs.index(key) + 1}]")

    reply = "".join(parts)
    if refs:
        reply += "\n\n---\n**Sources**\n"
        for i, (title, cited) in enumerate(refs, 1):
            snippet = cited[:100] + ("…" if len(cited) > 100 else "")
            reply += f'[{i}] `{title}` — *"{snippet}"*\n'
    return reply


class ChatSession:
    """Multi-turn RAG conversation with Claude, with prompt caching on retrieved chunks."""

    def __init__(self, client: anthropic.Anthropic, conn, repo_name: str | None = None) -> None:
        self.client = client
        self.conn = conn
        self.repo_name = repo_name
        self.messages: list[dict] = []

    def send(self, user_query: str) -> str:
        chunks = retrieve(user_query, self.conn, repo_name=self.repo_name)

        if chunks:
            # Pass chunks as document blocks so Claude can cite specific passages.
            # cache_control is applied to the last document, caching the full set.
            user_content: str | list = _chunks_to_documents(chunks) + [
                {"type": "text", "text": user_query}
            ]
        else:
            user_content = user_query

        self.messages.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=self.messages,
        )

        reply = _parse_cited_response(response)
        self.messages.append({"role": "assistant", "content": reply})

        u = response.usage
        print(
            f"\n[usage] input={u.input_tokens} | "
            f"cache_write={u.cache_creation_input_tokens} | "
            f"cache_read={u.cache_read_input_tokens} | "
            f"output={u.output_tokens}",
            file=sys.stderr,
        )

        return reply


def main(repo_name: str | None = None) -> None:
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    session = ChatSession(client, conn, repo_name=repo_name)

    scope = f" (repo: {repo_name})" if repo_name else " (all repos)"
    print(f"GitChat{scope} — type 'exit' or Ctrl-D to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        answer = session.send(user_input)
        print(f"\nClaude: {answer}\n")

    conn.close()


if __name__ == "__main__":
    rn = sys.argv[1] if len(sys.argv) > 1 else None
    main(rn)
