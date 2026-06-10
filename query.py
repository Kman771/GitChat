"""query.py — multi-turn RAG chatbot with prompt caching on retrieved chunks."""

import os
import sys

import anthropic
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from retrieval import retrieve

SYSTEM_PROMPT = "you are a senior engineer who can answer questions about any codebase to an intern"
MODEL = "claude-sonnet-4-6"
DEFAULT_DB_URL = "postgresql://kaashishvenkat@localhost:5432/repo_rag"


def _format_chunks(chunks: list[dict]) -> str:
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


class ChatSession:
    """Multi-turn RAG conversation with Claude, with prompt caching on retrieved chunks."""

    def __init__(self, client: anthropic.Anthropic, conn, repo_name: str | None = None) -> None:
        self.client = client
        self.conn = conn
        self.repo_name = repo_name
        self.messages: list[dict] = []

    def send(self, user_query: str) -> str:
        chunks = retrieve(user_query, self.conn, repo_name=self.repo_name)
        context_text = _format_chunks(chunks)

        # The context block carries cache_control so retrieved chunk tokens are
        # cached server-side and not re-billed on subsequent conversation turns.
        if context_text:
            user_content: str | list = [
                {
                    "type": "text",
                    "text": context_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": user_query},
            ]
        else:
            user_content = user_query

        self.messages.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=16_000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=self.messages,
        )

        reply = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
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
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    conn = psycopg2.connect(db_url)
    register_vector(conn)

    client = anthropic.Anthropic(api_key=api_key)
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
