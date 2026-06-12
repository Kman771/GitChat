"""query.py — multi-turn RAG chatbot with prompt caching on retrieved chunks."""

import sys

import anthropic
import psycopg2
from pgvector.psycopg2 import register_vector

from retrieval import retrieve_for_queries, RESULT_CAP
from config import DATABASE_URL, ANTHROPIC_API_KEY

SYSTEM_PROMPT = (
    "You are a senior engineer helping an intern understand a specific codebase. "
    "Rules:\n"
    "1. ALWAYS call search_codebase before answering any question. Never ask for clarification "
    "without searching first — use the codebase index to pick the most relevant identifiers "
    "and search for them. Even vague questions like 'how does it work' should trigger a search "
    "using entry-point names (e.g. main, run, start, or whatever the index shows).\n"
    "2. Only if the question is clearly unrelated to code (e.g. 'what is the weather') respond: "
    "\"I can only answer questions about this codebase.\"\n"
    "3. Match response length to question complexity — simple questions get 1-3 sentences, "
    "complex questions get a thorough answer.\n"
    "4. Always reference the relevant code from the retrieved chunks.\n"
    "5. Use markdown: backticks for identifiers, fenced blocks for code snippets, "
    "bullets for lists.\n"
    "6. Cite sources inline: each time you mention a function, class, or file from a retrieved "
    "chunk, link it using the GitHub URL provided in the 'Source:' line of that chunk. "
    "Format: [function_name](url) or [ClassName in file.py](url). Place the link immediately "
    "after the reference — do not add a separate Sources or References section at the end."
)
MODEL = "claude-haiku-4-5-20251001"
MAX_TOOL_ROUNDS = 3

SEARCH_TOOL = {
    "name": "search_codebase",
    "description": (
        "Search the indexed codebase for relevant source code chunks. "
        "Use precise technical vocabulary: function names, class names, module names, "
        "or specific concepts as they would appear in the source files. "
        "Prefer multiple focused terms over one broad question. "
        "The repository index shown at the start of the conversation lists all available file paths and identifiers — use those exact names as search terms. "
        "Call this once before answering — do not call it again unless the first results were completely empty."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "1-5 distinct search terms using code-level vocabulary: "
                    "identifiers, method names, class names, or technical concepts "
                    "likely to appear verbatim in the source files."
                ),
                "minItems": 1,
                "maxItems": 5,
            }
        },
        "required": ["queries"],
    },
}


def _build_repo_map(conn, repo_name: str | None) -> str:
    if not repo_name:
        return ""
    cur = conn.cursor()
    cur.execute(
        "SELECT file_path, chunk_type, metadata->>'name' "
        "FROM chunks WHERE repo_name = %s ORDER BY file_path",
        (repo_name,),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return ""
    by_file: dict[str, list[str]] = {}
    types: dict[str, str] = {}
    for file_path, chunk_type, name in rows:
        by_file.setdefault(file_path, [])
        if name:
            by_file[file_path].append(name)
        types[file_path] = chunk_type
    lines = ["Codebase index — use these identifiers when calling search_codebase:\n"]
    for fp in sorted(by_file):
        names = ", ".join(by_file[fp]) if by_file[fp] else "(no named chunks)"
        lines.append(f"[{types[fp]}] {fp} → {names}")
    return "\n".join(lines)


def _format_chunks(chunks: list[dict], repo_url: str | None = None, default_branch: str = "main") -> str:
    """Format chunks for tool results (with GitHub URLs) or the UI context panel."""
    if not chunks:
        return ""
    lines = ["Relevant source sections (retrieved by semantic similarity):"]
    for i, c in enumerate(chunks, 1):
        name = c["metadata"].get("name", "") if isinstance(c["metadata"], dict) else ""
        header = f"\n[{i}] {c['repo_name']}/{c['file_path']}"
        if name:
            header += f" · {name}"
        header += f" (similarity {c['similarity']:.3f})"
        if repo_url:
            gh_url = f"{repo_url.rstrip('/')}/blob/{default_branch}/{c['file_path']}"
            header += f"\nSource: {gh_url}"
        lines.append(f"{header}\n```\n{c['content']}\n```")
    return "\n".join(lines)


def _parse_response(response) -> str:
    """Extract text from all text content blocks in the response."""
    return "".join(block.text for block in response.content if block.type == "text")


class ChatSession:
    """Multi-turn RAG conversation with Claude, with prompt caching on retrieved chunks."""

    def __init__(self, client: anthropic.Anthropic, conn, repo_name: str | None = None, repo_url: str | None = None, default_branch: str = "main") -> None:
        self.client = client
        self.conn = conn
        self.repo_name = repo_name
        self.repo_url = repo_url
        self.default_branch = default_branch
        self.messages: list[dict] = []
        self.last_chunks: list[dict] = []
        self._repo_map_sent = False

    def send(self, user_query: str) -> str:
        if not self._repo_map_sent:
            repo_map = _build_repo_map(self.conn, self.repo_name)
            content = f"{repo_map}\n\n---\n{user_query}" if repo_map else user_query
            self._repo_map_sent = True
        else:
            content = user_query
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        self.last_chunks = []
        seen_ids: set[int] = set()
        tool_rounds = 0

        while True:
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
                tools=[SEARCH_TOOL],
                messages=self.messages,
            )

            u = response.usage
            print(
                f"\n[usage] input={u.input_tokens} | "
                f"cache_write={u.cache_creation_input_tokens} | "
                f"cache_read={u.cache_read_input_tokens} | "
                f"output={u.output_tokens}",
                file=sys.stderr,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks or tool_rounds >= MAX_TOOL_ROUNDS or len(self.last_chunks) >= RESULT_CAP:
                answer = _parse_response(response)
                # Compact: replace all intermediates with just question + plain text answer
                self.messages = self.messages[:turn_start] + [
                    self.messages[turn_start],
                    {"role": "assistant", "content": answer},
                ]
                return answer

            tool_rounds += 1
            self.messages.append({"role": "assistant", "content": response.content})
            print(f"\n[tool] round={tool_rounds} — Claude searching: {[b.input.get('queries') for b in tool_use_blocks]}", file=sys.stderr)

            tool_results = []
            for tb in tool_use_blocks:
                queries = tb.input.get("queries", [])
                chunks = retrieve_for_queries(queries, self.conn, repo_name=self.repo_name)
                new_chunks = [c for c in chunks if c["id"] not in seen_ids]
                for c in new_chunks:
                    seen_ids.add(c["id"])
                self.last_chunks.extend(new_chunks)
                result_text = _format_chunks(new_chunks, repo_url=self.repo_url, default_branch=self.default_branch) if new_chunks else "No new results — all relevant chunks already retrieved."
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": result_text,
                })

            self.messages.append({"role": "user", "content": tool_results})


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
