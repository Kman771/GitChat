"""chunker.py — clone a GitHub repo and chunk it into code and prose chunks.

Code files: parsed with tree-sitter; each top-level function/class is one chunk (no size limit).
Prose files: split by headers then paragraphs with 15% overlap, capped at 500 tokens.
"""

import re
import sys
import tempfile
import warnings
from pathlib import Path

import tiktoken
from git import Repo

warnings.filterwarnings("ignore", category=FutureWarning)  # suppress tree-sitter 0.21 deprecation

MAX_TOKENS = 500      # prose only — code chunks are never split
OVERLAP_TOKENS = 75   # 15% of 500 for paragraph overlap
MIN_TOKENS = 10       # quality filter — skip tiny chunks

# Extension → tree-sitter language name
TREE_SITTER_LANGUAGES: dict[str, str] = {
    '.py': 'python',
    '.js': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'tsx',
    '.go': 'go',
    '.java': 'java',
    '.rs': 'rust',
    '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.cs': 'c_sharp',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
}

# Top-level node types to extract per language (direct children of root)
_TOP_LEVEL: dict[str, set[str]] = {
    'python':     {'function_definition', 'class_definition', 'decorated_definition'},
    'javascript': {'function_declaration', 'class_declaration', 'lexical_declaration', 'export_statement'},
    'typescript': {'function_declaration', 'class_declaration', 'lexical_declaration', 'export_statement', 'ambient_declaration'},
    'tsx':        {'function_declaration', 'class_declaration', 'lexical_declaration', 'export_statement'},
    'go':         {'function_declaration', 'method_declaration', 'type_declaration'},
    'java':       {'class_declaration', 'interface_declaration', 'enum_declaration'},
    'rust':       {'function_item', 'impl_item', 'struct_item', 'enum_item', 'trait_item'},
    'cpp':        {'function_definition', 'class_specifier', 'namespace_definition'},
    'c':          {'function_definition'},
    'c_sharp':    {'class_declaration', 'interface_declaration', 'method_declaration'},
    'ruby':       {'method', 'class', 'module'},
    'php':        {'function_definition', 'class_declaration'},
    'swift':      {'function_declaration', 'class_declaration', 'struct_declaration'},
    'kotlin':     {'function_declaration', 'class_declaration', 'object_declaration'},
}

DOCS_EXTENSIONS = {'.md', '.mdx', '.rst', '.txt'}

# Files with these extensions are skipped entirely — binary, data, or too large to be useful
SKIP_EXTENSIONS = {
    # Jupyter notebooks (can contain MB of base64 image outputs)
    '.ipynb',
    # Binary ML/data files
    '.pkl', '.pickle', '.h5', '.hdf5', '.npy', '.npz', '.pt', '.pth', '.ckpt', '.pb',
    # Images and media
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico', '.webp',
    '.mp4', '.avi', '.mov', '.mp3', '.wav',
    # Archives
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
    # Documents and data
    '.pdf', '.csv', '.parquet', '.feather', '.arrow',
    # Compiled / binary
    '.pyc', '.pyo', '.so', '.dll', '.exe', '.o', '.a', '.lib',
    # Lock files and generated
    '.lock', '.sum',
}

# Skip files larger than this (catches binary-disguised-as-text and huge generated files)
MAX_FILE_BYTES = 200 * 1024  # 200 KB

# Hard cap on tokens per chunk sent to the embedder — nomic-embed-text context window
MAX_EMBED_TOKENS = 4000

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def _make_chunk(
    repo_name: str,
    file_path: str,
    chunk_type: str,
    chunk_name: str,
    content: str,
) -> dict | None:
    """Build a chunk dict after stripping whitespace and applying the MIN_TOKENS filter."""
    content = content.strip().replace('\x00', '')
    token_count = _token_count(content)
    if token_count < MIN_TOKENS:
        return None
    return {
        "repo_name": repo_name,
        "file_path": file_path,
        "chunk_type": chunk_type,
        "chunk_name": chunk_name,
        "content": content,
        "token_count": token_count,
    }


# ── Tree-sitter helpers ────────────────────────────────────────────────────────

def _get_parser(lang_name: str):
    """Return a tree-sitter Parser for *lang_name*, or None if unavailable."""
    try:
        from tree_sitter_languages import get_parser
        return get_parser(lang_name)
    except Exception:
        return None


def _get_node_name(node, source_bytes: bytes) -> str:
    """Extract the identifier name from a tree-sitter node."""
    # For decorated definitions (Python), look inside
    if node.type == 'decorated_definition':
        defn = node.child_by_field_name('definition')
        if defn:
            node = defn

    # Standard 'name' field
    name_node = node.child_by_field_name('name')
    if name_node:
        return source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8', errors='replace')

    # Fallback: first named identifier child
    for child in node.children:
        if child.type == 'identifier':
            return source_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='replace')

    return node.type


# ── Code chunking (tree-sitter) ────────────────────────────────────────────────

def _chunk_code_with_tree_sitter(
    source_bytes: bytes,
    lang_name: str,
    file_path: str,
    repo_name: str,
) -> list[dict]:
    parser = _get_parser(lang_name)
    if parser is None:
        return []

    tree = parser.parse(source_bytes)
    root = tree.root_node
    target_types = _TOP_LEVEL.get(lang_name, set())

    nodes = [child for child in root.children if child.type in target_types]

    if not nodes:
        # No functions or classes found — whole file is one chunk
        file_stem = Path(file_path).stem
        chunk = _make_chunk(
            repo_name, file_path, "code", file_stem,
            source_bytes.decode('utf-8', errors='replace'),
        )
        return [chunk] if chunk else []

    chunks = []
    for node in nodes:
        text = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
        name = _get_node_name(node, source_bytes)
        chunk = _make_chunk(repo_name, file_path, "code", name, text)
        if chunk:
            chunks.append(chunk)
    return chunks


def _looks_like_code(lines: list[str]) -> bool:
    """Heuristic: does a file look like code even without a known extension?"""
    if not lines:
        return False
    code_count = sum(
        1 for line in lines
        if line.startswith((' ', '\t')) or re.search(r'[{};()=]', line)
    )
    return code_count / len(lines) > 0.5


# ── Prose chunking ─────────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
_RST_HEADER_RE = re.compile(
    r'^(?![ \t])(.+)\n([=\-~^+#*\'"`.:_])\2{3,}[ \t]*$',
    re.MULTILINE,
)


def _split_by_paragraphs(
    body: str, header: str, file_path: str, repo_name: str
) -> list[dict]:
    """Break an oversized prose section into paragraph chunks with 15% overlap."""
    chunks: list[dict] = []
    paragraphs = re.split(r'\n{2,}', body.strip())
    current_parts: list[str] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current_parts, current_tokens
        if not current_parts:
            return
        chunk = _make_chunk(
            repo_name, file_path, "docs", header,
            '\n\n'.join(current_parts),
        )
        if chunk:
            chunks.append(chunk)
        # Carry the last paragraph as overlap prefix if it fits the budget
        overlap_para = current_parts[-1]
        if _token_count(overlap_para) <= OVERLAP_TOKENS:
            current_parts = [overlap_para]
            current_tokens = _token_count(overlap_para)
        else:
            current_parts = []
            current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = _token_count(para)

        if para_tokens > MAX_TOKENS:
            _flush()
            # Split oversized paragraph by halving
            tokens = _enc.encode(para)
            mid = len(tokens) // 2
            overlap_start = max(0, mid - OVERLAP_TOKENS)
            for piece in [
                _enc.decode(tokens[:mid]),
                _enc.decode(tokens[overlap_start:]),
            ]:
                c = _make_chunk(repo_name, file_path, "docs", header, piece)
                if c:
                    chunks.append(c)
            current_parts = []
            current_tokens = 0
            continue

        if current_tokens + para_tokens > MAX_TOKENS:
            _flush()

        current_parts.append(para)
        current_tokens += para_tokens

    # Final flush — no trailing overlap needed
    if current_parts:
        chunk = _make_chunk(
            repo_name, file_path, "docs", header,
            '\n\n'.join(current_parts),
        )
        if chunk:
            chunks.append(chunk)
    return chunks


def _chunk_markdown(source: str, file_path: str, repo_name: str) -> list[dict]:
    file_stem = Path(file_path).stem
    parts = _HEADER_RE.split(source)
    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        sections.append((file_stem, parts[0]))
    for i in range(1, len(parts), 2):
        header_text = parts[i].lstrip('#').strip()
        body = parts[i + 1] if i + 1 < len(parts) else ''
        sections.append((header_text, body))
    if not sections:
        sections = [(file_stem, source)]

    chunks: list[dict] = []
    for header, body in sections:
        combined = f"{header}\n\n{body.strip()}".strip()
        if not combined:
            continue
        if _token_count(combined) <= MAX_TOKENS:
            c = _make_chunk(repo_name, file_path, "docs", header, combined)
            if c:
                chunks.append(c)
        else:
            chunks.extend(_split_by_paragraphs(body, header, file_path, repo_name))
    return chunks


def _chunk_rst(source: str, file_path: str, repo_name: str) -> list[dict]:
    file_stem = Path(file_path).stem
    matches = list(_RST_HEADER_RE.finditer(source))
    sections: list[tuple[str, str]] = []
    if not matches:
        sections = [(file_stem, source)]
    else:
        pre = source[:matches[0].start()].strip()
        if pre:
            sections.append((file_stem, pre))
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
            body = source[body_start:body_end].strip()
            sections.append((title, body))

    chunks: list[dict] = []
    for header, body in sections:
        combined = f"{header}\n\n{body}".strip() if body else header
        if not combined:
            continue
        if _token_count(combined) <= MAX_TOKENS:
            c = _make_chunk(repo_name, file_path, "docs", header, combined)
            if c:
                chunks.append(c)
        else:
            chunks.extend(_split_by_paragraphs(body, header, file_path, repo_name))
    return chunks


def _chunk_plain_prose(source: str, file_path: str, repo_name: str) -> list[dict]:
    """Chunk a plain prose file (no headers) by paragraphs."""
    file_stem = Path(file_path).stem
    return _split_by_paragraphs(source, file_stem, file_path, repo_name)


# ── File processor ─────────────────────────────────────────────────────────────

def _truncate_to_embed_limit(chunks: list[dict]) -> list[dict]:
    """Truncate any chunk whose content exceeds MAX_EMBED_TOKENS so Ollama accepts it."""
    result = []
    for c in chunks:
        tokens = _enc.encode(c["content"])
        if len(tokens) > MAX_EMBED_TOKENS:
            c = dict(c)
            c["content"] = _enc.decode(tokens[:MAX_EMBED_TOKENS])
            c["token_count"] = MAX_EMBED_TOKENS
        result.append(c)
    return result


def _process_file(path: Path, rel: str, repo_name: str) -> list[dict]:
    suffix = path.suffix.lower()
    filename = path.name

    # ── Skip by extension ──
    if suffix in SKIP_EXTENSIONS:
        return []

    # ── Skip large files ──
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            print(f"[skip] {rel}: file too large ({path.stat().st_size // 1024} KB)", file=sys.stderr)
            return []
    except OSError:
        return []

    try:
        raw = path.read_bytes()
    except OSError as e:
        raise RuntimeError(f"could not read: {e}") from e

    # Skip binary files — NUL bytes mean binary data that PostgreSQL will reject
    if b'\x00' in raw:
        print(f"[skip] {rel}: binary file (NUL bytes detected)", file=sys.stderr)
        return []

    source = raw.decode('utf-8', errors='ignore')

    # ── Prose by extension ──
    if suffix in DOCS_EXTENSIONS:
        if suffix == '.rst':
            return _chunk_rst(source, rel, repo_name)
        return _chunk_markdown(source, rel, repo_name)

    # ── Code by tree-sitter ──
    if suffix in TREE_SITTER_LANGUAGES:
        lang_name = TREE_SITTER_LANGUAGES[suffix]
        source_bytes = source.encode('utf-8')
        chunks = _chunk_code_with_tree_sitter(source_bytes, lang_name, rel, repo_name)
        if chunks:
            return _truncate_to_embed_limit(chunks)
        # tree-sitter returned nothing (parse error or empty file) → fall through

    # ── Fallback: heuristic classification ──
    lines = source.splitlines()
    sample = lines[:50]

    if _looks_like_code(sample):
        file_stem = Path(rel).stem or filename
        chunk = _make_chunk(repo_name, rel, "code", file_stem, source)
        return _truncate_to_embed_limit([chunk]) if chunk else []
    else:
        return _chunk_plain_prose(source, rel, repo_name)


# ── Main entry point ───────────────────────────────────────────────────────────

def normalize_github_url(url: str) -> str:
    """Strip GitHub branch/blob navigation suffixes so git clone works.

    https://github.com/owner/repo/tree/branch  →  https://github.com/owner/repo
    https://github.com/owner/repo/blob/...      →  https://github.com/owner/repo
    """
    return re.sub(r'/(?:tree|blob)/.*$', '', url.rstrip('/'))


def chunk_repo(repo_url: str) -> list[dict]:
    """Clone *repo_url* and return a flat list of chunk dicts.

    Each chunk: {repo_name, file_path, chunk_type, chunk_name, content, token_count}
    """
    repo_url = normalize_github_url(repo_url)
    repo_name = repo_url.split('/')[-1].removesuffix('.git')
    all_chunks: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        Repo.clone_from(repo_url, tmpdir, depth=1)
        root = Path(tmpdir)

        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue

            rel_parts = path.relative_to(root).parts
            # Skip files inside hidden directories, but allow dotfiles at root level
            if any(part.startswith('.') for part in rel_parts[:-1]):
                continue

            rel = str(path.relative_to(root))

            try:
                chunks = _process_file(path, rel, repo_name)
                all_chunks.extend(chunks)
            except Exception as exc:
                print(f"[skip] {rel}: {exc}", file=sys.stderr)

    return all_chunks


if __name__ == "__main__":
    import json

    if len(sys.argv) != 2:
        print("Usage: python chunker.py <github-repo-url>")
        sys.exit(1)

    chunks = chunk_repo(sys.argv[1])
    code_n = sum(1 for c in chunks if c["chunk_type"] == "code")
    docs_n = sum(1 for c in chunks if c["chunk_type"] == "docs")
    print(f"Total chunks : {len(chunks)}")
    print(f"Code chunks  : {code_n}")
    print(f"Docs chunks  : {docs_n}")
    print()
    print(json.dumps(chunks[:3], indent=2))
