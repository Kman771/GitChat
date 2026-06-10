"""chunker.py — clone a GitHub repo and chunk it into code objects and doc sections."""

import ast
import re
import tempfile
from pathlib import Path

import tiktoken
from git import Repo

MAX_TOKENS = 500

CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rs',
    '.cpp', '.c', '.h', '.cs', '.rb', '.php', '.swift', '.kt',
}
DOCS_EXTENSIONS = {'.md', '.mdx', '.rst'}

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def _truncate(text: str, max_tokens: int = MAX_TOKENS) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


def _split_half(text: str, name: str, file_path: str, repo_name: str) -> list[dict]:
    """Recursively halve text until every piece is ≤ MAX_TOKENS, keeping equal-sized chunks."""
    tokens = _enc.encode(text)
    if len(tokens) <= MAX_TOKENS:
        return [{"name": name, "file_path": file_path, "repo_name": repo_name, "content": text}]
    mid = len(tokens) // 2
    first = _enc.decode(tokens[:mid])
    second = _enc.decode(tokens[mid:])
    chunks = []
    if first.strip():
        chunks.extend(_split_half(first, name, file_path, repo_name))
    if second.strip():
        chunks.extend(_split_half(second, name, file_path, repo_name))
    return chunks


def _enforce_limit(chunks: list[dict]) -> list[dict]:
    """Safety net: any chunk still over MAX_TOKENS (e.g. from separator token drift) gets halved."""
    result = []
    for chunk in chunks:
        if _token_count(chunk["content"]) > MAX_TOKENS:
            result.extend(_split_half(chunk["content"], chunk["name"], chunk["file_path"], chunk["repo_name"]))
        else:
            result.append(chunk)
    return result


# ── Code chunking ──────────────────────────────────────────────────────────────

def _chunk_python(source: str, file_path: str, repo_name: str) -> list[dict]:
    chunks = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    lines = source.splitlines()

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        body = '\n'.join(lines[node.lineno - 1 : node.end_lineno])
        if _token_count(body) <= MAX_TOKENS:
            chunks.append({"name": node.name, "file_path": file_path, "repo_name": repo_name, "content": body})
        else:
            chunks.extend(_split_half(body, node.name, file_path, repo_name))

        # Also surface each method inside a class as its own chunk
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_body = '\n'.join(lines[child.lineno - 1 : child.end_lineno])
                    if _token_count(method_body) <= MAX_TOKENS:
                        chunks.append({"name": f"{node.name}.{child.name}", "file_path": file_path, "repo_name": repo_name, "content": method_body})
                    else:
                        chunks.extend(_split_half(method_body, f"{node.name}.{child.name}", file_path, repo_name))

    return _enforce_limit(chunks)


# Covers JS/TS, Go, Rust, Java/C#, Ruby
_FUNC_RE = re.compile(
    '|'.join([
        r'(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*[(<]',
        r'(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(',
        r'(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)',
        r'func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?([A-Za-z_]\w*)\s*\(',
        r'(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*[(<]',
        r'(?:public|private|protected)\s+(?:static\s+)?(?:async\s+)?[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\(',
        r'def\s+([A-Za-z_]\w*)',
    ]),
    re.MULTILINE,
)


def _chunk_generic(source: str, file_path: str, repo_name: str) -> list[dict]:
    chunks = []
    lines = source.splitlines()

    for match in _FUNC_RE.finditer(source):
        name = next(g for g in match.groups() if g is not None)
        line_idx = source[: match.start()].count('\n')

        block_lines: list[str] = []
        token_count = 0
        for line in lines[line_idx:]:
            line_tokens = _token_count(line + '\n')
            if token_count + line_tokens > MAX_TOKENS and block_lines:
                break
            block_lines.append(line)
            token_count += line_tokens

        chunks.append({
            "name": name,
            "file_path": file_path,
            "repo_name": repo_name,
            "content": '\n'.join(block_lines),
        })

    return chunks


def _chunk_code_file(path: Path, file_path: str, repo_name: str) -> list[dict]:
    try:
        source = path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return []
    if path.suffix == '.py':
        return _chunk_python(source, file_path, repo_name)
    return _chunk_generic(source, file_path, repo_name)


# ── Docs chunking ──────────────────────────────────────────────────────────────

def _split_by_paragraphs(body: str, header: str, file_path: str, repo_name: str) -> list[dict]:
    """Break an oversized section into paragraph-sized sub-chunks."""
    chunks: list[dict] = []
    paragraphs = re.split(r'\n{2,}', body.strip())
    current_parts: list[str] = []
    current_tokens = 0

    def _flush():
        nonlocal current_parts, current_tokens
        if current_parts:
            chunks.append({
                "name": header,
                "file_path": file_path,
                "repo_name": repo_name,
                "content": '\n\n'.join(current_parts),
            })
            current_parts = []
            current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = _token_count(para)

        if para_tokens > MAX_TOKENS:
            # Flush whatever has accumulated, then split this paragraph into two equal halves
            _flush()
            chunks.extend(_split_half(para, header, file_path, repo_name))
            continue

        if current_tokens + para_tokens > MAX_TOKENS:
            _flush()

        current_parts.append(para)
        current_tokens += para_tokens

    _flush()
    return chunks


# RST underline-style headers: a title line followed by ≥4 repeated adornment chars.
# e.g.  "Section\n======="  or  "Subsection\n----------"
_RST_HEADER_RE = re.compile(
    r'^(?![ \t])(.+)\n([=\-~^+#*\'"`.:_])\2{3,}[ \t]*$',
    re.MULTILINE,
)

_HEADER_RE = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)


def _chunk_markdown(source: str, file_path: str, repo_name: str) -> list[dict]:
    file_stem = Path(file_path).stem
    parts = _HEADER_RE.split(source)
    # split result: [pre-text, header, body, header, body, ...]

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
            chunks.append({
                "name": header,
                "file_path": file_path,
                "repo_name": repo_name,
                "content": combined,
            })
        else:
            chunks.extend(_split_by_paragraphs(body, header, file_path, repo_name))

    return _enforce_limit(chunks)


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
            chunks.append({"name": header, "file_path": file_path, "repo_name": repo_name, "content": combined})
        else:
            chunks.extend(_split_by_paragraphs(body, header, file_path, repo_name))

    return _enforce_limit(chunks)


def _chunk_docs_file(path: Path, file_path: str, repo_name: str) -> list[dict]:
    try:
        source = path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return []
    if path.suffix == '.rst':
        return _chunk_rst(source, file_path, repo_name)
    return _chunk_markdown(source, file_path, repo_name)


# ── Main entry point ───────────────────────────────────────────────────────────

def chunk_repo(repo_url: str) -> dict[str, list[dict]]:
    """Clone *repo_url* and return ``{'code': [...], 'docs': [...]}``.

    Each code chunk: {'name', 'file_path', 'repo_name', 'content'}
    Each docs chunk: {'name', 'file_path', 'repo_name', 'content'}
    """
    repo_name = repo_url.rstrip('/').split('/')[-1].removesuffix('.git')
    result: dict[str, list[dict]] = {"code": [], "docs": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        Repo.clone_from(repo_url, tmpdir, depth=1)
        root = Path(tmpdir)

        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            # Skip anything inside hidden directories (e.g. .git)
            if any(part.startswith('.') for part in rel_parts[:-1]):
                continue

            rel = str(path.relative_to(root))
            suffix = path.suffix.lower()

            if suffix in CODE_EXTENSIONS:
                result["code"].extend(_chunk_code_file(path, rel, repo_name))
            elif suffix in DOCS_EXTENSIONS:
                result["docs"].extend(_chunk_docs_file(path, rel, repo_name))

    return result


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python chunker.py <github-repo-url>")
        sys.exit(1)

    chunks = chunk_repo(sys.argv[1])
    print(f"Code chunks : {len(chunks['code'])}")
    print(f"Docs chunks : {len(chunks['docs'])}")
    print(json.dumps(chunks, indent=2)[:3000], "\n...")
