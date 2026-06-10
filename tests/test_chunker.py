"""test_chunker.py — unit tests for chunker.py. No network, no DB required."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from chunker import (
    _chunk_python,
    _chunk_generic,
    _chunk_markdown,
    _chunk_rst,
    _split_half,
    _token_count,
    _truncate,
    MAX_TOKENS,
)

REQUIRED_KEYS = {"name", "file_path", "repo_name", "content"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def assert_chunk_structure(chunks: list[dict]):
    for chunk in chunks:
        assert REQUIRED_KEYS == REQUIRED_KEYS & chunk.keys(), (
            f"Chunk missing keys: {REQUIRED_KEYS - chunk.keys()}"
        )
        assert chunk["content"], "chunk content must be non-empty"


# ── Python chunking ────────────────────────────────────────────────────────────

PYTHON_SINGLE_FUNC = """\
def greet(name: str) -> str:
    return f"Hello, {name}"
"""

PYTHON_CLASS_WITH_METHODS = """\
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        return "Woof!"
"""

PYTHON_EMPTY = ""


def test_single_function():
    chunks = _chunk_python(PYTHON_SINGLE_FUNC, "greet.py", "myrepo")
    assert len(chunks) == 1
    assert chunks[0]["name"] == "greet"
    assert "def greet" in chunks[0]["content"]


def test_class_and_methods():
    chunks = _chunk_python(PYTHON_CLASS_WITH_METHODS, "dog.py", "myrepo")
    names = [c["name"] for c in chunks]
    # Expect the class itself + each method
    assert "Dog" in names
    assert "Dog.__init__" in names
    assert "Dog.bark" in names
    assert len(chunks) == 3


def test_empty_python_file():
    chunks = _chunk_python(PYTHON_EMPTY, "empty.py", "myrepo")
    assert chunks == []


def test_python_chunks_have_required_keys():
    chunks = _chunk_python(PYTHON_CLASS_WITH_METHODS, "dog.py", "myrepo")
    assert_chunk_structure(chunks)


# ── Generic chunking ───────────────────────────────────────────────────────────

JS_FUNCTION = """\
function processOrder(order) {
    return order.total * 1.1;
}
"""

TS_ARROW = """\
const fetchUser = async (id: number) => {
    const res = await api.get(`/users/${id}`);
    return res.data;
};
"""

GO_FUNC = """\
func MyHandler(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
}
"""

GO_METHOD = """\
func (s *Server) Run(addr string) error {
    return http.ListenAndServe(addr, s.mux)
}
"""


def test_js_function_keyword():
    chunks = _chunk_generic(JS_FUNCTION, "order.js", "myrepo")
    names = [c["name"] for c in chunks]
    assert "processOrder" in names


def test_ts_arrow_const():
    chunks = _chunk_generic(TS_ARROW, "api.ts", "myrepo")
    names = [c["name"] for c in chunks]
    assert "fetchUser" in names


def test_go_func():
    chunks = _chunk_generic(GO_FUNC, "handler.go", "myrepo")
    names = [c["name"] for c in chunks]
    assert "MyHandler" in names


def test_go_method_receiver():
    chunks = _chunk_generic(GO_METHOD, "server.go", "myrepo")
    names = [c["name"] for c in chunks]
    assert "Run" in names


def test_generic_chunks_have_required_keys():
    chunks = _chunk_generic(JS_FUNCTION, "order.js", "myrepo")
    assert_chunk_structure(chunks)


# ── Markdown chunking ──────────────────────────────────────────────────────────

MARKDOWN_THREE_HEADERS = """\
## Installation

Run `pip install mylib` to get started.

## Usage

Import and call the main function.

## Contributing

Open a pull request on GitHub.
"""

MARKDOWN_NO_HEADERS = "This is plain prose with no headers at all."

# Build a section body that exceeds 500 tokens
MARKDOWN_OVERSIZED = "## Big Section\n\n" + ("word " * 600)


def test_headers_split_sections():
    chunks = _chunk_markdown(MARKDOWN_THREE_HEADERS, "README.md", "myrepo")
    names = [c["name"] for c in chunks]
    assert "Installation" in names
    assert "Usage" in names
    assert "Contributing" in names


def test_no_headers_uses_file_stem():
    chunks = _chunk_markdown(MARKDOWN_NO_HEADERS, "intro.md", "myrepo")
    assert len(chunks) == 1
    assert chunks[0]["name"] == "intro"


def test_oversized_section_uses_paragraphs():
    chunks = _chunk_markdown(MARKDOWN_OVERSIZED, "big.md", "myrepo")
    # Every chunk must be within the token limit
    for chunk in chunks:
        assert _token_count(chunk["content"]) <= MAX_TOKENS, (
            f"Chunk '{chunk['name']}' exceeds {MAX_TOKENS} tokens"
        )


def test_markdown_chunks_have_required_keys():
    chunks = _chunk_markdown(MARKDOWN_THREE_HEADERS, "README.md", "myrepo")
    assert_chunk_structure(chunks)


# ── RST chunking ──────────────────────────────────────────────────────────────

RST_THREE_SECTIONS = """\
Introduction
============

This section covers the basics.

Installation
------------

Run pip install to get started.

Usage
~~~~~

Import and call the main function.
"""

RST_NO_HEADERS = "Plain RST prose with no headers at all."

RST_OVERSIZED = "Big Section\n===========\n\n" + ("word " * 600)


def test_rst_headers_split_sections():
    chunks = _chunk_rst(RST_THREE_SECTIONS, "docs/guide.rst", "myrepo")
    names = [c["name"] for c in chunks]
    assert "Introduction" in names
    assert "Installation" in names
    assert "Usage" in names


def test_rst_no_headers_uses_file_stem():
    chunks = _chunk_rst(RST_NO_HEADERS, "docs/intro.rst", "myrepo")
    assert len(chunks) == 1
    assert chunks[0]["name"] == "intro"


def test_rst_oversized_section_splits():
    chunks = _chunk_rst(RST_OVERSIZED, "docs/big.rst", "myrepo")
    for chunk in chunks:
        assert _token_count(chunk["content"]) <= MAX_TOKENS


def test_rst_chunks_have_required_keys():
    chunks = _chunk_rst(RST_THREE_SECTIONS, "docs/guide.rst", "myrepo")
    assert_chunk_structure(chunks)


# ── split_half ─────────────────────────────────────────────────────────────────

def test_split_half_produces_multiple_chunks():
    long_text = "token " * 600
    chunks = _split_half(long_text, "myname", "file.py", "myrepo")
    assert len(chunks) >= 2


def test_split_half_each_under_limit():
    long_text = "token " * 600
    chunks = _split_half(long_text, "myname", "file.py", "myrepo")
    for chunk in chunks:
        assert _token_count(chunk["content"]) <= MAX_TOKENS


def test_split_half_preserves_name():
    long_text = "token " * 600
    chunks = _split_half(long_text, "myname", "file.py", "myrepo")
    assert all(c["name"] == "myname" for c in chunks)


# ── Token counting and truncation ──────────────────────────────────────────────

def test_token_count_short_string():
    # "hello world" is 2 tokens in cl100k_base
    assert _token_count("hello world") == 2


def test_truncate_within_limit():
    short = "short text"
    assert _truncate(short) == short


def test_truncate_respects_limit():
    long_text = "token " * 1000  # ~1000 tokens
    result = _truncate(long_text)
    assert _token_count(result) <= MAX_TOKENS
