"""conftest.py — shared fixtures for all test modules."""

import os
import pytest
import psycopg2
from pgvector.psycopg2 import register_vector

REPO_URLS = [
    "https://github.com/bottlepy/bottle",       # pure Python + markdown
    "https://github.com/ssloy/tinyrenderer",    # C++ — tests generic regex chunker
    "https://github.com/mindsdb/minds-platform", # mixed Python at scale
]

DEFAULT_DB_URL = "postgresql://kaashishvenkat@localhost:5432/repo_rag"


@pytest.fixture(scope="function")
def db_conn():
    """Open a psycopg2 connection with pgvector registered. Caller is responsible for commit/rollback."""
    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def repo_urls():
    return REPO_URLS
