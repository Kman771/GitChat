"""conftest.py — shared fixtures for all test modules."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import psycopg2
from pgvector.psycopg2 import register_vector
from config import DATABASE_URL

REPO_URLS = [
    "https://github.com/bottlepy/bottle",
    "https://github.com/ssloy/tinyrenderer",
    "https://github.com/mindsdb/minds-platform",
]


@pytest.fixture(scope="function")
def db_conn():
    """Open a psycopg2 connection with pgvector registered. Caller is responsible for commit/rollback."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def repo_urls():
    return REPO_URLS
