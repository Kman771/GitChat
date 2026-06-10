CREATE DATABASE repo_rag;
\c repo_rag

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    repo_name    TEXT        NOT NULL,
    file_path    TEXT        NOT NULL,
    chunk_type   TEXT        NOT NULL  CHECK (chunk_type IN ('code', 'docs', 'git_log')),
    content      TEXT        NOT NULL,
    metadata     JSONB,
    embedding    vector(768),
    created_at   TIMESTAMP   NOT NULL DEFAULT now()
);
