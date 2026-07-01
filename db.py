"""PostgreSQL backend for both the job queue and its logs.

Log tables (one row per log record):
    kyc_logs      — SmartAadharDetector (kyc.py) logs + failures
    summary_logs  — PDF summarisation (summary.py) logs
    queue_logs    — queue engine (queue_system.py) logs

Job queue table (one row per queued job — the FIFO queue itself, replacing
the old SQLite queue.db):
    queue_items   — item_id, processor_name, data, enqueue_time, status

Connection is configured via environment variables and is fully optional for
logging — if Postgres is unreachable, log handlers fall back to console-only
so the processing pipeline never crashes because of its logging backend.
The job queue itself requires Postgres to be reachable, since jobs have
nowhere else to live.

Env vars:
    DATABASE_URL   — full Postgres DSN, overrides the PG_* vars below
    PG_HOST        — default "localhost"
    PG_PORT        — default "5432"
    PG_DATABASE    — default "newkycsummary"
    PG_USER        — default "postgres"
    PG_PASSWORD    — default ""
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

import psycopg2
import psycopg2.pool

_DATABASE_URL = os.getenv("DATABASE_URL")
_PG_HOST = os.getenv("PG_HOST", "localhost")
_PG_PORT = os.getenv("PG_PORT", "5432")
_PG_DATABASE = os.getenv("PG_DATABASE", "newkycsummary")
_PG_USER = os.getenv("PG_USER", "postgres")
_PG_PASSWORD = os.getenv("PG_PASSWORD", "")

_setup_logger = logging.getLogger("db")

_pool_lock = threading.Lock()
_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_disabled = False  # set True once Postgres is confirmed unreachable

_TABLE_DDL = {
    "kyc_logs": """
        CREATE TABLE IF NOT EXISTS kyc_logs (
            id         SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            level      TEXT NOT NULL,
            logger     TEXT NOT NULL,
            message    TEXT NOT NULL,
            zip_name   TEXT,
            trace_id   TEXT,
            stage      TEXT
        )
    """,
    "summary_logs": """
        CREATE TABLE IF NOT EXISTS summary_logs (
            id         SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            level      TEXT NOT NULL,
            logger     TEXT NOT NULL,
            message    TEXT NOT NULL,
            url        TEXT,
            news_id    INTEGER
        )
    """,
    "queue_logs": """
        CREATE TABLE IF NOT EXISTS queue_logs (
            id         SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            level      TEXT NOT NULL,
            logger     TEXT NOT NULL,
            message    TEXT NOT NULL,
            processor  TEXT,
            item_id    TEXT
        )
    """,
}

_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_kyc_logs_created_at ON kyc_logs (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_summary_logs_created_at ON summary_logs (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_queue_logs_created_at ON queue_logs (created_at)",
)

_QUEUE_ITEMS_DDL = """
    CREATE TABLE IF NOT EXISTS queue_items (
        item_id        TEXT PRIMARY KEY,
        processor_name TEXT NOT NULL,
        data           TEXT NOT NULL,
        enqueue_time   DOUBLE PRECISION NOT NULL,
        status         TEXT NOT NULL DEFAULT 'pending'
    )
"""

_QUEUE_ITEMS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_queue_items_proc_status "
    "ON queue_items (processor_name, status, enqueue_time)"
)


def _dsn() -> str:
    if _DATABASE_URL:
        return _DATABASE_URL
    return (
        f"host={_PG_HOST} port={_PG_PORT} dbname={_PG_DATABASE} "
        f"user={_PG_USER} password={_PG_PASSWORD}"
    )


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 20, _dsn())
    return _pool


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Public accessor for the shared connection pool — used by queue_system.py
    to store and dequeue jobs directly in Postgres."""
    return _get_pool()


def init_db() -> bool:
    """Create the log tables + queue_items table if they don't exist.

    Returns True if Postgres is reachable. Log handlers treat a failure here
    as "fall back to console only"; the job queue requires this to succeed.
    """
    global _disabled
    try:
        p = _get_pool()
        conn = p.getconn()
        try:
            with conn, conn.cursor() as cur:
                for ddl in _TABLE_DDL.values():
                    cur.execute(ddl)
                for ddl in _INDEX_DDL:
                    cur.execute(ddl)
                cur.execute(_QUEUE_ITEMS_DDL)
                cur.execute(_QUEUE_ITEMS_INDEX_DDL)
        finally:
            p.putconn(conn)
        _disabled = False
        return True
    except Exception as exc:
        _disabled = True
        _setup_logger.warning(
            f"Postgres unreachable, logging falls back to console only: {exc}"
        )
        return False


class PostgresLogHandler(logging.Handler):
    """Writes each log record as a row into a Postgres table.

    `extra_fields` names attributes pulled off the LogRecord (set via
    `logger.info(msg, extra={...})`) into their own columns. Never raises —
    a DB hiccup is swallowed so the processing pipeline keeps running.
    """

    def __init__(self, table: str, extra_fields: tuple[str, ...] = ()) -> None:
        super().__init__()
        if table not in _TABLE_DDL:
            raise ValueError(f"Unknown log table: {table}")
        self.table = table
        self.extra_fields = extra_fields

    def emit(self, record: logging.LogRecord) -> None:
        if _disabled:
            return
        try:
            columns = ["created_at", "level", "logger", "message", *self.extra_fields]
            values = [
                datetime.fromtimestamp(record.created, tz=timezone.utc),
                record.levelname,
                record.name,
                record.getMessage(),
                *(getattr(record, f, None) for f in self.extra_fields),
            ]
            placeholders = ", ".join(["%s"] * len(columns))
            sql = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({placeholders})"

            p = _get_pool()
            conn = p.getconn()
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(sql, values)
            finally:
                p.putconn(conn)
        except Exception:
            self.handleError(record)
