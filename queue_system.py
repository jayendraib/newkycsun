"""
Pluggable FIFO queue engine — disk-backed via Postgres (queue_items table).

Every job (KYC zip, summary URL, ...) is inserted into Postgres the moment
it is enqueued — no RAM list, no SQLite file. Each processor type has its
own dedicated worker thread that pulls the oldest pending job for that type
(first in, first out) and runs it. Jobs survive server restarts — any job
that was mid-processing when the server crashed is automatically reset to
'pending'.

To add a new processor:
    queue.register_processor("ocr", ocr_handler)

To add a new input source:
    queue.enqueue("ocr", {"file": "/path/to/file"})

queue_items table (Postgres, see db.py):
    item_id        — unique job ID
    processor_name — "kyc" or "summary"
    data           — JSON string of the job payload
    enqueue_time   — unix timestamp (float)
    status         — pending / processing / done / failed
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import db

_RESET = "\033[0m"

_AUTO_COLORS = [
    "\033[36m",  # cyan
    "\033[35m",  # magenta
    "\033[32m",  # green
    "\033[34m",  # blue
    "\033[31m",  # red
    "\033[37m",  # white
]
_QUEUE_COLOR = "\033[33m"  # yellow — queue manager logs


# ── Formatters ────────────────────────────────────────────────────────────────

def _console_fmt(label: str, color: str) -> logging.Formatter:
    class Fmt(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            lvl = record.levelname[:4]
            return f"{color}[{ts}][{label}][{lvl}]{_RESET} {record.getMessage()}"
    return Fmt()


class _ProcessorFilter(logging.Filter):
    """Injects the processor label onto every record, for the queue_logs table."""

    def __init__(self, processor: str) -> None:
        super().__init__()
        self.processor = processor

    def filter(self, record: logging.LogRecord) -> bool:
        record.processor = self.processor
        return True


def _build_logger(name: str, label: str, color: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_console_fmt(label, color))
    logger.addHandler(ch)

    # Persist queue logs to Postgres (queue_logs table) instead of a local file.
    db.init_db()
    pg = db.PostgresLogHandler(table="queue_logs", extra_fields=("processor",))
    pg.addFilter(_ProcessorFilter(label))
    logger.addHandler(pg)

    return logger


@contextmanager
def _connection():
    """Borrow a Postgres connection from the shared pool (db.py) and return it after."""
    pool = db.get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class QueueItem:
    item_id: str
    processor_name: str
    data: Any
    enqueue_time: float = field(default_factory=time.time)


@dataclass
class ProcessorEntry:
    handler: Callable
    logger: logging.Logger
    color: str


# ── QueueManager ──────────────────────────────────────────────────────────────

class QueueManager:
    """
    FIFO queue manager with Postgres persistence (see db.py: queue_items table).

    Jobs are written to Postgres the moment they are enqueued — no RAM list,
    no SQLite file. Each processor type gets its own dedicated worker thread
    that repeatedly pulls the oldest pending job for that type, marks it
    'processing', runs it, then marks it 'done' or 'failed'. On restart, any
    'processing' jobs are automatically reset to 'pending' so nothing is lost.
    """

    def __init__(self) -> None:
        self._processors: dict[str, ProcessorEntry] = {}
        self._lock = threading.Lock()
        self._running = False
        self._worker_threads: list[threading.Thread] = []
        self._color_idx = 0

        # ── Postgres-backed queue ────────────────────────────────────────────
        db.init_db()

        # Any job left as 'processing' from a previous crash → reset to 'pending'
        with _connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM queue_items WHERE status='processing'")
            crashed = cur.fetchone()[0]
            if crashed:
                cur.execute("UPDATE queue_items SET status='pending' WHERE status='processing'")
            conn.commit()

        self._mgr_logger = _build_logger(
            "queue.manager",
            "QUEUE",
            _QUEUE_COLOR,
        )

        if crashed:
            self._mgr_logger.warning(
                f"Recovered {crashed} job(s) that were mid-processing on last shutdown"
            )

    # ── Registration ──────────────────────────────────────────────────────────

    def register_processor(
        self,
        name: str,
        handler: Callable,
        color: str | None = None,
    ) -> None:
        """
        Register a processor.

        handler — sync or async callable(data: dict) -> Any
        """
        if color is None:
            color = _AUTO_COLORS[self._color_idx % len(_AUTO_COLORS)]
            self._color_idx += 1

        proc_logger = _build_logger(
            f"queue.proc.{name}",
            name.upper(),
            color,
        )
        entry = ProcessorEntry(
            handler=handler,
            logger=proc_logger,
            color=color,
        )
        with self._lock:
            self._processors[name] = entry

        # Show how many pending jobs are already queued in Postgres for this processor
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM queue_items WHERE processor_name=%s AND status='pending'",
                (name,),
            )
            pending = cur.fetchone()[0]

        proc_logger.info(f"Registered | pending_on_disk={pending}")
        self._mgr_logger.info(f"Registered '{name}'  pending_on_disk={pending}")

    # ── Enqueueing ────────────────────────────────────────────────────────────

    def enqueue(self, processor_name: str, data: Any) -> str:
        """Insert a job into Postgres. Returns item_id. Uses no RAM for the payload."""
        with self._lock:
            if processor_name not in self._processors:
                raise KeyError(f"No processor named '{processor_name}'")
        item_id = f"{processor_name}_{datetime.now().strftime('%H%M%S%f')}"

        with _connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO queue_items (item_id, processor_name, data, enqueue_time, status) "
                "VALUES (%s, %s, %s, %s, 'pending')",
                (item_id, processor_name, json.dumps(data), time.time()),
            )
            cur.execute(
                "SELECT COUNT(*) FROM queue_items WHERE processor_name=%s AND status='pending'",
                (processor_name,),
            )
            depth = cur.fetchone()[0]
            conn.commit()

        self._processors[processor_name].logger.info(
            f"Enqueued {item_id}  |  queue depth={depth}"
        )
        self._mgr_logger.info(
            f"[{processor_name.upper()}] {item_id} queued  depth={depth}"
        )
        return item_id

    # ── FIFO dequeue (atomic claim-oldest-pending) ────────────────────────────

    def _pop_item(self, processor_name: str) -> QueueItem | None:
        """Atomically claim the oldest pending job for this processor. None if empty."""
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE queue_items
                SET status = 'processing'
                WHERE item_id = (
                    SELECT item_id FROM queue_items
                    WHERE processor_name = %s AND status = 'pending'
                    ORDER BY enqueue_time ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING item_id, data, enqueue_time
                """,
                (processor_name,),
            )
            row = cur.fetchone()
            conn.commit()
            if row is None:
                return None
            item_id, data_json, enqueue_time = row

        return QueueItem(
            item_id=item_id,
            processor_name=processor_name,
            data=json.loads(data_json),
            enqueue_time=enqueue_time,
        )

    def _mark_done(self, item_id: str) -> None:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE queue_items SET status='done' WHERE item_id=%s", (item_id,))
            conn.commit()

    def _mark_failed(self, item_id: str) -> None:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE queue_items SET status='failed' WHERE item_id=%s", (item_id,))
            conn.commit()

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, dict]:
        result = {}
        with _connection() as conn, conn.cursor() as cur:
            for name in self._processors:
                cur.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE processor_name=%s AND status='pending'",
                    (name,),
                )
                depth = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE processor_name=%s AND status='failed'",
                    (name,),
                )
                failed = cur.fetchone()[0]
                result[name] = {"depth": depth, "failed": failed}
        return result

    # ── Item execution ────────────────────────────────────────────────────────

    def _run_item(self, item: QueueItem, job_num: int) -> None:
        entry = self._processors[item.processor_name]
        log = entry.logger

        wait = time.time() - item.enqueue_time
        sep = "─" * 50

        log.info(sep)
        log.info(f"START  {item.item_id}  [job #{job_num}]  waited {wait:.2f}s")
        self._mgr_logger.info(f"▶ [{item.processor_name.upper()}] {item.item_id}  job #{job_num}")

        t0 = time.monotonic()
        try:
            if inspect.iscoroutinefunction(entry.handler):
                result = asyncio.run(entry.handler(item.data))
            else:
                result = entry.handler(item.data)

            elapsed = time.monotonic() - t0
            preview = str(result)[:150] if result is not None else "None"
            log.info(f"DONE   {item.item_id}  |  {elapsed:.2f}s  |  {preview}")
            self._mgr_logger.info(
                f"✓ [{item.processor_name.upper()}] {item.item_id}  {elapsed:.2f}s"
            )
            self._mark_done(item.item_id)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            log.exception(
                f"FAILED {item.item_id}  |  {elapsed:.2f}s  |  {exc}"
            )
            self._mgr_logger.error(
                f"✗ [{item.processor_name.upper()}] {item.item_id}  FAILED: {exc}"
            )
            self._mark_failed(item.item_id)
        finally:
            log.info(sep)

    # ── Worker (per-processor, concurrent) ────────────────────────────────────

    def _processor_worker(self, name: str) -> None:
        """Dedicated worker thread for a single processor type — runs concurrently with others."""
        entry = self._processors[name]
        idle_logged = False
        job_num = 0

        self._mgr_logger.info(f"Worker started for '{name}'")
        while self._running:
            item = self._pop_item(name)
            if item is None:
                if not idle_logged:
                    entry.logger.info("Idle — queue empty, waiting for jobs...")
                    idle_logged = True
                time.sleep(0.05)
                continue

            idle_logged = False
            job_num += 1
            self._run_item(item, job_num)

        entry.logger.info(f"Worker stopped after {job_num} jobs")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_threads = []
        for name in list(self._processors.keys()):
            t = threading.Thread(
                target=self._processor_worker,
                args=(name,),
                daemon=True,
                name=f"queue-worker-{name}",
            )
            self._worker_threads.append(t)
            t.start()
        self._mgr_logger.info(
            f"QueueManager started — {len(self._worker_threads)} parallel workers "
            f"({', '.join(self._processors.keys())})"
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._mgr_logger.info("Stopping all workers...")
        self._running = False
        for t in self._worker_threads:
            t.join(timeout=timeout)
        self._mgr_logger.info("QueueManager stopped")
