"""
Pluggable weighted round-robin queue engine — disk-backed via SQLite.

All jobs are persisted to queue.db on disk. RAM usage stays flat no matter
how many jobs are queued. Jobs survive server restarts — any job that was
mid-processing when the server crashed is automatically reset to 'pending'.

Cycle example — KYC(slots=3), SUMMARY(slots=1):
    [KYC×3  →  SUMMARY×1  →  KYC×3  →  SUMMARY×1  → ...]

To add a new processor:
    queue.register_processor("ocr", ocr_handler, priority=3, slots=2)

To add a new input source:
    queue.enqueue("ocr", {"file": "/path/to/file"})

To change balance at runtime:
    queue.set_slots("kyc", 5)       # KYC gets 5 items per cycle
    queue.set_priority("kyc", 2)    # push KYC later in the cycle

queue.db table: queue_items
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
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

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


def _file_fmt(label: str) -> logging.Formatter:
    class Fmt(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            return f"[{ts}][{label}][{record.levelname}] {record.getMessage()}"
    return Fmt()


def _build_logger(name: str, label: str, color: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_console_fmt(label, color))
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(_file_fmt(label))
        logger.addHandler(fh)

    return logger


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
    priority: int       # order within a cycle — lower runs first
    slots: int          # items to process per cycle — controls share of throughput
    logger: logging.Logger
    color: str


# ── QueueManager ──────────────────────────────────────────────────────────────

class QueueManager:
    """
    Weighted round-robin queue manager with SQLite disk persistence.

    Jobs are written to queue.db the moment they are enqueued — no RAM list.
    The worker reads one job at a time from disk, marks it 'processing', runs
    it, then marks it 'done' or 'failed'. On restart, any 'processing' jobs
    are automatically reset to 'pending' so nothing is lost.

    Runtime controls:
        set_priority(name, n)  — reorder processors in the cycle
        set_slots(name, n)     — change throughput share per cycle
    """

    def __init__(self, log_dir: str = ".") -> None:
        self._log_dir = log_dir
        self._processors: dict[str, ProcessorEntry] = {}
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._color_idx = 0

        # ── SQLite disk queue ─────────────────────────────────────────────────
        self._db_path = f"{log_dir}/queue.db"
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads/writes
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS queue_items (
                item_id        TEXT PRIMARY KEY,
                processor_name TEXT NOT NULL,
                data           TEXT NOT NULL,
                enqueue_time   REAL NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending'
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_proc_status
            ON queue_items (processor_name, status, enqueue_time)
        """)
        self._db.commit()

        # Any job left as 'processing' from a previous crash → reset to 'pending'
        crashed = self._db.execute(
            "SELECT COUNT(*) FROM queue_items WHERE status='processing'"
        ).fetchone()[0]
        if crashed:
            self._db.execute("UPDATE queue_items SET status='pending' WHERE status='processing'")
            self._db.commit()

        self._mgr_logger = _build_logger(
            "queue.manager",
            "QUEUE",
            _QUEUE_COLOR,
            f"{log_dir}/queue_manager.log",
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
        priority: int = 10,
        slots: int = 1,
        color: str | None = None,
    ) -> None:
        """
        Register a processor.

        priority  — order in each cycle (lower = runs before higher)
        slots     — how many items this processor handles per cycle
        handler   — sync or async callable(data: dict) -> Any
        """
        if color is None:
            color = _AUTO_COLORS[self._color_idx % len(_AUTO_COLORS)]
            self._color_idx += 1

        proc_logger = _build_logger(
            f"queue.proc.{name}",
            name.upper(),
            color,
            f"{self._log_dir}/{name}_queue.log",
        )
        entry = ProcessorEntry(
            handler=handler,
            priority=priority,
            slots=slots,
            logger=proc_logger,
            color=color,
        )
        with self._lock:
            self._processors[name] = entry

        # Show how many pending jobs are already on disk for this processor
        pending = self._db.execute(
            "SELECT COUNT(*) FROM queue_items WHERE processor_name=? AND status='pending'",
            (name,),
        ).fetchone()[0]

        proc_logger.info(f"Registered | priority={priority}  slots={slots}  pending_on_disk={pending}")
        self._mgr_logger.info(
            f"Registered '{name}'  priority={priority}  slots={slots}  pending_on_disk={pending}"
        )

    # ── Runtime controls ──────────────────────────────────────────────────────

    def set_priority(self, name: str, priority: int) -> None:
        """Change a processor's position within each cycle. Takes effect next cycle."""
        with self._lock:
            entry = self._processors.get(name)
            if entry is None:
                raise KeyError(f"No processor named '{name}'")
            old = entry.priority
            entry.priority = priority
        entry.logger.info(f"Priority {old} → {priority}")
        self._mgr_logger.info(f"'{name}' priority {old} → {priority}")

    def set_slots(self, name: str, slots: int) -> None:
        """
        Change how many items this processor handles per cycle.
        Takes effect at the start of the next cycle.

        Increase to give more throughput. Decrease to yield more to others.
        Example: kyc=3, summary=1  →  process 3 KYC then 1 Summary, repeat.
        """
        if slots < 1:
            raise ValueError("slots must be >= 1")
        with self._lock:
            entry = self._processors.get(name)
            if entry is None:
                raise KeyError(f"No processor named '{name}'")
            old = entry.slots
            entry.slots = slots
        entry.logger.info(f"Slots {old} → {slots}")
        self._mgr_logger.info(f"'{name}' slots {old} → {slots}")

    # ── Enqueueing ────────────────────────────────────────────────────────────

    def enqueue(self, processor_name: str, data: Any) -> str:
        """Write a job to disk queue. Returns item_id. Uses no RAM for the payload."""
        with self._lock:
            if processor_name not in self._processors:
                raise KeyError(f"No processor named '{processor_name}'")
            item_id = f"{processor_name}_{datetime.now().strftime('%H%M%S%f')}"
            self._db.execute(
                "INSERT INTO queue_items (item_id, processor_name, data, enqueue_time, status) "
                "VALUES (?, ?, ?, ?, 'pending')",
                (item_id, processor_name, json.dumps(data), time.time()),
            )
            self._db.commit()
            depth = self._db.execute(
                "SELECT COUNT(*) FROM queue_items WHERE processor_name=? AND status='pending'",
                (processor_name,),
            ).fetchone()[0]

        self._processors[processor_name].logger.info(
            f"Enqueued {item_id}  |  queue depth={depth}"
        )
        self._mgr_logger.info(
            f"[{processor_name.upper()}] {item_id} queued  depth={depth}"
        )
        return item_id

    # ── Disk dequeue (atomic select + mark processing) ────────────────────────

    def _pop_item(self, processor_name: str) -> QueueItem | None:
        """Take the oldest pending job from disk. Returns None if queue is empty."""
        with self._lock:
            row = self._db.execute(
                "SELECT item_id, data, enqueue_time FROM queue_items "
                "WHERE processor_name=? AND status='pending' "
                "ORDER BY enqueue_time ASC LIMIT 1",
                (processor_name,),
            ).fetchone()
            if row is None:
                return None
            item_id, data_json, enqueue_time = row
            self._db.execute(
                "UPDATE queue_items SET status='processing' WHERE item_id=?",
                (item_id,),
            )
            self._db.commit()
        return QueueItem(
            item_id=item_id,
            processor_name=processor_name,
            data=json.loads(data_json),
            enqueue_time=enqueue_time,
        )

    def _mark_done(self, item_id: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE queue_items SET status='done' WHERE item_id=?", (item_id,)
            )
            self._db.commit()

    def _mark_failed(self, item_id: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE queue_items SET status='failed' WHERE item_id=?", (item_id,)
            )
            self._db.commit()

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, dict]:
        with self._lock:
            result = {}
            for name in self._processors:
                row = self._db.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE processor_name=? AND status='pending'",
                    (name,),
                ).fetchone()
                failed = self._db.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE processor_name=? AND status='failed'",
                    (name,),
                ).fetchone()[0]
                result[name] = {
                    "priority": self._processors[name].priority,
                    "slots": self._processors[name].slots,
                    "depth": row[0],
                    "failed": failed,
                }
            return result

    def _log_cycle_status(self, label: str) -> None:
        with self._lock:
            parts = "  |  ".join(
                f"{n}(p={self._processors[n].priority}, s={self._processors[n].slots}, "
                f"q={self._db.execute('SELECT COUNT(*) FROM queue_items WHERE processor_name=? AND status=?', (n, 'pending')).fetchone()[0]})"
                for n in sorted(self._processors, key=lambda x: self._processors[x].priority)
            )
        self._mgr_logger.info(f"{label}  →  {parts or 'no processors'}")

    # ── Item execution ────────────────────────────────────────────────────────

    def _run_item(self, item: QueueItem, slot_num: int, total_slots: int) -> None:
        entry = self._processors[item.processor_name]
        log = entry.logger

        wait = time.time() - item.enqueue_time
        sep = "─" * 50

        log.info(sep)
        log.info(
            f"START  {item.item_id}  "
            f"[slot {slot_num}/{total_slots}]  "
            f"waited {wait:.2f}s"
        )
        self._mgr_logger.info(
            f"▶ [{item.processor_name.upper()}] {item.item_id}  "
            f"slot {slot_num}/{total_slots}"
        )

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

    # ── Worker (weighted round-robin) ─────────────────────────────────────────

    def _worker(self) -> None:
        self._mgr_logger.info("Worker started  (weighted round-robin, disk-backed)")
        idle_logged = False
        cycle = 0

        while self._running:
            with self._lock:
                ordered_names = sorted(
                    self._processors,
                    key=lambda n: self._processors[n].priority,
                )

            cycle_processed = 0

            for name in ordered_names:
                with self._lock:
                    slots = self._processors[name].slots

                slot_count = 0
                while slot_count < slots:
                    item = self._pop_item(name)
                    if item is None:
                        break

                    slot_count += 1
                    cycle_processed += 1
                    self._run_item(item, slot_count, slots)

                if slot_count > 0:
                    with self._lock:
                        remaining = self._db.execute(
                            "SELECT COUNT(*) FROM queue_items WHERE processor_name=? AND status='pending'",
                            (name,),
                        ).fetchone()[0]
                    self._processors[name].logger.info(
                        f"Cycle {cycle}: processed {slot_count}/{slots} slots  "
                        f"|  {remaining} remaining in queue"
                    )

            if cycle_processed == 0:
                if not idle_logged:
                    self._mgr_logger.info("Idle — all queues empty, waiting...")
                    idle_logged = True
                time.sleep(0.05)
                continue

            idle_logged = False
            cycle += 1
            self._log_cycle_status(f"Cycle {cycle} done")

        self._mgr_logger.info("Worker stopped")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="queue-worker"
        )
        self._worker_thread.start()
        self._mgr_logger.info("QueueManager started")

    def stop(self, timeout: float = 10.0) -> None:
        self._mgr_logger.info("Stopping...")
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        self._db.close()
        self._mgr_logger.info("QueueManager stopped")
