"""
Pluggable weighted round-robin queue engine.

Solves starvation: each processor gets `slots` items per cycle.
Even if KYC has 10,000 items queued, Summary still runs every cycle.

Cycle example — KYC(slots=3), SUMMARY(slots=1):
    [KYC×3  →  SUMMARY×1  →  KYC×3  →  SUMMARY×1  → ...]

To add a new processor:
    queue.register_processor("ocr", ocr_handler, priority=3, slots=2)

To add a new input source:
    queue.enqueue("ocr", {"file": "/path/to/file"})

To change balance at runtime:
    queue.set_slots("kyc", 5)       # KYC gets 5 items per cycle
    queue.set_priority("kyc", 2)    # push KYC later in the cycle
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import threading
import time
from collections import deque
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
    enqueue_time: float = field(default_factory=time.monotonic)


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
    Weighted round-robin queue manager.

    Each cycle processes `slots` items from each registered processor
    in `priority` order. If a processor queue is empty during its turn,
    it is skipped. This guarantees no processor starves as long as it
    has items, regardless of how busy other processors are.

    Runtime controls:
        set_priority(name, n)  — reorder processors in the cycle
        set_slots(name, n)     — change throughput share per cycle
    """

    def __init__(self, log_dir: str = ".") -> None:
        self._log_dir = log_dir
        self._processors: dict[str, ProcessorEntry] = {}
        self._queues: dict[str, deque[QueueItem]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._color_idx = 0
        self._mgr_logger = _build_logger(
            "queue.manager",
            "QUEUE",
            _QUEUE_COLOR,
            f"{log_dir}/queue_manager.log",
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
                    e.g. slots=3 means process 3 items, then yield to next processor
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
            self._queues[name] = deque()

        proc_logger.info(f"Registered | priority={priority}  slots={slots}")
        self._mgr_logger.info(
            f"Registered '{name}'  priority={priority}  slots={slots}"
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
        """Push an item onto the named processor's queue. Returns item_id."""
        with self._lock:
            if processor_name not in self._processors:
                raise KeyError(f"No processor named '{processor_name}'")
            item_id = f"{processor_name}_{datetime.now().strftime('%H%M%S%f')}"
            item = QueueItem(
                item_id=item_id,
                processor_name=processor_name,
                data=data,
            )
            self._queues[processor_name].append(item)
            depth = len(self._queues[processor_name])

        self._processors[processor_name].logger.info(
            f"Enqueued {item_id}  |  queue depth={depth}"
        )
        self._mgr_logger.info(
            f"[{processor_name.upper()}] {item_id} queued  depth={depth}"
        )
        return item_id

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, dict]:
        with self._lock:
            return {
                name: {
                    "priority": self._processors[name].priority,
                    "slots": self._processors[name].slots,
                    "depth": len(self._queues[name]),
                }
                for name in self._processors
            }

    def _log_cycle_status(self, label: str) -> None:
        with self._lock:
            parts = "  |  ".join(
                f"{n}(p={self._processors[n].priority}, s={self._processors[n].slots}, q={len(self._queues[n])})"
                for n in sorted(self._processors, key=lambda x: self._processors[x].priority)
            )
        self._mgr_logger.info(f"{label}  →  {parts or 'no processors'}")

    # ── Item execution ────────────────────────────────────────────────────────

    def _run_item(self, item: QueueItem, slot_num: int, total_slots: int) -> None:
        entry = self._processors[item.processor_name]
        log = entry.logger

        wait = time.monotonic() - item.enqueue_time
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
        except Exception as exc:
            elapsed = time.monotonic() - t0
            log.exception(
                f"FAILED {item.item_id}  |  {elapsed:.2f}s  |  {exc}"
            )
            self._mgr_logger.error(
                f"✗ [{item.processor_name.upper()}] {item.item_id}  FAILED: {exc}"
            )
        finally:
            log.info(sep)

    # ── Worker (weighted round-robin) ─────────────────────────────────────────

    def _worker(self) -> None:
        self._mgr_logger.info("Worker started  (weighted round-robin)")
        idle_logged = False
        cycle = 0

        while self._running:
            # Snapshot processor order for this cycle
            with self._lock:
                ordered_names = sorted(
                    self._processors,
                    key=lambda n: self._processors[n].priority,
                )

            cycle_processed = 0

            for name in ordered_names:
                with self._lock:
                    entry = self._processors[name]
                    slots = entry.slots  # snapshot slots for this turn

                slot_count = 0
                while slot_count < slots:
                    with self._lock:
                        if not self._queues[name]:
                            break
                        item = self._queues[name].popleft()

                    slot_count += 1
                    cycle_processed += 1
                    self._run_item(item, slot_count, slots)

                if slot_count > 0:
                    with self._lock:
                        remaining = len(self._queues[name])
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
        self._mgr_logger.info("QueueManager stopped")
