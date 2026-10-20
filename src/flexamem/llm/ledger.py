"""Phase-tagged JSONL cost ledger. Every model call (agent, write, link, merge, summarize, gist,
lookup, reader, judge) and every embedding batch appends one row. Context fields (run, method,
item, budget) come from a contextvar set by the runner around each work item."""
import contextvars
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("ledger_ctx", default={})
_LEDGER = None


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, **row):
        row = {"ts": round(time.time(), 3), **_ctx.get(), **row}
        line = json.dumps(row, default=str)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")


def set_ledger(path) -> Ledger:
    global _LEDGER
    _LEDGER = Ledger(path)
    return _LEDGER


def get_ledger() -> Ledger:
    if _LEDGER is None:
        raise RuntimeError("no ledger set: every model call must be recorded (call set_ledger first)")
    return _LEDGER


@contextmanager
def ledger_context(**fields):
    token = _ctx.set({**_ctx.get(), **fields})
    try:
        yield
    finally:
        _ctx.reset(token)


def current_context() -> dict:
    return dict(_ctx.get())


def pmap(fn, items, workers: int = 8) -> list:
    """Thread-pool map that carries the ledger context into worker threads; preserves order."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if len(items) <= 1 or workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        futs = [ex.submit(contextvars.copy_context().run, fn, x) for x in items]
        return [f.result() for f in futs]
