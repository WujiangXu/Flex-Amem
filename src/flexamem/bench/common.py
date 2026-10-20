"""Run bookkeeping shared by the benchmark runners: manifest, resumable JSONL results."""
import fcntl
import importlib.metadata
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from flexamem.llm.client import ROOT, load_model_config


def git_state(*paths) -> dict:
    """HEAD and whether the tree differs from it, untracked files included; with paths, only under those paths."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = run("rev-parse", "HEAD")
    return {"sha": sha or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=normal", "--", *paths)) if sha else None}


# settings that change how a run executes but are not command-line arguments (unset -> None)
RUN_ENV = ("FLEXAMEM_MAX_CONCURRENCY", "FLEXAMEM_EMBED_THREADS", "DOCKER_HOST", "FLEXAMEM_ROLES")


def write_manifest(run_dir: Path, args: dict, allow_dirty: bool, **extra) -> dict:
    """Refuses an uncommitted tree unless allow_dirty. `extra` holds runner settings resolved from RUN_ENV (e.g. the
    SWE exec backend with its default applied)."""
    g = git_state()
    if not allow_dirty and (g["sha"] is None or g["dirty"]):
        raise SystemExit(f"refusing to run on an uncommitted tree ({g}); commit first or pass --allow-dirty")
    m = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "git": g, "args": args, "models": load_model_config(),
         "env": {k: os.environ.get(k) for k in RUN_ENV}, **extra,
         "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"manifest_{int(time.time())}_{os.getpid()}.json").write_text(json.dumps(m, indent=2, default=str))
    return m


def read_rows(path: Path, key_fields: tuple) -> list[dict]:
    """Read-only view of a results file (last row per key); ignores a partial last line."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[tuple(r[k] for k in key_fields)] = r
    return list(rows.values())


class Results:
    """Append-only JSONL keyed by a tuple of fields; lets a crashed run resume. Every row carries the
    code version (git_sha); resuming with rows from another version is refused unless allow_mixed.
    A key may appear more than once (e.g. after re-grading); the last row wins."""

    def __init__(self, path: Path, key_fields: tuple, allow_mixed: bool = False):
        self.path, self.key_fields = Path(path), key_fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.git_sha = git_state()["sha"]
        self._rows: dict[tuple, dict] = {}
        if self.path.exists():
            text = self.path.read_text()
            if text and not text.endswith("\n"):  # a crash mid-write left a partial last line
                text = text[:text.rfind("\n") + 1]
                self.path.write_text(text)
            for line in text.splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._rows[tuple(r[k] for k in key_fields)] = r
        shas = {r.get("git_sha") for r in self._rows.values()} - {self.git_sha}
        if shas and not allow_mixed:
            raise SystemExit(f"{self.path} holds rows from other code versions {sorted(map(str, shas))}; "
                             f"current {self.git_sha}. Use a new run id (or pass --allow-mixed).")

    def require(self, field: str, value, missing=None):
        """Refuse to resume over rows that recorded another value of field (e.g. the reader model of a second-backbone
        run, which an unset FLEXAMEM_ROLES would silently switch). Rows without the field are older: they count as
        `missing` when given (the value they ran under), else they pass."""
        other = {r.get(field, missing) for r in self._rows.values()} - {value, None}
        if other:
            raise SystemExit(f"{self.path} holds rows with {field} {sorted(map(str, other))}; current {value}")

    @property
    def done(self) -> set:
        return set(self._rows)

    def has(self, **key) -> bool:
        return tuple(key[k] for k in self.key_fields) in self._rows

    def rows(self) -> list[dict]:
        return list(self._rows.values())

    def add(self, row: dict):
        row = {**row, "git_sha": row.get("git_sha", self.git_sha)}
        with self._lock:
            with self.path.open("a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # sharded runs: several processes append to one file
                f.write(json.dumps(row, default=str) + "\n")
            self._rows[tuple(row[k] for k in self.key_fields)] = row
