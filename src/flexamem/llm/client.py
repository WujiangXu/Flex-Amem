"""The only path to LLMs (any OpenAI-compatible endpoint; configs/models/default.yaml). Records every call in the ledger,
retries transient errors with backoff, and raises on persistent failure (no silent fallbacks).

Calls may pass cache=True: identical prompts are answered from an on-disk cache, and the ledger row
keeps the original usage (summed over parse retries) with cache_hit=True so costs can still be
reported as if uncached. `salt` separates otherwise identical calls (e.g. reader seeds). Within one process,
concurrent calls with the same key send one request (single flight); separate processes can still race.
An empty reply is retried as unusable unless the caller passes allow_empty=True (a filter whose prompt allows
"nothing relevant"); an empty reply that stopped normally then counts as an answer and is cached like any other.

forbid_text(s): inside this context any prompt containing s raises LeakError (LongMemEval builds run
under the question text, so a question-aware build fails loudly).
"""
import contextlib
import contextvars
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from pathlib import Path

import openai
import yaml

from flexamem.llm.ledger import get_ledger
from flexamem.llm.tokens import billed_usage

ROOT = Path(__file__).resolve().parents[3]
TRANSIENT = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
# A prompt rejected with HTTP 400 (e.g. a provider's content-policy flag, which can be stochastic) is sent at most
# REJECT_SENDS times, REJECT_WAIT_S apart, and then fails for good.
REJECT_SENDS, REJECT_WAIT_S = 3, 30


class LLMError(RuntimeError):
    pass


class LLMTransientError(LLMError):
    """The API kept failing (rate limit, timeout, 5xx): an infrastructure error, retried on resume."""


class LeakError(AssertionError):
    pass


_FORBID: contextvars.ContextVar[tuple] = contextvars.ContextVar("forbid_text", default=())


@contextlib.contextmanager
def forbid_text(text: str | tuple | list):
    """Within the block, any model call whose prompt contains the text (or any of several texts; LoCoMo guards a
    conversation's build with all its questions) raises LeakError. Empty texts are ignored."""
    texts = (text,) if isinstance(text, str) else tuple(text)
    token = _FORBID.set(tuple(t.strip() for t in texts if t and t.strip()))
    try:
        yield
    finally:
        _FORBID.reset(token)


def load_model_config(path=None) -> dict:
    cfg = yaml.safe_load(Path(path or os.environ.get("FLEXAMEM_MODELS") or ROOT / "configs/models/default.yaml").read_text())
    cfg["base_url"] = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")  # None: the OpenAI API
    if os.environ.get("FLEXAMEM_ROLES"):  # second-backbone runs: a pinned file in configs/models/ replaces some roles
        cfg["roles"].update(yaml.safe_load((ROOT / os.environ["FLEXAMEM_ROLES"]).read_text())["roles"])
    if os.environ.get("FLEXAMEM_MAX_CONCURRENCY"):  # per-process cap on concurrent calls, raised for large runs
        cfg["max_concurrency"] = int(os.environ["FLEXAMEM_MAX_CONCURRENCY"])
    return cfg


class _Cache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, k):
        with self._lock:
            r = self._db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self._db.commit()


class LLM:
    def __init__(self, cfg: dict | None = None, cache_path=None):
        self.cfg = cfg or load_model_config()
        key = next((os.environ[k] for k in self.cfg["api_key_env"] if os.environ.get(k)), None)
        if not key:
            raise LLMError(f"none of {self.cfg['api_key_env']} is set")
        self.api_key = key
        self.client = openai.OpenAI(api_key=key, base_url=self.cfg["base_url"], max_retries=0, timeout=600)
        self._sem = threading.BoundedSemaphore(self.cfg.get("max_concurrency", 32))
        # FLEXAMEM_LLM_CACHE: a run's own cache, so cost accounting never meets another run's answers
        self._cache = _Cache(cache_path or os.environ.get("FLEXAMEM_LLM_CACHE") or ROOT / "artifacts/cache/llm_cache.sqlite")
        self.cache_path = str(cache_path or os.environ.get("FLEXAMEM_LLM_CACHE") or ROOT / "artifacts/cache/llm_cache.sqlite")
        self._flights: dict[str, tuple[threading.Lock, int]] = {}  # cache key -> (lock, waiting callers)
        self._flights_guard = threading.Lock()

    def role(self, role: str) -> dict:
        return self.cfg["roles"][role]

    def _request(self, spec, messages, schema, max_tokens=None, tools=None, tool_choice=None, response_format=None,
                 reasoning_effort=None):
        kw = {"model": spec["model"], "messages": messages}
        if tools:
            kw["tools"] = tools
            if tool_choice:
                kw["tool_choice"] = tool_choice
        if max_tokens or spec.get("max_tokens"):
            kw["max_completion_tokens"] = max_tokens or spec["max_tokens"]
        if reasoning_effort or spec.get("reasoning_effort"):
            kw["reasoning_effort"] = reasoning_effort or spec["reasoning_effort"]
        if "temperature" in spec:
            kw["temperature"] = spec["temperature"]
        if schema is not None:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "out", "strict": True, "schema": schema}}
        if response_format is not None:
            kw["response_format"] = response_format
        attempt, rejected = 0, 0
        while True:
            try:
                with self._sem:
                    t0 = time.time()
                    r = self.client.chat.completions.create(**kw)
                return r, time.time() - t0
            except openai.BadRequestError:
                rejected += 1
                if rejected == REJECT_SENDS:
                    raise
                time.sleep(REJECT_WAIT_S)
            except TRANSIENT as e:
                if attempt == 13:
                    raise LLMTransientError(f"{spec['model']}: persistent transient failure: {e!r}") from e
                time.sleep(min(90.0, 2 ** min(attempt, 6)) * (0.5 + random.random()))
                attempt += 1

    def call(self, role: str, messages, *, phase: str, schema: dict | None = None, cache: bool = False,
             salt=None, tools: list | None = None, tool_choice: str | None = None, allow_empty: bool = False,
             **meta):
        """Returns str (no schema) or dict (schema). messages may be a plain prompt string. With tools, one
        tool-enabled turn: {"content": str, "tool_calls": [{"id", "type", "function": {"name", "arguments"}}]}."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        hit = next((f for f in _FORBID.get() for m in messages if f in (m.get("content") or "")), None)
        if hit is not None:
            raise LeakError(f"[{phase}] prompt contains forbidden text {hit[:60]!r}")
        spec = self.role(role)
        tk = (tools, tool_choice) if tools else None
        if not cache:
            return self._call(spec, messages, schema, phase, None, meta, tk, allow_empty)
        ck = _cache_key(spec, messages, schema, salt, tk)
        # single flight: a concurrent call with the same key waits for the first one and gets its answer. Without it
        # both missed, both asked, and each kept its own reply (the judges labelled identical answers differently).
        with self._flight(ck):
            hit = self._cache.get(ck)
            if hit is not None:
                get_ledger().record(phase=phase, model=spec["model"], cache_hit=True, **hit["usage"], **meta,
                                    **({"usage_v": hit["usage_v"]} if hit.get("usage_v") else {}))
                return hit["out"]
            return self._call(spec, messages, schema, phase, ck, meta, tk, allow_empty)

    def cached(self, role: str, messages, *, schema: dict | None = None, salt=None):
        """The cached answer of call(..., cache=True) with these arguments, or None. No request, no ledger row."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        hit = self._cache.get(_cache_key(self.role(role), messages, schema, salt))
        return None if hit is None else hit["out"]

    @contextlib.contextmanager
    def _flight(self, ck: str):
        with self._flights_guard:
            lock, n = self._flights.get(ck, (threading.Lock(), 0))
            self._flights[ck] = (lock, n + 1)
        try:
            with lock:
                yield
        finally:
            with self._flights_guard:
                n = self._flights[ck][1] - 1
                if n:
                    self._flights[ck] = (lock, n)
                else:
                    del self._flights[ck]

    def _call(self, spec, messages, schema, phase, ck, meta, tk=None, allow_empty=False):
        last_err, total, max_tokens = None, {}, spec.get("max_tokens")
        for _ in range(4):  # parse/truncation retries (each attempt is billed and logged)
            r, dt = self._request(spec, messages, schema, max_tokens, *(tk or ()))
            u = r.usage
            pt, ct = billed_usage(u.model_dump()) if u is not None else (0, 0)
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0,
            }
            total = {k: total.get(k, 0) + v for k, v in usage.items()}
            text = (r.choices[0].message.content or "").strip()
            truncated = r.choices[0].finish_reason == "length"
            ok = (bool(text) or allow_empty and r.choices[0].finish_reason == "stop") and not truncated
            out = text
            if tk:
                calls = [{"id": c.id, "type": "function",
                          "function": {"name": c.function.name, "arguments": c.function.arguments}}
                         for c in r.choices[0].message.tool_calls or []]
                ok, out = (bool(text) or bool(calls)) and not truncated, {"content": text, "tool_calls": calls}
            if ok and schema is not None:
                try:
                    out = json.loads(_strip_fence(text))
                except json.JSONDecodeError as e:
                    ok, last_err = False, e
            reported = getattr(u, "prompt_tokens", 0) or 0
            get_ledger().record(phase=phase, model=spec["model"], latency_s=round(dt, 3), ok=ok, truncated=truncated,
                                cache_hit=False, usage_v=2, **usage, **meta,
                                **({"prompt_tokens_reported": reported} if reported != usage["prompt_tokens"] else {}))
            if ok:
                if ck:
                    self._cache.put(ck, {"out": out, "usage": total, "usage_v": 2})
                return out
            if truncated and max_tokens:  # reasoning ate the output limit: retry with twice the room
                max_tokens *= 2
                last_err = ValueError("output truncated at max_tokens")
            last_err = last_err or ValueError("empty completion")
        raise LLMError(f"{spec['model']} [{phase}]: unusable output after 4 attempts: {last_err!r}")


def _cache_key(spec, messages, schema, salt, tk=None) -> str:
    key = [spec, messages, schema] + ([salt] if salt else []) + ([list(tk)] if tk else [])
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _strip_fence(t: str) -> str:
    t = t.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


_LLM = None
_LLM_LOCK = threading.Lock()


def get_llm() -> LLM:
    global _LLM
    with _LLM_LOCK:
        if _LLM is None:
            _LLM = LLM()
        return _LLM
