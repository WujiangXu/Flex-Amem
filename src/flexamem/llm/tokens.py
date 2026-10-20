"""Token accounting shared by every method: tiktoken o200k_base over all message fields + tool schema.

tiktoken downloads the o200k_base BPE file on first use, or reads it from TIKTOKEN_CACHE_DIR; the paper's runs used
a file matching tiktoken's pinned sha256.
"""
import json
import os
from functools import lru_cache

os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.expanduser("~/.cache/tiktoken"))
import tiktoken  # noqa: E402

_ENC = tiktoken.get_encoding("o200k_base")
MSG_OVERHEAD = 4  # role/separator tokens per chat message
REPLY_PRIMING = 3


@lru_cache(maxsize=200_000)
def _count_cached(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def count(text: str) -> int:
    """Only short strings (blocks, messages) are memoised; whole rendered histories are not kept."""
    if not text:
        return 0
    return _count_cached(text) if len(text) <= 20_000 else len(_ENC.encode(text, disallowed_special=()))


def billed_usage(usage: dict) -> tuple[int, int]:
    """(prompt, completion) tokens a usage report bills, whatever the provider's convention (journal 2026-10-06):
    GPT models count reasoning inside completion_tokens and total = prompt + completion; Gemini reports reasoning
    outside completion_tokens (total = prompt + completion + reasoning), so it is added to the completion; Claude models
    behind some OpenAI-compatible gateways report only the uncached tail of an automatically cached prompt as prompt_tokens (10
    for a 5k-token prompt) while total_tokens holds the whole bill, so the rest of the gap is added to the prompt."""
    pt, ct = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    gap = (usage.get("total_tokens") or 0) - pt - ct
    if rt and gap >= rt:
        ct, gap = ct + rt, gap - rt
    return pt + max(gap, 0), ct


def full_prompt_tokens(usage: dict) -> int | None:
    """The billed prompt tokens of a usage report (billed_usage), None without a report."""
    return billed_usage(usage)[0] if usage and usage.get("prompt_tokens") is not None else None


def truncate(text: str, n: int) -> str:
    ids = _ENC.encode(text, disallowed_special=())
    return text if len(ids) <= n else _ENC.decode(ids[:max(n, 0)])


def chunks(text: str, n: int) -> list[str]:
    """Consecutive pieces of about n tokens each."""
    ids = _ENC.encode(text, disallowed_special=())
    return [_ENC.decode(ids[i:i + n]) for i in range(0, len(ids), n)]


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def count_message(m: dict) -> int:
    n = MSG_OVERHEAD + count(_content_text(m.get("content")))
    if isinstance(m.get("reasoning_content"), str):  # re-sent by mini-swe-agent if the API returned it
        n += count(m["reasoning_content"])
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        n += count(fn.get("name", "")) + count(fn.get("arguments", "")) + 8
    if m.get("tool_call_id"):
        n += count(m["tool_call_id"])
    return n


def count_messages(messages: list[dict], tools: list | None = None) -> int:
    n = REPLY_PRIMING + sum(count_message(m) for m in messages)
    if tools:
        n += count(json.dumps(tools, sort_keys=True))
    return n
