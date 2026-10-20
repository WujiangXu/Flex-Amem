"""LongMemEval-S (cleaned) loading with label stripping.

The haystack given to any method contains only session dates and turns (role, content). Session ids
are replaced by opaque ids s0..sN in date order (the originals mark evidence sessions with an
"answer_" prefix), and turn-level `has_answer` flags are removed. Gold fields stay in a separate
record used only by the judge and the oracle reference.
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "artifacts/data/longmemeval/longmemeval_s_cleaned.json"
DEV_SPLIT = ROOT / "configs/splits/lme_dev60.txt"


@lru_cache(maxsize=1)
def _raw() -> list[dict]:
    return json.loads(DATA.read_text())


def load(ids: list[str] | None = None) -> list[dict]:
    """Returns [{"qid", "qtype", "question", "question_date", "sessions", "gold"}]."""
    rows = _raw()
    if ids is not None:
        want = set(ids)
        rows = [r for r in rows if r["question_id"] in want]
    return [_prepare(r) for r in rows]


def _prepare(r: dict) -> dict:
    idx = sorted(range(len(r["haystack_sessions"])), key=lambda i: r["haystack_dates"][i])  # stable
    sessions, evidence = [], []
    for rank, i in enumerate(idx):
        sid = f"s{rank}"
        turns = [{"role": t["role"], "content": t["content"]} for t in r["haystack_sessions"][i]]
        sessions.append({"sid": sid, "date": r["haystack_dates"][i], "turns": turns})
        if r["haystack_session_ids"][i] in set(r["answer_session_ids"]):
            evidence.append(sid)
    for s in sessions:
        assert set(s) == {"sid", "date", "turns"} and not s["sid"].startswith("answer_")
        for t in s["turns"]:
            assert set(t) == {"role", "content"}
    return {
        "qid": r["question_id"],
        "qtype": r["question_type"],
        "question": r["question"],
        "question_date": r["question_date"],
        "sessions": sessions,
        "gold": {"answer": str(r["answer"]), "evidence_sids": evidence,
                 "abstention": r["question_id"].endswith("_abs")},
    }


def dev_ids() -> list[str]:
    return [x.strip() for x in DEV_SPLIT.read_text().splitlines() if x.strip()]
