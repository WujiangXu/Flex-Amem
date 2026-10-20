"""LoCoMo answer checking.

J: the judge template of bench/locomo/prompts.py (JUDGE, with per-category overrides JUDGE_BY_CATEGORY, and verdict()):
Mem0's LLM-as-a-Judge prompt for every category (category 5 is not used).

F1: LoCoMo's token F1 (Maharana et al., 2024, evaluation code as described there and used by A-MEM): lowercase,
commas, punctuation and the words a/an/the/and removed; for category 1 (multi-hop) the gold answer is split at commas
and each part scored against the best comma-separated part of the prediction, averaged. Two departures, both forced:
no Porter stemming (nltk is not installed here), and the prediction is what follows the response's last "Answer:" (or
"Final answer:") marker, the rest of that line or else the next non-empty line, and the last non-empty line when there
is no marker, since a reader may answer step by step. As in LoCoMo's scoring, a category-3 (open-domain) gold answer is
cut at its first ";". F1 is therefore a secondary, comparative measure; J is the primary one."""
import re
import string
from collections import Counter

from flexamem.bench.locomo import prompts
from flexamem.llm.client import get_llm

_LEAD = re.compile(r"^\W*(?:final\s+)?answer\s*[:\-]\s*", re.I)
_MARK = re.compile(r"(?:final\s+)?answer\W*?[:\-]", re.I)


def judge(role: str, q: dict, gold: str, response: str, **meta) -> dict:
    out = get_llm().call(role, prompts.judge_prompt(q["category"], q["question"], gold, response), phase="judge",
                         cache=True, **meta)
    return {"label": prompts.verdict(out), "raw": out}


def final_line(response: str | None) -> str:
    text = response or ""
    marks = list(_MARK.finditer(text))
    if marks:
        rest = [ln.strip().strip("*").strip() for ln in text[marks[-1].end():].splitlines()]
        rest = [ln for ln in rest if ln]
        if rest:
            return rest[0]
    lines = [ln.strip().strip("*").strip() for ln in text.splitlines() if ln.strip()]
    return _LEAD.sub("", lines[-1]) if lines else ""


def _norm(s: str) -> list[str]:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return re.sub(r"\b(a|an|the|and)\b", " ", s).split()


def _f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    same = sum((Counter(p) & Counter(g)).values())
    if not same:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1(response: str | None, gold: str, category: int) -> float:
    pred = final_line(response)
    if category == 3:
        gold = gold.split(";")[0]
    if category == 1:
        parts_g = [x for x in gold.split(",") if x.strip()] or [gold]
        parts_p = [x for x in pred.split(",") if x.strip()] or [pred]
        return sum(max(_f1(p, g) for p in parts_p) for g in parts_g) / len(parts_g)
    return _f1(pred, gold)
