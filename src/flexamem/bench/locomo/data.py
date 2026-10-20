"""LoCoMo loading (Maharana et al., ACL 2024; decision 2026-10-11_locomo_afm), with gold kept apart.

The file is `locomo10.json` (the paper's runs read the KimmoZZZ/locomo HuggingFace mirror; its sha256 is checked at
load). It reproduces the LoCoMo paper's counts and agrees turn by turn with MTEB's copy.

A conversation's memory side sees only its sessions: session dates and turns (role = the speaker's name, content = the
text, plus " [shares a photo: <caption>]" for a shared image, as A-MEM and Mem0 render it; the image-search `query`
field is dropped). Questions of categories 1-4 are used (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop);
category 5 (adversarial) is excluded, as in A-MEM and Mem0. A question's id is <sample_id>_q<index in the original qa
list>. Gold answers and evidence ids stay in a separate record that only the judge reads."""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "artifacts/data/locomo/locomo10.json"
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


@lru_cache(maxsize=1)
def _raw() -> list[dict]:
    b = DATA.read_bytes()
    assert hashlib.sha256(b).hexdigest() == SHA256, "locomo10.json differs from the checked file"
    return json.loads(b)


def _content(t: dict) -> str:
    cap = t.get("blip_caption")
    return t["text"].strip() + (f" [shares a photo: {cap.strip()}]" if cap else "")


def load(cids: list[str] | None = None) -> list[dict]:
    """Returns [{"cid", "speaker_a", "speaker_b", "sessions", "questions", "guard", "gold"}]: sessions as LongMemEval's loader
    gives them ({"sid", "date", "turns": [{"role", "content"}]}), questions [{"qid", "cid", "category", "qtype",
    "question", "question_date", "speaker_a", "speaker_b"}] with the last session's date as the question date, gold
    {qid: {"answer", "evidence"}}; guard: every used question's text, which the build guard rejects even when a run
    takes only some questions (--limit)."""
    out = []
    for c in _raw():
        if cids is not None and c["sample_id"] not in set(cids):
            continue
        conv = c["conversation"]
        nums = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
        sessions = [{"sid": f"s{n}", "date": conv[f"session_{n}_date_time"],
                     "turns": [{"role": t["speaker"], "content": _content(t)} for t in conv[f"session_{n}"]]}
                    for n in nums]
        for s in sessions:
            assert set(s) == {"sid", "date", "turns"}
            for t in s["turns"]:
                assert set(t) == {"role", "content"} and t["role"] in (conv["speaker_a"], conv["speaker_b"])
        questions, gold = [], {}
        for i, q in enumerate(c["qa"]):
            if q["category"] not in CATEGORIES:
                continue
            qid = f"{c['sample_id']}_q{i}"
            questions.append({"qid": qid, "cid": c["sample_id"], "category": q["category"],
                              "qtype": CATEGORIES[q["category"]], "question": q["question"],
                              "question_date": sessions[-1]["date"], "speaker_a": conv["speaker_a"],
                              "speaker_b": conv["speaker_b"]})
            gold[qid] = {"answer": str(q["answer"]), "evidence": list(q.get("evidence") or [])}
        out.append({"cid": c["sample_id"], "speaker_a": conv["speaker_a"], "speaker_b": conv["speaker_b"],
                    "sessions": sessions, "questions": questions, "guard": tuple(q["question"] for q in questions),
                    "gold": gold})
    return out
