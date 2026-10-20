"""Checkpoints of the recall diagnostic: prefixes of an uncapped reference trajectory.

A checkpoint is the first agent call whose prompt (raw history, counted as the runner's `raw_tokens`:
all messages plus the tool schema) reaches t*B, for each threshold t. Its prefix is the messages before
that call's reply, the state the runner commits as a snapshot. A trajectory that ends before some
threshold may contribute one "last" checkpoint at its final call, if that call's raw history is at
least last_min*B (off when last_min is None)."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from flexamem.adapters.mswea import TOOLS
from flexamem.llm.tokens import count_messages

BUDGET = 8000  # primary SWE budget (16k secondary)
THRESHOLDS = (1.25, 2.0)


@dataclass
class Checkpoint:
    cid: str
    iid: str
    label: str        # threshold as written ("1.25") or "last"
    budget: int
    step: int         # 1-based agent call
    raw_tokens: int
    messages: list = field(repr=False)

    def meta(self) -> dict:
        return {"cid": self.cid, "iid": self.iid, "label": self.label, "budget": self.budget, "step": self.step,
                "n_messages": len(self.messages), "raw_tokens": self.raw_tokens}


def load_messages(traj) -> list[dict]:
    return [m for m in json.loads(Path(traj).read_text())["messages"] if m.get("role") != "exit"]


def call_cuts(messages: list[dict]) -> list[int]:
    """Prefix length at each agent call: a reply is an assistant message, or a user message after a
    format error (mini-swe-agent adds no assistant message then)."""
    return [i for i, m in enumerate(messages) if i >= 2 and m.get("role") in ("assistant", "user")]


def checkpoint_id(iid: str, budget: int, label: str) -> str:
    return f"{iid}@{budget}x{label}"


def find_checkpoints(traj, budget: int = BUDGET, thresholds=THRESHOLDS,
                     last_min: float | None = None) -> list[Checkpoint]:
    """Checkpoints of one trajectory in step order. A threshold reached at the same call as an earlier
    one is not repeated."""
    msgs = load_messages(traj)
    iid = Path(traj).name.removesuffix(".traj.json")
    cuts = call_cuts(msgs)
    raw = [count_messages(msgs[:c], TOOLS) for c in cuts]
    out, seen = [], set()

    def add(k, label):
        seen.add(k)
        out.append(Checkpoint(checkpoint_id(iid, budget, label), iid, label, budget, k + 1, raw[k], msgs[:cuts[k]]))

    missing = False
    for t in thresholds:
        k = next((k for k, r in enumerate(raw) if r >= t * budget), None)
        if k is None:
            missing = True
        elif k not in seen:
            add(k, f"{t:g}")
    if missing and last_min is not None and raw and raw[-1] >= last_min * budget and len(raw) - 1 not in seen:
        add(len(raw) - 1, "last")
    return out
