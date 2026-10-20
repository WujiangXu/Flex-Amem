"""Segment interaction into units at natural boundaries (shared by every method).

LongMemEval: a session is cut into ceil(T / UNIT_MAX) balanced units at user-turn boundaries
(a user turn stays with the assistant reply that follows it); units never cross sessions.
SWE: a step is one assistant message plus all of its observations; steps are packed online and a
unit closes as soon as it reaches UNIT_MIN tokens, so closing decisions depend only on the prefix
and units never change once formed.
"""
import json
import math

from flexamem.llm.tokens import count, count_messages
from flexamem.types import Unit

UNIT_MIN, UNIT_MAX = 1000, 2000


def lme_units(sessions: list[dict], start_order: int = 0, starter: str | None = None) -> list[Unit]:
    """sessions: [{"sid", "date", "turns": [{"role","content"}]}] in chronological order. starter: the role whose turns
    open an item (LoCoMo: the conversation's first speaker, its roles being the two speakers' names); default "user"."""
    units, order = [], start_order
    for s in sessions:
        items = []  # (first_turn_idx, last_turn_idx, text)
        for i, t in enumerate(s["turns"]):
            line = f"{t['role']}: {t['content'].strip()}"
            if t["role"] == (starter or "user") or not items:
                items.append([i, i, line])
            else:
                items[-1][1] = i
                items[-1][2] += "\n" + line
        if not items:
            continue
        toks = [count(it[2]) for it in items]
        k = max(1, math.ceil(sum(toks) / UNIT_MAX))
        target = sum(toks) / k
        groups, cur, cur_t = [], [], 0
        for it, t in zip(items, toks):
            cur.append(it)
            cur_t += t
            if cur_t >= target and len(groups) < k - 1:
                groups.append(cur)
                cur, cur_t = [], 0
        if cur:
            groups.append(cur)
        for g in groups:
            text = "\n".join(it[2] for it in g)
            units.append(Unit(uid=f"u{order}", order=order, source=s["sid"], time=s["date"],
                              span=f"session {s['sid']}, turns {g[0][0] + 1}-{g[-1][1] + 1}",
                              text=text, tokens=count(text)))
            order += 1
    return units


ACTION_TOOLS = {"bash": "command", "python": "code"}  # the agent's action tools (SWE, AppWorld) and their argument


def parse_steps(messages: list[dict]) -> list[list[dict]]:
    """Group post-task messages into steps: an assistant message and the messages that follow it."""
    steps = []
    for m in messages:
        if m.get("role") == "exit":
            continue
        if m.get("role") == "assistant" or not steps:
            steps.append([m])
        else:
            steps[-1].append(m)
    return steps


def step_text(step: list[dict], n: int) -> str:
    out = []
    for m in step:
        role = m.get("role")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "assistant":
            out.append(f"[step {n}] assistant:\n{content.strip()}".rstrip())
            for tc in m.get("tool_calls") or []:
                try:
                    name = tc["function"]["name"]
                    if name not in ACTION_TOOLS:  # a context tool (Sculptor): name and arguments
                        out.append(f"> {name} {tc['function']['arguments']}")
                        continue
                    cmd = json.loads(tc["function"]["arguments"]).get(ACTION_TOOLS[name], tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    cmd = str(tc)
                out.append(f"$ {cmd}")
        elif role == "tool":
            out.append(f"[step {n}] output:\n{content.strip()}")
        else:
            out.append(f"[step {n}] {role}:\n{content.strip()}")
    return "\n".join(out)


def swe_units(steps: list[list[dict]], first_step: int = 1) -> list[Unit]:
    """Pack steps into units online. The last unit may still be open (tokens < UNIT_MIN)."""
    units, cur, cur_idx = [], [], []

    def close():
        msgs = [m for s in cur for m in s]
        text = "\n".join(step_text(s, i) for s, i in zip(cur, cur_idx))
        o = len(units)
        span = f"step {cur_idx[0]}" if len(cur_idx) == 1 else f"steps {cur_idx[0]}-{cur_idx[-1]}"
        units.append(Unit(uid=f"u{o}", order=o, source="traj", time="", span=span, text=text,
                          tokens=count(text), messages=msgs, msg_tokens=count_messages(msgs) - 3))

    for i, s in enumerate(steps):
        cur.append(s)
        cur_idx.append(first_step + i)
        if count("\n".join(step_text(x, j) for x, j in zip(cur, cur_idx))) >= UNIT_MIN:
            close()
            cur, cur_idx = [], []
    if cur:
        close()
    return units
