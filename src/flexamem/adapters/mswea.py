"""mini-swe-agent adapter. The agent loop, tool, templates and saved trajectory are mini-swe-agent's;
the only change is which messages the model is shown at each call. A context policy receives the full
trajectory and returns a prompt of at most cap = 0.95 B tokens (tiktoken over all messages plus the tool
schema), asserted before every call. Below the cap every policy passes the trajectory through unchanged.

Memory policies share one layout, so memory methods differ only in the memory text:
    [system, task, user(EARLIER_HEADER + memory text), *recent units as the original messages]
The recent window is the longest suffix of units within RECENT_FRAC * B (always at least the newest
unit). Units leaving it are written to the memory method once, in order. A unit starts at an assistant
message and holds all of its tool results, so tool-call pairs are never split.

Sculptor (agent-driven) uses the same layout; once compaction starts, its context tools are offered next to bash
(their schemas count against the cap) and the agent runs their calls on the memory instead of the container.
"""
import copy
import hashlib
import os
import time

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded, TimeExceeded
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.embed import embed_one
from flexamem.llm.ledger import ledger_context
from flexamem.llm.tokens import MSG_OVERHEAD, count, count_message, count_messages, full_prompt_tokens, truncate
from flexamem.memory.prompts import SCULPTOR_MANAGE, SCULPTOR_TOOLS
from flexamem.memory.segmenter import parse_steps, swe_units
from flexamem.render.renderer import normalize
from flexamem.types import Unit

TOOLS = [BASH_TOOL]
RESERVE = 0.95
RECENT_FRAC = 0.25
MASK_WINDOW = 10  # Complexity Trap (Lindenbauer et al., 2025): observations of the last M=10 turns stay
EARLIER_HEADER = ("Your earlier work on this task no longer fits in the context window. A memory of it follows "
                  "(condensed records and verbatim excerpts, in chronological order). Your most recent steps "
                  "come after it as regular messages.\n\n")
SCULPTOR_HEADER = ("Your earlier work on this task no longer fits in the context window. It follows as fragments in "
                   "chronological order, each shown in full, summarized or folded under its fragment ID; your most "
                   "recent steps come after it as regular messages. " + SCULPTOR_MANAGE + "\n\n")
OMITTED = "[output omitted: too long for the context budget]"
SAVE_EVERY = 10  # trajectory writes are O(steps); upstream rewrites it after every step


def _is_obs(m: dict) -> bool:
    return m.get("role") in ("tool", "user")


def _shrink(msgs: list[dict], limit: int, info: dict) -> list[dict]:
    """Last resort when even the newest step exceeds the cap: cut the largest observation to its head
    and tail until the messages fit `limit` tokens; an output with nothing left to keep is replaced by a
    placeholder and not picked again. Logged in info."""
    msgs = [copy.copy(m) for m in msgs]
    omitted = set()
    while count_messages(msgs) - 3 > limit:
        i = max((j for j, m in enumerate(msgs) if _is_obs(m) and j not in omitted),
                key=lambda j: count_message(msgs[j]), default=None)
        if i is None:
            raise RuntimeError("newest step exceeds the budget even with all its outputs omitted")
        text, excess = msgs[i]["content"], count_messages(msgs) - 3 - limit
        keep = max(0, count(text) - excess - 40) // 2
        if keep == 0:
            msgs[i]["content"] = OMITTED
            omitted.add(i)
        else:
            tail = text[len(truncate(text, count(text) - keep)):]
            msgs[i]["content"] = f"{truncate(text, keep)}\n[... output shortened to fit the context budget ...]\n{tail}"
        info["shrunk_observations"] = info.get("shrunk_observations", 0) + 1
    return msgs


def unique_tool_ids(msgs: list[dict]) -> list[dict]:
    """Claude behind some OpenAI-compatible gateways gives the parallel tool calls of one reply a single id, and refuses the next
    request while that reply is in the prompt ("toolUse blocks ... contain duplicate Ids"; journal 2026-10-06). Gives
    every repeated id a suffix, in the reply and in its tool results in order; the saved trajectory keeps the originals.
    Prompts without a repeated id are returned unchanged."""
    seen, pending, out, changed = set(), {}, [], False
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                i = j = tc.get("id")
                n = 1
                while j in seen:
                    j, n = f"{i}_{n}", n + 1
                seen.add(j)
                pending.setdefault(i, []).append(j)
                if j != i:
                    changed, tc = True, {**tc, "id": j}
                tcs.append(tc)
            m = {**m, "tool_calls": tcs}
        elif m.get("role") == "tool" and pending.get(m.get("tool_call_id")):
            j = pending[m["tool_call_id"]].pop(0)
            if j != m["tool_call_id"]:
                m = {**m, "tool_call_id": j}
        out.append(m)
    return out if changed else msgs


class Policy:
    name = "passthrough"
    compacting = False  # set at the first step over the cap; stays set, since the trajectory only grows

    def __init__(self, budget: int | None, tools: list | None = None):
        self.budget = budget
        self.cap = int(budget * RESERVE) if budget else None
        self.base_tools = tools or TOOLS  # the action tool: bash (SWE) or python (AppWorld)

    @property
    def tools(self) -> list:
        """Offered at the next model call; counted in every prompt size."""
        return self.base_tools

    def build(self, messages: list[dict]) -> tuple[list[dict], dict]:
        n = count_messages(messages, self.base_tools)
        if self.cap is None or n <= self.cap:
            return messages, {"mode": "pass", "raw_tokens": n, "tokens": n}
        self.compacting = True
        pin = count_messages(messages[:2], self.tools)
        if pin > self.cap:  # no policy can drop the pinned prefix; fails at step 1 for every capped arm
            raise RuntimeError(f"fixed prefix (system+task+tools) {pin} tokens exceeds cap {self.cap}")
        info = {"mode": self.name, "raw_tokens": n, "pin_tokens": pin}
        out = self._compact(messages, info)
        info["tokens"] = count_messages(out, self.tools)
        assert info["tokens"] <= self.cap, f"{self.name}: prompt {info['tokens']} > cap {self.cap}"
        return out, info

    def _compact(self, messages, info):
        raise NotImplementedError

    def _fit_suffix(self, pinned, steps, info) -> list[dict]:
        """Shared overflow rule: the longest suffix of whole steps that fits after the pinned prefix."""
        room = self.cap - count_messages(pinned, self.tools)
        kept, used = [], 0
        for s in reversed(steps):
            t = sum(count_message(m) for m in s)
            if used + t > room:
                break
            kept.insert(0, s)
            used += t
        if not kept:
            kept = [_shrink(steps[-1], room, info)]
        info["dropped_steps"] = len(steps) - len(kept)
        return pinned + [m for s in kept for m in s]

    def stats(self) -> dict:
        return {}


class TruncationPolicy(Policy):
    """Reference: pinned task + the most recent whole steps that fit."""
    name = "truncation"

    def _compact(self, messages, info):
        return self._fit_suffix(messages[:2], parse_steps(messages[2:]), info)


class MaskingPolicy(Policy):
    """Observation masking (Lindenbauer et al., 2025): reasoning and actions are kept, observations
    older than the last M turns become a one-line placeholder. If the prompt still exceeds the cap, M
    shrinks (down to 1), then the shared overflow rule drops the oldest steps."""
    name = "masking"

    def _compact(self, messages, info):
        pinned, steps = messages[:2], parse_steps(messages[2:])
        for m_win in range(MASK_WINDOW, 0, -1):
            masked = []
            for i, s in enumerate(steps):
                if i < len(steps) - m_win:
                    s = [{**m, "content": f"Previous {len((m.get('content') or '').splitlines())} lines omitted "
                                         f"for brevity."} if _is_obs(m) else m for m in s]
                masked.append(s)
            out = pinned + [m for s in masked for m in s]
            if count_messages(out, self.tools) <= self.cap:
                info["window"] = m_win
                return out
        info["window"] = 1
        return self._fit_suffix(pinned, masked, info)


TRACE_SHARE = 0.25  # masking_trace: the rule-written log's share of the room once the masked history no longer fits
TRACE_HEADER = ("Your earlier work on this task no longer fits in the context window. A log of its steps follows (one "
                "line per step, written by rules from your actions and their outputs; older lines condensed), in "
                "chronological order. Your more recent steps come after it as regular messages.\n\n")


class MaskedTracePolicy(Policy):
    """Observation masking combined with a rule-written action log (the paper's Limitations, 2026-10-12): as
    MaskingPolicy, but an output older than the last M turns becomes its rule-written outcome (methods/trace.outcome: ok
    with the output on one line, digests of docs and API lists, or the error's last line) instead of a placeholder. When
    the masked history still exceeds the cap (M down to 1), the oldest steps leave the prompt into the rule-written trace
    (methods/trace.TraceMemory, merged deterministically, no LLM call), shown in one message before the kept steps,
    instead of being dropped. The kept steps take at most 1 - TRACE_SHARE of the room (the newest step always stays).
    Like masking, nothing persists between calls: each call recomputes the masked history and writes a fresh trace of
    the steps that do not fit, so a step pushed out by one large output comes back once that output is condensed
    (review of 2026-10-12). Outcome lines are longer than masking's placeholders, so the log starts earlier than
    masking's drops would."""
    name = "masking_trace"
    header = TRACE_HEADER


    def _masked(self, s):
        from flexamem.methods.trace import outcome
        names = {tc.get("id"): tc["function"]["name"] for tc in s[0].get("tool_calls") or []}
        out = []
        for m in s:
            if m.get("role") == "tool":
                out.append({**m, "content": "Output condensed by rule: "
                                            f"{outcome(names.get(m.get('tool_call_id'), 'python'), m.get('content') or '')}"})
            elif _is_obs(m):
                out.append({**m, "content": f"Previous {len((m.get('content') or '').splitlines())} lines omitted for brevity."})
            else:
                out.append(m)
        return out

    def _compact(self, messages, info):
        from flexamem.methods.trace import TraceMemory
        pinned, steps = messages[:2], parse_steps(messages[2:])
        for m_win in range(MASK_WINDOW, 0, -1):  # masking with rule-written outcomes, if the whole history fits
            masked = [self._masked(s) if i < len(steps) - m_win else s for i, s in enumerate(steps)]
            out = pinned + [m for x in masked for m in x]
            if count_messages(out, self.tools) <= self.cap:
                info.update(window=m_win, trace_steps=0)
                return out
        pin_tok = count_messages(pinned, self.tools)
        overhead = MSG_OVERHEAD + count(self.header) + 2
        room = self.cap - pin_tok - overhead
        if room <= 0:
            raise RuntimeError(f"system prompt + task ({pin_tok} tokens) leave no room under cap {self.cap}")
        masked = [self._masked(s) for s in steps[:-1]] + [steps[-1]]
        start, used = len(steps) - 1, count_messages(masked[-1]) - 3
        while start > 0 and used + count_messages(masked[start - 1]) - 3 <= (1 - TRACE_SHARE) * room:
            start -= 1
            used += count_messages(masked[start]) - 3
        kept = [m for x in masked[start:] for m in x]
        if used > room:  # the newest step alone exceeds the room: the shared last resort
            kept = _shrink(kept, room, info)
            used = count_messages(kept) - 3
        trace = TraceMemory()  # a fresh log of the steps that do not fit, every call
        trace.write([Unit(uid=f"s{i + 1}", order=i, source="traj", time="", span=f"step {i + 1}", text="", tokens=0,
                          messages=steps[i]) for i in range(start)])
        allowance = room - used
        while True:
            r = trace.render(None, max(allowance, 0))
            memory = [{"role": "user", "content": self.header + r.text}] if r.text else []
            out = pinned + memory + kept
            excess = count_messages(out, self.tools) - self.cap
            if excess <= 0 or allowance <= 0:
                break
            allowance -= excess
        info.update(window=1, trace_steps=start, kept_steps=len(steps) - start, allowance=allowance,
                    memory_tokens=r.tokens, render=r.stats)
        return out


class MemoryPolicy(Policy):
    """Any MemoryMethod (Flow and its ablations, A-MEM, Summary+A-MEM, MemGAS, ReadAgent) behind one layout."""
    header = EARLIER_HEADER

    def __init__(self, method: MemoryMethod, budget: int, task: str, name: str | None = None,
                 tools: list | None = None):
        super().__init__(budget, tools)
        self.method, self.name, self.task = method, name or method.name, task
        self.recent_start = 0   # first unit of the recent window; never moves back
        self.written: list[str] = []  # texts of units handed to the method, in order
        self.task_emb = None
        self._prev_memory = ""  # last memory message, for the cache-reuse log

    def _compact(self, messages, info):
        pinned = messages[:2]
        units = swe_units(parse_steps(messages[2:]))
        assert all(u.text == t for u, t in zip(units, self.written)), "a written unit changed"
        pin_tok = count_messages(pinned, self.tools)
        overhead = MSG_OVERHEAD + count(self.header) + 2
        if pin_tok + overhead >= self.cap:
            raise RuntimeError(f"system prompt + task ({pin_tok} tokens) leave no room under cap {self.cap}")
        reserve = self.method.reserve(units[:-1])  # 0 for every method but Flow's explored-so-far floor
        if reserve:
            info["reserve"] = reserve
        start, tok = len(units) - 1, units[-1].msg_tokens
        while start > self.recent_start and tok + units[start - 1].msg_tokens <= int(RECENT_FRAC * self.budget):
            start -= 1
            tok += units[start].msg_tokens
        while start < len(units) - 1 and pin_tok + tok + overhead + reserve > self.cap:  # recent gives way first
            tok -= units[start].msg_tokens
            start += 1
        recent = [m for u in units[start:] for m in u.messages]
        if pin_tok + tok + overhead + reserve > self.cap:
            if reserve:  # tokens the floor takes from the newest unit beyond what the shared rule takes
                info["reserve_cut"] = min(reserve, pin_tok + tok + overhead + reserve - self.cap)
            try:
                recent = _shrink(recent, self.cap - pin_tok - overhead - reserve, info)
            except RuntimeError:  # the newest unit has no room for the floor's reserve: shrink as the shared rule does
                if not reserve:
                    raise
                info.update(reserve_dropped=reserve, reserve_cut=0)
                info.pop("shrunk_observations", None)
                recent = _shrink(recent, self.cap - pin_tok - overhead, info)
            tok = count_messages(recent) - 3
        new = units[len(self.written):start]
        if new:
            with ledger_context(stage="memory_write"):
                self.method.write(new)
            self.written += [u.text for u in new]
        self.recent_start = start

        if self.task_emb is None:
            self.task_emb = embed_one(self.task, phase="embed")
        q = Query(text=f"{self.task}\n\n{units[-1].text}",
                  emb=normalize(self.task_emb + embed_one(units[-1].text, phase="embed")), recent=units[-1].text,
                  now=units[-1].order)
        allowance = self.cap - pin_tok - tok - overhead
        # permanent memory operations (merges, persisted summaries) are sized by this step-independent
        # allowance, so one unusually large recent window cannot force an irreversible compression
        ref_allowance = self.cap - pin_tok - overhead - int(RECENT_FRAC * self.budget)
        while True:
            with ledger_context(stage="memory_render"):
                r = self.method.render(q, allowance, ref_allowance=ref_allowance)
            memory = [{"role": "user", "content": self.header + r.text}] if r.text else []
            out = pinned + memory + recent
            excess = count_messages(out, self.tools) - self.cap
            if excess <= 0:
                break
            allowance -= excess  # token merges across the join; re-render slightly smaller
        text = memory[0]["content"] if memory else ""
        same = len(os.path.commonprefix([self._prev_memory, text]))  # unchanged start, reusable by a prompt cache
        self._prev_memory = text
        info.update(n_units=len(units), recent_start=start, recent_tokens=tok, written=len(self.written),
                    allowance=allowance, ref_allowance=ref_allowance, memory_tokens=r.tokens, render=r.stats,
                    header_tokens=count(self.header), memory_prefix_tokens=count(text[:same]), memory_sha=hashlib.sha1(text.encode()).hexdigest()[:12])
        return out

    def stats(self) -> dict:
        return {"units_written": len(self.written), **self.method.stats()}


class SculptorPolicy(MemoryPolicy):
    """Sculptor: the fragments are the memory text; from the first compaction on, the model may also call the
    context tools, which edit the fragments (the next render shows the edit) or return search results."""
    header = SCULPTOR_HEADER

    @property
    def tools(self):
        return self.base_tools + SCULPTOR_TOOLS if self.compacting else self.base_tools

    def apply_tool(self, name: str, args) -> dict:
        with ledger_context(stage="memory_tool"):
            out = self.method.apply(name, args)
        return {"output": out, "returncode": 0, "exception_info": ""}


class ContextAgent(DefaultAgent):
    """DefaultAgent whose model call sees policy.build(self.messages) instead of the full trajectory.
    self.messages (and the saved trajectory) always hold the complete, unmodified interaction."""

    def __init__(self, model, env, *, policy: Policy, on_step=None, **kwargs):
        super().__init__(model, env, **kwargs)
        self.policy, self.on_step = policy, on_step
        self.context_log: list[dict] = []

    def query(self) -> dict:
        # limit checks as in DefaultAgent.query
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded({"role": "exit", "content": "LimitsExceeded",
                                  "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded({"role": "exit", "content": "TimeExceeded",
                                "extra": {"exit_status": "TimeExceeded", "submission": ""}})
        self.n_calls += 1
        t0 = time.time()
        with ledger_context(step=self.n_calls):
            prompt, info = self.policy.build(self.messages)
            prompt = unique_tool_ids(prompt)
            info.update(step=self.n_calls, render_s=round(time.time() - t0, 3))
            try:
                message = self.model.query(prompt, tools=self.policy.tools)
            except FormatError:  # the call was made and billed; keep its context record
                info.update(api_prompt_tokens=(self.model.last_usage or {}).get("prompt_tokens"), format_error=True)
                self.context_log.append(info)
                raise
        usage = (message.get("extra", {}).get("response") or {}).get("usage") or {}
        info["api_prompt_tokens"] = full_prompt_tokens(usage)
        self.context_log.append(info)
        if self.on_step:
            self.on_step(self, info)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        # as DefaultAgent.execute_actions; context-tool calls run in the policy, in order with the bash calls
        outputs = [self.policy.apply_tool(a["tool"], a["args"]) if "tool" in a else self.env.execute(a)
                   for a in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def save(self, path, *extra_dicts) -> dict:
        if self.messages and self.messages[-1].get("role") != "exit" and self.n_calls % SAVE_EVERY:
            return {}
        return super().save(path, *extra_dicts)

    def serialize(self, *extra_dicts) -> dict:
        ctx = {"info": {"context": {"policy": self.policy.name, "budget": self.policy.budget, "cap": self.policy.cap,
                                    "memory": self.policy.stats(), "log": self.context_log}}}
        return super().serialize(ctx, *extra_dicts)
