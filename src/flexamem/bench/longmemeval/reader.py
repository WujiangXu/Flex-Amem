"""Reader: LongMemEval's Chain-of-Note style reading prompt (verbatim from the paper), shared by every
method. The token budget B bounds the whole reader prompt. Sculptor's reader uses the same prompt in a tool loop
(read_sculptor)."""
from flexamem.llm.client import get_llm
from flexamem.llm.tokens import count, count_messages, truncate
from flexamem.methods.sculptor import parse_args

READ_PROMPT = ("I will give you several history chats between you and a user. Please answer the question based on the "
               "relevant chat history. Answer the question step by step: first extract all the relevant information, "
               "and then reason over the information to get the answer.\n\n"
               "History Chats: {chat_history}\n\nCurrent Date: {question_date}\n\nQuestion: {question}\n"
               "Answer (step by step):")
HISTORY_LEAD = "\n\n"  # history starts on its own line


def prompt(item: dict, history: str) -> str:
    return READ_PROMPT.format(chat_history=HISTORY_LEAD + history, question_date=item["question_date"],
                              question=item["question"])


def history_allowance(item: dict, cap: int) -> int:
    """Tokens left for memory text once the rest of the reader prompt is counted (+2 slack for joins)."""
    return cap - count_messages([{"role": "user", "content": prompt(item, "")}]) - 2


def read(item: dict, history: str, cap: int | None, salt: str | None = None, **meta) -> tuple[str, int]:
    """salt separates the reader cache between seeds (identical prompts otherwise share one answer)."""
    msgs = [{"role": "user", "content": prompt(item, history)}]
    n = count_messages(msgs)
    if cap is not None:
        assert n <= cap, f"reader prompt {n} > cap {cap}"
    return get_llm().call("reader", msgs, phase="reader", cache=True, salt=salt, **meta), n


# ---- Sculptor: the reader manages its context with tools (Li et al., ICLR 2026) ----
MAX_TOOL_CALLS = 20  # the paper's limit on tool calls per question
RESULT_RESERVE = 64  # room kept for the result message of a tool call that grows the history
REMOVED = "[result removed to fit the context budget]"


def _saving(m: dict) -> int:
    """Tokens saved by replacing a tool result with REMOVED (0 or less: a short or already removed result)."""
    return count_messages([m]) - count_messages([{**m, "content": REMOVED}]) if m["role"] == "tool" else 0


def _stale(msgs: list[dict], latest: int) -> list[dict]:
    """The prompt with every removable tool result (before index `latest`) removed: what _fit can free."""
    return [{**m, "content": REMOVED} if i < latest and _saving(m) > 0 else m for i, m in enumerate(msgs)]


def _fit(msgs: list[dict], cap: int, tools: list, latest: int, stats: dict) -> list[dict]:
    """Fit the prompt to the cap: tool results before index `latest` are removed oldest first, then the largest of
    the latest results is cut to its head. The result can still exceed the cap when the tool-call messages alone
    outgrow the room (read_sculptor then hides fragments)."""
    msgs = [dict(m) for m in msgs]
    over = count_messages(msgs, tools) - cap
    for i, m in enumerate(msgs):
        if over <= 0:
            break
        if i < latest and (save := _saving(m)) > 0:
            over -= save
            m["content"] = REMOVED
            stats["results_removed"] = stats.get("results_removed", 0) + 1
    while over > 0:
        i = max((j for j in range(latest, len(msgs)) if msgs[j]["role"] == "tool"),
                key=lambda j: count(msgs[j]["content"]), default=None)
        if i is None or count(msgs[i]["content"]) <= 16:
            break
        keep = max(0, count(msgs[i]["content"]) - over - 16)
        msgs[i]["content"] = truncate(msgs[i]["content"], keep) + "\n[... cut to fit the context budget]"
        over = count_messages(msgs, tools) - cap
        stats["results_cut"] = stats.get("results_cut", 0) + 1
    return msgs


def read_sculptor(item: dict, method, cap: int, salt: str | None = None, **meta) -> tuple[str, int, dict]:
    """The reader sees Sculptor's unified system prompt, the tools, and the verbatim reading prompt with the fragments
    as its history. The first turn must call a tool (the paper's tool_choice="required"); then the model calls tools
    until it answers without one, at most MAX_TOOL_CALLS calls, after which one last turn may not call tools. Each turn
    re-renders the history with the fragments' current state (the paper's in-place tool effects); every prompt is
    asserted within the cap. A growing edit may use the room left once earlier tool results are removed (they rank
    below the fragments). Returns (answer, largest prompt, stats)."""
    from flexamem.memory.prompts import SCULPTOR_SYSTEM, SCULPTOR_TOOLS
    tools, system = SCULPTOR_TOOLS, {"role": "system", "content": SCULPTOR_SYSTEM}

    def messages(turns):
        return [system, {"role": "user", "content": prompt(item, method.text())}] + turns

    allowance = cap - count_messages([system, {"role": "user", "content": prompt(item, "")}], tools) - 2
    r = method.render(None, allowance)
    stats = {"allowance": allowance, "initial": r.stats, "tool_calls": 0, "turns": 0}
    turns, latest, n_max, n_sum = [], 0, 0, 0
    while True:
        msgs = _fit(messages(turns), cap, tools, latest + 2, stats)
        while (over := count_messages(msgs, tools) - cap) > 0:  # tool-call messages alone outgrew the room
            method.hide(over)
            msgs = messages(msgs[2:])
        turns = msgs[2:]
        n = count_messages(msgs, tools)
        assert n <= cap, f"Sculptor reader prompt {n} > cap {cap}"
        n_max, n_sum = max(n_max, n), n_sum + n
        choice = "required" if not turns else ("none" if stats["tool_calls"] >= MAX_TOOL_CALLS else "auto")
        reply = get_llm().call("reader", msgs, phase="reader", cache=True, salt=salt, tools=tools, tool_choice=choice,
                               **meta)
        stats["turns"] += 1
        if not reply["tool_calls"] or choice == "none":
            stats["forced_final"], stats["mean_prompt"] = choice == "none", round(n_sum / stats["turns"], 1)
            stats["method"] = method.stats()
            return reply["content"], n_max, stats
        turns.append({"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]})
        latest = len(turns)
        for c in reply["tool_calls"]:
            stats["tool_calls"] += 1
            if stats["tool_calls"] > MAX_TOOL_CALLS:
                out = f"Tool call limit ({MAX_TOOL_CALLS}) reached; answer the question now."
            else:
                method.room = cap - count_messages(_stale(messages(turns), latest + 2), tools) - RESULT_RESERVE
                out = method.apply(c["function"]["name"], parse_args(c["function"]["arguments"]))
            turns.append({"role": "tool", "tool_call_id": c["id"], "content": out})
