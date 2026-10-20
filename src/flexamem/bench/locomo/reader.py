"""LoCoMo's reader: one call with prompts.READER (LongMemEval's Chain-of-Note prompt by default). B bounds the whole
reader prompt, as on LongMemEval; the memory gets what the template with an empty history leaves."""
from flexamem.bench.locomo import prompts
from flexamem.llm.client import get_llm
from flexamem.llm.tokens import count_messages


def history_allowance(q: dict, cap: int) -> int:
    """Tokens left for memory text once the rest of the reader prompt is counted (+2 slack for joins)."""
    return cap - count_messages([{"role": "user", "content": prompts.reader_prompt(q, "")}]) - 2


def read(q: dict, history: str, cap: int | None, **meta) -> tuple[str, int]:
    msgs = [{"role": "user", "content": prompts.reader_prompt(q, history)}]
    n = count_messages(msgs)
    if cap is not None:
        assert n <= cap, f"reader prompt {n} > cap {cap}"
    return get_llm().call("reader", msgs, phase="reader", cache=True, **meta), n
