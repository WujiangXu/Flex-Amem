"""ReadAgent-style online memory (Lee et al., ICML 2024): each unit is a "page" with a gist; at each
render one lookup call picks pages, whose raw text replaces their gists in place. The allowance is
split as in Flow: gists beyond the history share are hidden oldest-first (ReadAgent has no merge
mechanism) so that pages can be expanded, and room the chosen pages leave unused is lent back to the
newest hidden gists. Pages are numbered over the shown gists only: absolute numbers starting above 1
made the lookup request the hidden page just before the first shown one (docs/logs/debugging_journal.md).
On SWE the lookup prompt shows no step numbers (neither page spans nor the `[step n]` tags that the latest progress
and many gists carry): with them, the lookup asked for the latest step's number as a page in over a third of renders."""
import re
import statistics

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.client import get_llm
from flexamem.llm.ledger import pmap
from flexamem.llm.tokens import truncate
from flexamem.memory.prompts import READAGENT_LOOKUP
from flexamem.memory.store import NoteStore
from flexamem.memory.writer import parse_pages, readagent_gist
from flexamem.render import format as F
from flexamem.render.renderer import _finalize
from flexamem.types import Note, Unit

MAX_PAGES_CAP = 20
STEP_TAG = re.compile(r"^\[step \d+\] ", re.M)  # memory/segmenter.py::swe_units
SPLIT = 0.5  # history share of the allowance, as Flow's default split
LOOKUP_GOALS = {
    "lme": dict(goal="question", goal_short="question", no_answer_yet="answer the question", goal_label="Question"),
    "swe": dict(goal="software engineering task you are working on", goal_short="next step of the task",
                no_answer_yet="take the next step", goal_label="Task and latest progress"),
}


class ReadAgentMethod(MemoryMethod):
    name = "readagent"

    def __init__(self, store: NoteStore | None = None, domain="lme", workers=8, query_tokens=3000):
        self.store = store if store is not None else NoteStore()
        self.domain, self.workers, self.query_tokens = domain, workers, query_tokens

    def write(self, units: list[Unit]):
        notes = [Note.from_unit(u) for u in units]
        pmap(readagent_gist, notes, self.workers)
        for n in notes:
            self.store.add(n)

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None):
        allowance = max(allowance, 0)
        order = self.store.order
        page = {u: i + 1 for i, u in enumerate(order)}
        gist_tok = {u: F.block_tokens(F.gist_block(self.store.notes[u], page[u])) for u in order}
        n_drop, cost = 0, sum(gist_tok.values())
        h0 = int(allowance * SPLIT)
        while cost > h0 and n_drop < len(order):
            cost -= gist_tok[order[n_drop]]
            n_drop += 1
        shown = order[n_drop:]
        stats = {"overflow_dropped": n_drop, "gist_tokens": cost, "n_pages": len(order)}
        if not shown:
            return _finalize(lambda k, d: "", 0, allowance, stats, 0)
        delta = {u: F.block_tokens(F.page_block(self.store.notes[u], page[u])) - gist_tok[u] for u in shown}
        room = allowance - cost
        max_pages = int(max(1, min(MAX_PAGES_CAP, room // max(1, statistics.median(delta.values())))))
        page = {u: i + 1 for i, u in enumerate(shown)}
        swe = self.domain == "swe"
        if swe:
            gists = STEP_TAG.sub("", F.join([F.lookup_page(self.store.notes[u], page[u]) for u in shown]))
            qtext = STEP_TAG.sub("", query.text)
        else:
            gists, qtext = F.join([F.gist_block(self.store.notes[u], page[u]) for u in shown]), query.text
        prompt = READAGENT_LOOKUP.format(gists=gists, max_pages=max_pages, query=truncate(qtext, self.query_tokens),
                                         **LOOKUP_GOALS[self.domain])
        reply = get_llm().call("memory", prompt, phase="lookup", cache=True)
        by_page = {page[u]: u for u in shown}
        requested, chosen = parse_pages(reply), []
        for p in requested[:max_pages]:
            u = by_page.get(p)
            if u and delta[u] <= room:
                chosen.append(u)
                room -= delta[u]
        # page numbers are those of the lookup prompt, renumbered below; the unit ids stay valid
        stats.update(requested=requested, requested_units=[by_page.get(p) for p in requested], max_pages=max_pages)
        while n_drop > 0 and gist_tok[order[n_drop - 1]] <= room:  # lend unused room back to history
            n_drop -= 1
            room -= gist_tok[order[n_drop]]
        shown = order[n_drop:]
        page = {u: i + 1 for i, u in enumerate(shown)}
        stats.update(overflow_dropped=n_drop, n_shown=len(shown))

        def build(k_exp, k_drop):
            sel = set(chosen[:k_exp])
            return F.join([F.page_block(self.store.notes[u], page[u]) if u in sel
                           else F.gist_block(self.store.notes[u], page[u]) for u in shown[k_drop:]])

        r = _finalize(build, len(chosen), allowance, stats, len(shown))
        r.stats["expanded"] = chosen[:r.stats["n_expanded"]]
        return r

    def stats(self):
        return {"pages": len(self.store)}
