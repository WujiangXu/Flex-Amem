"""A-MEM Flow: Linked Memory Notes + Budgeted Context Rendering, with ablation switches.

  expand=False   notes only (the whole allowance goes to history)
  links=False    details chosen by vector similarity alone (no link or temporal neighbours)
  stacked=True   raw details appended after the notes instead of replacing them in place
  evolve=False   links from nearest neighbours, no linker call and no retrieval-field updates
  rewrite=False  the linker chooses links but rewrites no neighbour's retrieval fields (exploratory)
  full_embed=True  the whole retrieval text is embedded, window by window (post-hoc embedding fix, exploratory)
  raw_share=f    graded views: raw for candidates within share f of the details room, excerpts for the others that fit
                 (LongMemEval, exploratory)
  zoom=E         A-MEM Zoom, fit-or-degrade: a candidate whose raw text does not fit gets an excerpt of its best-matching
                 pieces (at most E tokens of pieces) if that fits (exploratory, design deviation 32); with raw_share=1.0,
                 raw text goes first and excerpts take the room left
  recent=True    A-MEM Zoom's excerpts rank pieces against the query's newest unit (SWE: what the agent is doing now)
                 instead of the whole query text (task + newest unit); candidates still come from the whole query
  floor=False    ablation of Zoom: no history of lines, the candidates' raw text or excerpts only
  zoom_rank="lead"  ablation of Zoom: an excerpt keeps the unit's first pieces instead of its best-matching ones
Exploratory variants of the main arm from record_content's findings (2026-10-08, deviation 36; methods/digest.py):
  index=True     explored-so-far floor: one rule-built line naming what every unit in memory called and bound, shown
                 first; the policy reserves its tokens by shrinking the newest unit first, so once a unit has left the
                 recent window the memory is never empty
  digest=True    each record ends with its unit's exact names (calls, REPL names bound), a merge with its leaves' union
  docsig=True    an expanded unit shows each complete API doc it printed as one signature line, the rest verbatim (the
                 record is still written from the full text)
  rand_recovery=True  a control for FlexA-MEM's recovery (2026-10-09, deviation 38): the units to expand are every unit
                 in a random order seeded by the query, instead of the relevance-ranked candidates
Render controls of FlexA-MEM (2026-10-11, deviation 39; one change each, render only):
  rand_order=True  the same relevance-ranked candidates, tried in a random order seeded by the query
  after=True     the same selection, shown after the history under a one-line header instead of in place
  fixed=True     fixed-length recovery: each candidate in order gets its excerpt if that fits, never its raw text
"""
import hashlib

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.ledger import pmap
from flexamem.llm.tokens import count, truncate
from flexamem.memory.linker import flow_link
from flexamem.memory.merge import MergeState
from flexamem.memory.store import NoteStore
from flexamem.memory.writer import flow_write
from flexamem.methods.digest import index_line, names, sign_docs
from flexamem.render import format as F
from flexamem.render.renderer import LINK_DISCOUNT, Rendered, render_flow
from flexamem.render.views import PieceIndex, TurnIndex
from flexamem.types import Note, Unit


# dev configurations of Flow's render (results/decisions/2026-09-24_flow_lme_render_tuning.md): history share of
# the allowance, and the score discount of link neighbours relative to the direct hit they hang off
FLOW_VARIANTS = {"flow_s30": dict(split=0.3), "flow_s30_ld50": dict(split=0.3, link_discount=0.5)}
# graded views: share of the details room for raw, chosen offline on dev60 (results/decisions/2026-09-28_graded_views.md)
RAW_SHARE = 0.6
# A-MEM Zoom = the main arm (w/o evolution) with fit-or-degrade; excerpt length chosen offline on dev60 by the plan's
# rule (E0: C2; results/decisions/2026-10-05_zoom_e0.md)
ZOOM = dict(zoom=300)
# SWE: chosen offline on the main arm's Eval-95 replays among S1/S2 (docs/plans/resolution_allocation.md, E3b)
ZOOM_SWE = dict(zoom=300, recent=True)


class FlowMethod(MemoryMethod):
    name = "flow"

    def __init__(self, store: NoteStore | None = None, *, expand=True, links=True, stacked=False, evolve=True,
                 rewrite=True, full_embed=False, split=0.5, link_discount=LINK_DISCOUNT, raw_share=None, zoom=None, recent=False,
                 floor=True, zoom_rank="bm25", index=False, digest=False, docsig=False, rand_recovery=False,
                 rand_order=False, after=False, fixed=False, workers=8):
        assert sum(map(bool, (rand_recovery, rand_order, after, fixed))) <= 1, "one control at a time"
        assert not (after or fixed) or (zoom is not None and raw_share is None and floor and not stacked), "Zoom only"
        self.store = store if store is not None else NoteStore()
        self.ms = MergeState()
        self.expand, self.links, self.stacked, self.evolve, self.split = expand, links, stacked, evolve, split
        self.rewrite, self.full_embed = rewrite, full_embed
        self.link_discount = link_discount
        self.raw_share, self.turns = raw_share, TurnIndex()
        self.zoom, self.pieces = zoom, PieceIndex.of(self.store) if zoom is not None else None
        self.recent, self.floor, self.zoom_rank = recent, floor, zoom_rank
        self.index, self.digest, self.docsig = index, digest, docsig
        self.rand_recovery, self.rand_order, self.after, self.fixed = rand_recovery, rand_order, after, fixed
        self.names: dict[str, tuple] = {}  # unit text -> (names called, REPL names bound), for index and digest
        self.held: list[tuple] = []        # index: the names of every unit written, in order
        self.workers = workers
        self.link_stats = []

    def _names(self, u: Unit) -> tuple:
        if u.text not in self.names:
            self.names[u.text] = names(u.messages)
        return self.names[u.text]

    def write(self, units: list[Unit]) -> None:
        notes = [Note.from_unit(u) for u in units]
        pmap(flow_write, notes, self.workers)
        for n, u in zip(notes, units):
            if self.digest:
                n.digest = self._names(u)
            if self.docsig:  # after the writer saw the full text; the expansion shows the signatures
                n.raw = sign_docs(n.raw)
                n.raw_tokens = count(n.raw)
        if self.index:
            self.held += [self._names(u) for u in units]
        for n in notes:
            self.link_stats.append(flow_link(self.store, n, evolve=self.evolve, rewrite=self.rewrite,
                                             full_embed=self.full_embed))

    def _index(self, held) -> str:
        return index_line([c for x in held for c in x[0]], [v for x in held for v in x[1]])

    def reserve(self, units: list[Unit]) -> int:
        line = self._index([self._names(u) for u in units]) if self.index else ""
        return F.block_tokens(line) if line else 0

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None):
        line = self._index(self.held) if self.index else ""
        cut = bool(line) and F.block_tokens(line) > allowance
        if cut:  # the policy reserves the line; only its re-render after a merge across the join can cut below it
            line = truncate(line, max(allowance - F.SEP_TOKENS, 0))
        if not line:
            return self._render(query, allowance, ref_allowance)
        t = F.block_tokens(line)
        a = allowance - t
        while True:
            r = self._render(query, a, None if ref_allowance is None else ref_allowance - t)
            text = F.join([line, r.text]) if r.text else line
            tok = count(text)
            if tok <= allowance:
                break
            a -= tok - allowance
        assert tok <= allowance, (tok, allowance)
        return Rendered(text, tok, {**r.stats, "index_tokens": t, **({"index_cut": True} if cut else {})})

    def _render(self, query: Query, allowance: int, ref_allowance: int | None = None):
        excerpt = None
        if self.zoom is not None:
            self.pieces.update(self.store)
            excerpt = self.pieces.excerpter(self.store, self.excerpt_text(query), query.emb, self.zoom, self.zoom_rank)
        elif self.raw_share is not None:
            self.turns.update(self.store)
            excerpt = self.turns.excerpter(self.store, query.text, query.emb)
        shuffle = (int(hashlib.sha1(query.text.encode()).hexdigest()[:8], 16)
                   if self.rand_recovery or self.rand_order else None)
        return render_flow(self.store, self.ms, query.emb, allowance, ref_allowance=ref_allowance,
                           expand=self.expand, links=self.links, stacked=self.stacked, split=self.split,
                           link_discount=self.link_discount, raw_share=self.raw_share, excerpt=excerpt,
                           floor=self.floor, shuffle=shuffle, shuffle_all=not self.rand_order, after=self.after,
                           fixed=self.fixed)

    def excerpt_text(self, query: Query) -> str:
        """The text Zoom's excerpts rank pieces against."""
        return query.recent if self.recent and query.recent else query.text

    def stats(self) -> dict:
        return {"notes": len(self.store), "merges": self.ms.n_merges, "merge_depth": self.ms.depth(),
                "growing_merges": self.ms.n_growing,
                "links": sum(s.get("links", 0) for s in self.link_stats),
                "neighbor_updates": sum(s.get("updates", 0) for s in self.link_stats),
                "invalid_links": sum(s.get("invalid_links", 0) for s in self.link_stats)}
