"""Timeline + BM25 pieces: a hybrid baseline the paper reviews asked for (2026-10-09; design deviation 38). It keeps
FlexA-MEM's timeline, the main arm's one-line records of every unit, merged when they do not fit, in the history share
of the allowance. FlexA-MEM's recovery (candidates by vector similarity and links, each shown raw or as an excerpt by
fit-or-degrade) is replaced by zero-build BM25 piece retrieval (rag_pieces, methods/rag.py) in the rest. Same notes,
same timeline, another recovery: does any timeline plus keyword retrieval do what FlexA-MEM's allocation does?

The timeline is rendered first, within split x allowance (its permanent merges sized by split x ref_allowance, as the
main arm's history); a timeline shorter than its share lends the rest, which BM25 pieces fill, shown after a one-line
divider per unit chronologically, as rag_pieces shows them. Differences from FlexA-MEM besides the recovery (review of
2026-10-09): the timeline never borrows from the pieces' room (BM25 pieces always want more), and the pieces rank
against task + newest unit, as rag_pieces does, where FlexA-MEM's SWE/AppWorld excerpts rank against the newest unit."""
from flexamem.interface import MemoryMethod, Query
from flexamem.llm.tokens import count
from flexamem.methods.flow import FlowMethod
from flexamem.methods.rag import PieceRetrieval
from flexamem.memory.store import NoteStore
from flexamem.render import format as F
from flexamem.render.renderer import Rendered

PIECES_HEADER = "Retrieved excerpts of earlier steps (verbatim, chosen by keyword match to your latest step):"


class TimelinePieces(MemoryMethod):
    name = "timeline_pieces"

    def __init__(self, store: NoteStore | None = None, *, split: float = 0.5, workers: int = 8):
        self.flow = FlowMethod(store, evolve=False, expand=False, split=split, workers=workers)  # the timeline
        self.store, self.split = self.flow.store, split
        self.pieces = PieceRetrieval(self.store)  # the same notes' raw text, cut into BM25 pieces

    def write(self, units) -> None:
        self.flow.write(units)

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None) -> Rendered:
        a = max(allowance, 0)
        ref = None if ref_allowance is None else int(ref_allowance * self.split)
        t = self.flow.render(query, int(a * self.split), ref)
        room = a - (F.block_tokens(t.text) if t.text else 0) - F.block_tokens(PIECES_HEADER)
        while True:
            p = self.pieces.render(query, max(room, 0))
            text = F.join([x for x in (t.text, PIECES_HEADER if p.text else "", p.text) if x])
            tok = count(text) if text else 0
            if tok <= a or room <= 0:
                break
            room -= tok - a  # a token merge across the join; re-render the pieces slightly smaller
        if tok > a:  # only when the timeline alone exceeds the allowance, which its share rules out
            text, tok = t.text, t.tokens
        assert tok <= a, (tok, a)
        return Rendered(text, tok, {**t.stats, "tokens": tok, "timeline_tokens": t.tokens,
                                    "pieces": p.stats.get("pieces", 0), "piece_units": p.stats.get("units", 0),
                                    "pieces_tokens": p.tokens})

    def stats(self) -> dict:
        return self.flow.stats()
