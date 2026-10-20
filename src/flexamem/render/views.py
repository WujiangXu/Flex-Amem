"""Excerpt views of a note, between its short record and its raw text.

Graded views (LongMemEval, exploratory; design deviation 30, TurnIndex): an excerpt keeps the note's turns that best
match the question, verbatim, up to EXCERPT tokens (the best turn is always kept, even if longer), and marks the omitted
turns. Turns are ranked by BM25 against the question words, with IDF over every turn in the store; ties go to the turn
whose best piece is closest to the question in the embedding, then to the earlier turn.

A-MEM Zoom (both benchmarks, exploratory; design deviation 32, PieceIndex): each unit is cut once into pieces of at most
PIECE tokens, runs of whole lines inside one turn (LongMemEval) or one block of a step (SWE: the assistant message with
its commands, or one output). An excerpt keeps the pieces that best match the query, verbatim and in order, up to a limit
on their tokens (the best piece is always kept), ranked as above but over pieces. A kept piece that does not begin its
turn or block is shown under the turn's speaker or the block's step and command."""
import math
import re
import weakref
from collections import Counter

import numpy as np

from flexamem.llm.embed import embed
from flexamem.llm.tokens import count
from flexamem.render import format as F

EXCERPT = 150  # o200k tokens of kept turns, the best one included; a later turn that does not fit is skipped
PIECE = 150    # embedding tie-break: a turn is cut into pieces of at most this many tokens, and scores its best piece
STOP = set("""a an the and or but if then so of to in on at for from by with about as into over after before is are was
were be been being am do does did have has had i me my mine you your yours we our us he she it its they them their what
which who whom whose when where why how this that these those there here not no yes can could would should will shall may
might must just also very more most much many any some all each other than too only own same such up down out off again
further once s t don now""".split())
_SPAN = re.compile(r"session s\d+, turns (\d+)-(\d+)")
_TURN = re.compile(r"\n(?=(?:user|assistant): )")
_BLOCK = re.compile(r"\n(?=\[step \d+\] \w+:(?:\n|$))")  # segmenter.step_text's block headers
_HEAD = re.compile(r"\[step (\d+)\] (\w+):")
_ROLE = re.compile(r"(user|assistant): ")
CMD_LABEL = 100  # characters of a command's first line shown in a label
# Markup that every turn or block carries, written by the segmenter (speaker, step headers) or by mini-swe-agent's
# templates (THOUGHT, output tags, the long-output warning): not words of the content, so BM25 leaves it out of both the
# pieces and the query (on SWE the query's newest unit carries the same markup)
_MARKUP = re.compile(r"<(returncode|elided_chars|warning)>.*?</\1>|</?(?:output|output_head|output_tail|returncode|"
                     r"elided_chars|warning)>|^\[step \d+\] \w+:|^(?:user|assistant): |^thought:", re.S | re.M | re.I)
_SHARED = weakref.WeakKeyDictionary()  # store -> its PieceIndex


def words(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


def content_words(t: str) -> list[str]:
    return words(_MARKUP.sub(" ", t))


def pieces(line: str) -> list[str]:
    """Split one turn into pieces of at most PIECE tokens at sentence or line ends, else between words."""
    if count(line) <= PIECE:
        return [line]
    fine = []
    for s in (s for s in re.split(r"(?<=[.!?\n])", line) if s):
        fine.extend([s] if count(s) <= PIECE else [w for w in re.split(r"(?<= )", s) if w])
    out, cur, used = [], "", 0
    for s in fine:
        k = count(s)
        if cur and used + k > PIECE:
            out.append(cur)
            cur, used = "", 0
        cur += s
        used += k
    return out + [cur] if cur else out


def _cut(w: str) -> list[str]:
    """A run without break points (one long word) cut at character positions into pieces of at most PIECE tokens."""
    out = []
    while count(w) > PIECE:
        lo, hi = 1, min(len(w), 64 * PIECE)  # the longest prefix within PIECE tokens, by bisection
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if count(w[:mid]) <= PIECE else (lo, mid - 1)
        out.append(w[:lo])
        w = w[lo:]
    return out + [w] if w else out


def split(block: str) -> list[str]:
    """Zoom pieces of one turn or block: runs of whole lines of at most PIECE tokens; a longer line is cut at sentence
    ends, then between words, then (one long word) anywhere. The pieces concatenate to the block."""
    fine = []
    for ln in (x for x in re.split(r"(?<=\n)", block) if x):
        if count(ln) <= PIECE:
            fine.append(ln)
            continue
        for s in (s for s in re.split(r"(?<=[.!?])", ln) if s):
            if count(s) <= PIECE:
                fine.append(s)
                continue
            for w in (w for w in re.split(r"(?<=\s)", s) if w):
                fine.extend([w] if count(w) <= PIECE else _cut(w))
    out, cur = [], ""
    for s in fine:
        if cur and count(cur + s) > PIECE:
            out.append(cur)
            cur = ""
        cur += s
    out = out + [cur] if cur else out
    assert "".join(out) == block
    return out


def blocks(n) -> list[tuple[str, str, str]]:
    """(head, command, text) of each turn (LongMemEval) or block (SWE) of a note's raw text, in order; the texts joined
    by newlines give the raw text. head names the turn's speaker or the block's step and role; an SWE output's command is
    the first line of the command that produced it (the k-th of its step's last k "$ " lines, k the step's outputs)."""
    if n.source != "traj":
        out = [(m.group(1) if (m := _ROLE.match(t)) else "", "", t) for t in turns(n)]
    else:
        parts = _BLOCK.split(n.raw)
        heads = [_HEAD.match(t) for t in parts]
        cmds, k_out, seen, out = {}, Counter(), Counter(), []
        for h, t in zip(heads, parts):
            if h and h.group(2) == "assistant":
                cmds[h.group(1)] = [x[2:] for x in t.split("\n") if x.startswith("$ ")]
            elif h and h.group(2) == "output":
                k_out[h.group(1)] += 1
        for h, t in zip(heads, parts):
            if not h:
                out.append(("", "", t))
                continue
            s, c = h.group(1), ""
            if h.group(2) == "output":
                cs = cmds.get(s, [])[-k_out[s]:]
                c = cs[seen[s]] if seen[s] < len(cs) else ""
                seen[s] += 1
            out.append((f"[step {s}] {h.group(2)}", c, t))
    assert "\n".join(t for _, _, t in out) == n.raw
    return out


def turns(n) -> list[str]:
    """The turn lines of a note's raw text ("role: content"; content may span lines). If the split does not give the
    note's span of turns, each line is taken as a turn."""
    t = _TURN.split(n.raw)
    m = _SPAN.fullmatch(n.span)
    return t if m and len(t) == int(m.group(2)) - int(m.group(1)) + 1 else n.raw.split("\n")


class TurnIndex:
    """Turns, their words and token counts, and the BM25 statistics over every turn of a store; piece embeddings are
    computed on first use (local MiniLM, not stored). Rebuilt when the store grows; on LongMemEval each render has its
    own FlowMethod, so each render builds its own index."""

    def __init__(self):
        self.n = -1

    def update(self, store):
        if len(store) == self.n:
            return
        self.n = len(store)
        self.turns = {u: turns(store.notes[u]) for u in store.order}
        self.words = {u: [words(t) for t in ts] for u, ts in self.turns.items()}
        self.tokens = {u: [count(t) for t in ts] for u, ts in self.turns.items()}
        ws = [w for u in store.order for w in self.words[u]]
        self.df = Counter(x for w in ws for x in set(w))
        self.N, self.avgdl = len(ws), (sum(map(len, ws)) / len(ws) if ws else 0) or 1.0
        self.pvecs = {}

    def bm25(self, u: str, j: int, qw: set) -> float:
        tf, dl = Counter(self.words[u][j]), len(self.words[u][j]) or 1
        return sum(math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5)) * tf[w] * 2.2
                   / (tf[w] + 1.2 * (0.25 + 0.75 * dl / self.avgdl)) for w in qw if tf[w])

    def closeness(self, u: str, q) -> list[float]:
        """Per turn, the best cosine of its pieces to the query embedding."""
        if u not in self.pvecs:
            ps = [pieces(t) for t in self.turns[u]]
            e = embed([p for x in ps for p in x])
            self.pvecs[u] = [e[i - len(x):i] for x, i in zip(ps, np.cumsum([len(x) for x in ps]))]
        return [float((v @ q).max()) for v in self.pvecs[u]]

    def excerpter(self, store, qtext: str, q):
        """uid -> the note's excerpt block for this question (cached per question)."""
        qw, cache = set(words(qtext)), {}

        def excerpt(u):
            if u not in cache:
                ts, tok, near = self.turns[u], self.tokens[u], self.closeness(u, q)
                score = [self.bm25(u, j, qw) + 1e-6 * near[j] for j in range(len(ts))]
                keep, used = set(), 0
                for j in sorted(range(len(ts)), key=lambda j: -score[j]):
                    if not keep or used + tok[j] <= EXCERPT:
                        keep.add(j)
                        used += tok[j]
                lines, gap = [], 0
                for j, t in enumerate(ts):
                    if j not in keep:
                        gap += 1
                        continue
                    if gap:
                        lines.append(f"[... {gap} turn{'s' if gap > 1 else ''} omitted]")
                        gap = 0
                    lines.append(t)
                if gap:
                    lines.append(f"[... {gap} turn{'s' if gap > 1 else ''} omitted]")
                cache[u] = F.excerpt(store.notes[u], lines)
            return cache[u]
        return excerpt


class PieceIndex:
    """A-MEM Zoom's pieces of every written unit, with their words, token counts and the BM25 statistics over every
    piece in the store. Units never change, so each is cut once and the statistics grow with the store; a unit's piece
    embeddings (local MiniLM, not stored) are computed on its first excerpt and kept. Words leave out the markup
    (_MARKUP)."""

    @classmethod
    def of(cls, store) -> "PieceIndex":
        """The store's index, shared by every Zoom render of it (on LongMemEval a question's budgets, which render the
        same store in turn), so each unit is cut and embedded once."""
        if store not in _SHARED:
            _SHARED[store] = cls()
        return _SHARED[store]

    def __init__(self):
        self.blocks, self.pieces, self.words, self.tokens, self.vecs = {}, {}, {}, {}, {}
        self.df, self.N, self.total = Counter(), 0, 0

    def update(self, store):
        for u in store.order:
            if u in self.blocks:
                continue
            self.blocks[u] = blocks(store.notes[u])
            self.pieces[u] = [(j, p) for j, (_, _, t) in enumerate(self.blocks[u]) for p in split(t)]  # (block, text)
            self.words[u] = [content_words(p) for _, p in self.pieces[u]]
            self.tokens[u] = [count(p) for _, p in self.pieces[u]]
            self.df.update(x for w in self.words[u] for x in set(w))
            self.N += len(self.words[u])
            self.total += sum(map(len, self.words[u]))

    def bm25(self, u: str, i: int, qw: set) -> float:
        tf, dl, avgdl = Counter(self.words[u][i]), len(self.words[u][i]) or 1, (self.total / self.N if self.N else 0) or 1.0
        return sum(math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5)) * tf[w] * 2.2
                   / (tf[w] + 1.2 * (0.25 + 0.75 * dl / avgdl)) for w in qw if tf[w])

    def closeness(self, u: str, q) -> np.ndarray:
        if u not in self.vecs:
            self.vecs[u] = embed([p for _, p in self.pieces[u]])
        return self.vecs[u] @ q

    def excerpter(self, store, qtext: str, q, limit: int, rank: str = "bm25", record: bool = True) -> "Excerpter":
        return Excerpter(self, store, set(content_words(qtext)), q, limit, rank, record)


class Excerpter:
    """uid -> the unit's zoom excerpt for one query (cached): its record, then its best pieces up to `limit` tokens of
    pieces (the best always kept), in order. Consecutive kept pieces of one turn or block are joined as in the raw text,
    so a turn kept whole appears verbatim. "[...]" marks an omitted stretch inside a turn or block, "[... k turns|blocks
    omitted]" whole omitted ones; a turn or block whose first kept piece is not its first is headed by its speaker or
    step (and command), and an SWE output whose command line is not in the excerpt by "[step n] $ <command>". With
    record=False (A-MEM's notes, zero-build retrieval) the excerpt is headed like a raw block, without a record.
    rank="lead" keeps the unit's first pieces instead (an ablation). kept[uid]: the kept pieces' indices."""

    def __init__(self, index: PieceIndex, store, qw: set, q, limit: int, rank: str = "bm25", record: bool = True):
        assert rank in ("bm25", "lead"), rank
        self.ix, self.store, self.qw, self.q, self.limit, self.rank = index, store, qw, q, limit, rank
        self.record = record
        self.cache, self.kept, self.near = {}, {}, {}

    def score(self, u: str, i: int) -> float:
        """Piece i of unit u against the query: BM25, ties broken by MiniLM closeness."""
        if u not in self.near:
            self.near[u] = self.ix.closeness(u, self.q)
        return self.ix.bm25(u, i, self.qw) + 1e-6 * float(self.near[u][i])

    def __call__(self, u: str) -> str:
        if u not in self.cache:
            ix, n = self.ix, len(self.ix.pieces[u])
            order = range(n) if self.rank == "lead" else sorted(range(n), key=lambda i: (-self.score(u, i), i))
            keep, used = set(), 0
            for i in order:
                if not keep or used + ix.tokens[u][i] <= self.limit:
                    keep.add(i)
                    used += ix.tokens[u][i]
            self.kept[u] = keep
            note, lines = self.store.notes[u], excerpt_lines(ix, self.store, u, keep)
            self.cache[u] = F.excerpt(note, lines) if self.record else F.raw_excerpt(note, lines)
        return self.cache[u]


def excerpt_lines(ix: PieceIndex, store, u: str, keep: set) -> list[str]:
    """The lines of unit u's excerpt that keeps the pieces in keep (see Excerpter)."""
    bl, ps = ix.blocks[u], ix.pieces[u]
    what = "block" if store.notes[u].source == "traj" else "turn"
    segs = [[] for _ in bl]  # per block: [kept, text] runs of consecutive pieces
    for i, (j, p) in enumerate(ps):
        k = i in keep
        if segs[j] and segs[j][-1][0] == k:
            segs[j][-1][1] += p
        else:
            segs[j].append([k, p])
    lines, gap = [], 0
    for (head, cmd, _), runs in zip(bl, segs):
        if not any(k for k, _ in runs):
            gap += 1
            continue
        if gap:
            lines.append(f"[... {gap} {what}{'s' if gap > 1 else ''} omitted]")
            gap = 0
        c = cmd if len(cmd) <= CMD_LABEL else cmd[:CMD_LABEL] + " …"
        if not runs[0][0]:
            lines.append((f"{head} of $ {c}: [...]" if c else f"{head}: [...]") if head else "[...]")
        elif cmd and f"$ {cmd}" not in "\n".join(lines).split("\n"):
            lines.append(f"{head.rsplit(' ', 1)[0]} $ {c}")
        for n, (k, t) in enumerate(runs):
            if k:
                lines.append(t.rstrip("\n"))
            elif n:
                lines.append("[...]")
    if gap:
        lines.append(f"[... {gap} {what}{'s' if gap > 1 else ''} omitted]")
    return lines
