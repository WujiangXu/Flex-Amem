"""Budgeted rendering of memory text (the part of the context between the pinned task and the recent
raw window). Every renderer returns text whose exact token count is <= the allowance it was given.

render_flow implements Budgeted Context Rendering: all earlier notes appear chronologically as short
records (merged when they do not fit), and the notes most relevant to the current query - found by
vector similarity plus one-hop links - are expanded to their raw text in place of their short
record. History and details split the allowance evenly, and either side lends what it cannot use.
With graded views (exploratory), raw goes to candidates in order within a share of the details room, and the other
candidates that fit get an excerpt of their raw instead. A-MEM Zoom (exploratory) keeps the single pass and degrades a
candidate whose raw text does not fit to its excerpt, if that fits. Merges are permanent, so they are sized by the larger of the current allowance and a step-independent
reference allowance; when the current allowance dips below it, the oldest history is hidden for this
render only instead of being merged.
Three render controls of A-MEM Zoom (2026-10-11, design deviation 39) change one thing each: the order of the same
candidates (shuffle with shuffle_all=False), where the chosen views are shown (after=True: after the history instead of
in place, the selection unchanged), and the resolution rule (fixed=True: every chosen candidate gets its excerpt, never
its raw text).
"""
import random
from dataclasses import dataclass, field

import numpy as np

from flexamem.llm.tokens import count
from flexamem.memory.merge import MergeState
from flexamem.memory.store import NoteStore
from flexamem.render import format as F

TOP_K = 10         # direct vector hits that seed the candidate set
LINK_DISCOUNT = 0.9
STACK_HEADER = "Retrieved details (full records):"
AFTER_HEADER = "Details of earlier steps, shown here instead of at their place in the history above (full records or excerpts):"


@dataclass
class Rendered:
    text: str
    tokens: int
    stats: dict = field(default_factory=dict)


def _finalize(build, n_expand: int, allowance: int, stats: dict, n_hist: int):
    """build(k_expand, k_drop_oldest) -> text. Drop the last expansion, then the oldest history node,
    until the exact count fits."""
    k_exp, k_drop = n_expand, 0
    while True:
        text = build(k_exp, k_drop)
        tok = count(text)
        if tok <= allowance:
            break
        if k_exp > 0:
            k_exp -= 1
            stats["fit_dropped_expansions"] = stats.get("fit_dropped_expansions", 0) + 1
        elif k_drop < n_hist:
            k_drop += 1
            stats["overflow_dropped"] = stats.get("overflow_dropped", 0) + 1
        else:
            text, tok = "", 0
            break
    assert tok <= allowance, (tok, allowance)
    stats.update(tokens=tok, n_expanded=k_exp, n_hist_shown=n_hist - k_drop)
    return Rendered(text, tok, stats)


def candidates(store: NoteStore, q, links: bool, link_discount: float = LINK_DISCOUNT) -> list[tuple[str, float]]:
    sims = store.sims(q)
    if not sims:
        return []
    ranked = sorted(sims, key=lambda u: -sims[u])
    score = {u: sims[u] for u in ranked[:TOP_K]}
    if links:
        for p in ranked[:TOP_K]:
            for c in list(store.notes[p].links) + store.neighbors_in_time(p):
                score[c] = max(score.get(c, -1.0), link_discount * sims[p])
    return sorted(score.items(), key=lambda x: (-x[1], store.notes[x[0]].order))


def render_flow(store: NoteStore, ms: MergeState, q, allowance: int, *, ref_allowance: int | None = None,
                expand: bool = True, links: bool = True, stacked: bool = False, split: float = 0.5,
                link_discount: float = LINK_DISCOUNT, raw_share: float | None = None, excerpt=None,
                floor: bool = True, shuffle: int | None = None, shuffle_all: bool = True, after: bool = False,
                fixed: bool = False) -> Rendered:
    """excerpt: uid -> excerpt block. With raw_share, graded views (raw_share is the share of the details room for raw);
    without, A-MEM Zoom's fit-or-degrade. floor=False (an ablation of Zoom): no history of lines and no merges; the
    candidates get raw text or excerpts by fit-or-degrade over the whole allowance, shown chronologically.
    shuffle: a seed; every unit (shuffle_all, random recovery) or the ranked candidates themselves (not shuffle_all)
    are tried in a random order. after: the views fit-or-degrade chose leave their place in the history and follow it
    under AFTER_HEADER, chronologically; the room pays the header, nothing else changes. fixed: fixed-length recovery,
    each candidate in order gets its excerpt if that fits, never its raw text; the history lends by the excerpts'
    demand."""
    assert excerpt is None or not stacked
    assert not (after or fixed) or (excerpt is not None and raw_share is None and not stacked and floor), \
        "Zoom's controls only"
    if not floor:
        return _render_details_only(store, q, allowance, links, link_discount, excerpt)
    stats = {"merges_before": ms.n_merges}
    ms.sync(store)
    if allowance <= 0 or not store.order:
        return Rendered("", 0, stats)

    def hist_cost():
        return sum(F.block_tokens(ms.block(store, x)) for x in ms.top)

    cands = candidates(store, q, links, link_discount) if expand else []
    if shuffle is not None and expand:  # random recovery (deviation 38): every unit; or the same candidates (39)
        pool = store.order if shuffle_all else [u for u, _ in cands]
        cands = [(u, 0.0) for u in random.Random(shuffle).sample(pool, len(pool))]

    def delta(u):
        full = F.block_tokens(F.expanded(store.notes[u]))
        return full if (stacked or u not in ms.top) else full - F.block_tokens(F.compressed(store.notes[u]))

    def ex_delta(u):
        return F.block_tokens(excerpt(u)) - (F.block_tokens(F.compressed(store.notes[u])) if u in ms.top else 0)

    hc = hist_cost()
    demand = sum((ex_delta if fixed else delta)(u) for u, _ in cands)
    if fixed:  # an excerpt of a short unit (its record + all its raw text) costs more than its raw text, so the
        # history's share, and with it the merges, can differ from the default's either way: both demands are logged
        stats.update(demand=demand, demand_raw=sum(delta(u) for u, _ in cands))

    def history_share(a):
        if not expand:
            return a
        h0 = int(a * split)
        if hc <= h0:
            return hc
        return a - demand if demand < a - h0 else h0

    perm = max(allowance, ref_allowance) if ref_allowance is not None else allowance
    h_perm, h_now = history_share(perm), history_share(allowance)
    while hc > h_perm and ms.merge_step(store):
        hc = hist_cost()
    top = list(ms.top)
    # Hide the oldest history nodes: down to this render's share during a dip, else the shared
    # last-resort overflow rule (logged in stats).
    limit = min(h_now, allowance) if allowance < perm else allowance
    n_drop = 0
    while hc > limit and n_drop < len(top):
        hc -= F.block_tokens(ms.block(store, top[n_drop]))
        n_drop += 1
    stats.update(overflow_dropped=n_drop, dip=allowance < perm)
    top = top[n_drop:]
    visible = {u: t for t in top for u in ms.leaves(t)}

    head = F.block_tokens(STACK_HEADER) if stacked else F.block_tokens(AFTER_HEADER) if after else 0

    def select(room):
        """Candidates in order, each raw if that fits the room left; with excerpts (A-MEM Zoom's fit-or-degrade), a
        candidate whose raw text does not fit gets its excerpt if that fits, else it keeps its line. fixed: the
        excerpt or nothing."""
        chosen = []  # (uid, 2 raw | 1 excerpt)
        for u, s in cands:
            if u not in visible:
                continue
            d = delta(u)
            if not fixed and d <= room:
                chosen.append((u, 2))
                room -= d
            elif excerpt is not None and (d := ex_delta(u)) <= room:
                chosen.append((u, 1))
                room -= d
        return chosen, room

    room = allowance - hc - head
    if excerpt is None or raw_share is None:
        chosen, room = select(room)
        if after:  # the in-place selection, without the header: differs only when the header's tokens tip a choice
            stats["after_selection_differs"] = select(allowance - hc)[0] != chosen
    else:  # graded views: raw within a share of the room, then excerpts for the other candidates that fit
        chosen, raw_room = [], raw_share * room
        for u, s in cands:
            if u in visible and (d := delta(u)) <= raw_room:
                chosen.append((u, 2))
                raw_room -= d
        room -= sum(delta(u) for u, _ in chosen)
        full = {u for u, _ in chosen}
        for u, s in cands:
            if u in visible and u not in full:
                d = ex_delta(u)
                d, lv = (delta(u), 2) if delta(u) <= d else (d, 1)  # an excerpt that costs as much as raw: show raw
                if d <= room:
                    chosen.append((u, lv))
                    room -= d
    if excerpt is not None:
        # An excerpt goes only to a candidate whose raw text did not fit when it was visited. The room shrinks with
        # every choice except the raw text of a unit shorter than its line, so unless such a unit was chosen no excerpt
        # could now be swapped for its raw text; graded views below a raw share of 1 break this. Counted on the
        # selection, before _finalize, like room_left.
        over = sum(delta(u) <= room + ex_delta(u) for u, lv in chosen if lv == 1)
        assert (fixed or over == 0 or (raw_share is not None and raw_share < 1)
                or any(delta(u) < 0 for u, lv in chosen if lv == 2)), f"{over} excerpts whose raw text fits"
        stats.update(excerpt_raw_fits=None if fixed else over, room_left=room)

    def view(u, level):
        return F.expanded(store.notes[u]) if level == 2 else excerpt(u)

    def build(k_exp, k_drop):
        sel = dict(chosen[:k_exp])
        blocks, details = [], []
        for t in top[k_drop:]:
            if t in ms.nodes:
                blocks.append(ms.block(store, t))
                inner = [view(u, sel[u]) for u in ms.leaves(t) if u in sel]
                (details if stacked or after else blocks).extend(inner)
            elif t in sel and not stacked:
                (details if after else blocks).append(view(t, sel[t]))
            else:
                blocks.append(F.compressed(store.notes[t]))
                if t in sel:
                    details.append(view(t, sel[t]))
        if (stacked or after) and details:
            blocks.append(STACK_HEADER if stacked else AFTER_HEADER)
            blocks.extend(details)
        return F.join(blocks)

    r = _finalize(build, len(chosen), allowance, stats, len(top))
    shown = chosen[:r.stats["n_expanded"]]
    r.stats.update(merges=ms.n_merges, merge_depth=ms.depth(), n_top=r.stats["n_hist_shown"],
                   expanded=[u for u, lv in shown if lv == 2], n_notes=len(store.order))
    if excerpt is not None:
        r.stats["excerpted"] = [u for u, lv in shown if lv == 1]
        r.stats.update(raw_tokens=sum(F.block_tokens(F.expanded(store.notes[u])) for u in r.stats["expanded"]),
                       excerpt_tokens=sum(F.block_tokens(excerpt(u)) for u in r.stats["excerpted"]))
    return r


def _render_details_only(store: NoteStore, q, allowance: int, links: bool, link_discount: float, excerpt) -> Rendered:
    """render_flow without the floor: the same candidates in the same order, each raw if that fits the room left, else
    its excerpt if that fits, else nothing; no lines, no merges."""
    stats, allowance = {}, max(allowance, 0)
    if allowance <= 0 or not store.order:
        return Rendered("", 0, stats)
    chosen, room = [], allowance
    for u, _ in candidates(store, q, links, link_discount):
        d = F.block_tokens(F.expanded(store.notes[u]))
        if d <= room:
            chosen.append((u, 2))
            room -= d
        elif excerpt is not None and (d := F.block_tokens(excerpt(u))) <= room:
            chosen.append((u, 1))
            room -= d

    def build(k_exp, k_drop):
        sel = sorted(chosen[:k_exp], key=lambda x: store.notes[x[0]].order)
        return F.join([F.expanded(store.notes[u]) if lv == 2 else excerpt(u) for u, lv in sel])

    r = _finalize(build, len(chosen), allowance, stats, 0)
    shown = chosen[:r.stats["n_expanded"]]
    r.stats.update(expanded=[u for u, lv in shown if lv == 2], excerpted=[u for u, lv in shown if lv == 1],
                   n_notes=len(store.order), room_left=room, merges=0)
    return r


def retrieval_order(store: NoteStore, q, follow_links: bool = True) -> list[str]:
    """A-MEM search_agentic order: each hit followed by its linked notes."""
    sims = store.sims(q)
    out = []
    for u in sorted(sims, key=lambda x: -sims[x]):
        for v in [u] + (list(store.notes[u].links) if follow_links else []):
            if v not in out:
                out.append(v)
    return out


def render_retrieval(store: NoteStore, q, allowance: int, prefix_blocks: list[str] = (), excerpt=None,
                     order=None) -> Rendered:
    """Retrieved raw notes, greedily until the allowance is used, shown chronologically after any
    prefix blocks (Summary+A-MEM puts the summary there). excerpt (fit-or-degrade plugged into A-MEM, an ablation): a
    note whose raw text does not fit gets its excerpt if that fits. order: the notes to try, in order (default A-MEM's
    search_agentic order, retrieval_order)."""
    stats = {}
    room = max(allowance, 0) - sum(F.block_tokens(b) for b in prefix_blocks)
    chosen = []  # (uid, 2 raw | 1 excerpt)
    for u in (retrieval_order(store, q) if order is None else order):
        d = F.block_tokens(F.raw_block(store.notes[u]))
        if d <= room:
            chosen.append((u, 2))
            room -= d
        elif excerpt is not None and (d := F.block_tokens(excerpt(u))) <= room:
            chosen.append((u, 1))
            room -= d

    def build(k_exp, k_drop):
        sel = sorted(chosen[:k_exp], key=lambda x: store.notes[x[0]].order)
        return F.join(list(prefix_blocks) + [F.raw_block(store.notes[u]) if lv == 2 else excerpt(u) for u, lv in sel])

    r = _finalize(build, len(chosen), max(allowance, 0), stats, 0)
    shown = chosen[:r.stats["n_expanded"]]
    r.stats["retrieved"] = sorted((u for u, lv in shown if lv == 2), key=lambda u: store.notes[u].order)
    if excerpt is not None:
        r.stats["excerpted"] = sorted((u for u, lv in shown if lv == 1), key=lambda u: store.notes[u].order)
    return r


def render_recent_raw(store: NoteStore, allowance: int) -> Rendered:
    """Truncation reference: the most recent raw units that fit, chronologically."""
    allowance = max(allowance, 0)
    chosen, room = [], allowance
    for u in reversed(store.order):
        d = F.block_tokens(F.raw_block(store.notes[u]))
        if d > room:
            break
        chosen.append(u)
        room -= d

    def build(k_exp, k_drop):
        return F.join([F.raw_block(store.notes[u]) for u in reversed(chosen[:k_exp])])

    return _finalize(build, len(chosen), allowance, {}, 0)


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v
