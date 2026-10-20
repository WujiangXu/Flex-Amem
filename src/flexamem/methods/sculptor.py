"""Sculptor (Li et al., ICLR 2026): the model manages its own context with tools, under the shared budget.

Every unit that has left the recent window is a fragment, shown under an ID that is stable across renders, in
chronological order, either raw, summarized (for the focus it was summarized for) or folded to a marker. The model
calls Sculptor's tools on them: summarize_fragment, fold_fragment, restore_fragment, search_context and
get_search_detail (fragment_context is not offered, since the fragments are the shared units). Search is an exact,
case-insensitive substring match ("Ctrl+F") over the raw text of every fragment, whatever it currently shows; its
results come back as tool results and count against the budget like any other message.

Sculptor assumes the whole history fits the window, so under the budget a forced fallback keeps it there. Fragments
the model has not restored must fit SPLIT of the allowance: the oldest raw ones are summarized (focus "main points"),
then the oldest summarized ones are folded. Restored fragments give way only when the whole memory exceeds the
allowance. What the fallback leaves free is the room a restore (or any other growing edit) may use; one that does not
fit is refused with a message. As in Flow, the permanent operations are sized by the larger of the allowance and the
step-independent reference allowance, and during a dip the oldest fragments are hidden for that render only.
Adaptations: docs/method/design.md, deviation 27."""
import hashlib
import json
import re

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.client import get_llm
from flexamem.llm.ledger import pmap
from flexamem.memory.prompts import SCULPTOR_FOCUS, SCULPTOR_SUMMARIZE, SCULPTOR_TOOLS
from flexamem.memory.store import NoteStore
from flexamem.render import format as F
from flexamem.render.renderer import _finalize
from flexamem.types import Note, Unit

SPLIT = 0.5  # share of the allowance for fragments the model has not restored, as Flow's default split
TOOL_NAMES = tuple(t["function"]["name"] for t in SCULPTOR_TOOLS)
# message headers inside a fragment's raw text (memory/segmenter.py); SWE outputs count as the user side
ROLE_HEADER = {"lme": re.compile(r"^(user|assistant): ", re.M),
               "swe": re.compile(r"^\[step \d+\] (assistant|output|user):\n", re.M)}
ROLE_OF = {"user": "user", "assistant": "assistant", "output": "user"}


def _sid(*parts) -> str:
    return hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()


class SculptorMethod(MemoryMethod):
    name = "sculptor"

    def __init__(self, store: NoteStore | None = None, domain="lme", workers=8, split=SPLIT):
        self.store = store if store is not None else NoteStore()
        self.domain, self.workers, self.split = domain, workers, split
        self.status: dict[str, str] = {}     # uid -> raw | summarized | folded
        self.focus: dict[str, str] = {}      # uid -> focus of the summary it shows
        self.summaries: dict[tuple, str] = {}  # (uid, focus) -> summary
        self.restored: set[str] = set()      # restored by the model; the fallback moves them last
        self.fid: dict[str, str] = {}        # uid -> fragment id
        self.uid_of: dict[str, str] = {}     # fragment id -> uid
        self.searches: dict[str, tuple] = {}  # search id -> (uid, segment start, segment end, match start, match end)
        self.hidden: set[str] = set()        # oldest fragments left out of the last render (a dip, or markers overflow)
        self.room = 0                        # tokens the memory text may still grow by (set by render or the reader)
        self.counts: dict[str, int] = {}     # tool calls, outcomes and forced operations, this method's lifetime
        for u in self.store.order:
            self._register(u)

    def _bump(self, key: str, n: int = 1):
        self.counts[key] = self.counts.get(key, 0) + n

    def _register(self, uid: str):
        self.status[uid] = "raw"
        salt = 0
        while (fid := "f" + _sid(uid, salt)[:5]) in self.uid_of:
            salt += 1
        self.fid[uid], self.uid_of[fid] = fid, uid
        n = self.store.notes[uid]
        if n.short:  # LongMemEval builds carry the "main points" summary
            self.summaries[(uid, SCULPTOR_FOCUS)] = n.short

    # ---- write ----
    def write(self, units: list[Unit]) -> None:
        for u in units:
            self.store.add(Note.from_unit(u))
            self._register(u.uid)

    def _summarize(self, uid: str, focus: str) -> str:
        if (uid, focus) not in self.summaries:
            s = get_llm().call("memory", SCULPTOR_SUMMARIZE.format(focus=focus, text=self.store.notes[uid].raw),
                               phase="summarize", cache=True, unit=uid).strip()
            if not s:
                raise ValueError(f"empty Sculptor summary for {uid}")
            self.summaries[(uid, focus)] = s
        return self.summaries[(uid, focus)]

    def summarize_all(self) -> None:
        """LongMemEval build: every fragment's "main points" summary, question-independent, kept in note.short."""
        pmap(lambda u: self._summarize(u, SCULPTOR_FOCUS), list(self.store.order), self.workers)
        for u in self.store.order:
            self.store.notes[u].short = self.summaries[(u, SCULPTOR_FOCUS)]

    # ---- render ----
    def block(self, uid: str, status: str | None = None, focus: str | None = None) -> str:
        n, fid, st = self.store.notes[uid], self.fid[uid], status or self.status[uid]
        if st == "raw":
            return F.fragment_raw(n, fid)
        if st == "summarized":
            f = focus or self.focus[uid]
            return F.fragment_summary(n, fid, f, self.summaries[(uid, f)])
        return F.fragment_folded(n, fid)

    def tokens(self, uid: str) -> int:
        return F.block_tokens(self.block(uid))

    def _set_summary(self, uid: str, focus: str):
        self._summarize(uid, focus)
        self.status[uid], self.focus[uid] = "summarized", focus

    def _enforce(self, perm: int):
        """The forced fallback (permanent): fragments not restored within split * perm, all within perm."""
        h = int(self.split * perm)
        while True:
            free = [u for u in self.store.order if u not in self.restored]
            cost_free = sum(self.tokens(u) for u in free)
            if cost_free > h:
                raw = [u for u in free if self.status[u] == "raw"]
                if raw:  # summarize the oldest raw fragments that cover the excess, in one parallel batch
                    batch, excess = [], cost_free - h
                    for u in raw:
                        batch.append(u)
                        excess -= self.tokens(u)
                        if excess <= 0:
                            break
                    pmap(lambda u: self._set_summary(u, SCULPTOR_FOCUS), batch, self.workers)
                    self._bump("forced_summaries", len(batch))
                    continue
                summ = [u for u in free if self.status[u] == "summarized"]
                if summ:
                    excess = cost_free - h
                    for u in summ:
                        before = self.tokens(u)
                        self.status[u] = "folded"
                        self._bump("forced_folds")
                        excess -= before - self.tokens(u)
                        if excess <= 0:
                            break
                    continue
                # all folded: the markers stay over the share (render hides the oldest beyond the allowance)
            if self.restored and cost_free + sum(self.tokens(u) for u in self.restored) > perm:
                u = next(x for x in self.store.order if x in self.restored)  # oldest restored gives way
                self.restored.discard(u)
                self._set_summary(u, self.focus.get(u, SCULPTOR_FOCUS))
                self._bump("forced_unrestores")
                continue
            break

    def text(self) -> str:
        return F.join([self.block(u) for u in self.store.order if u not in self.hidden])

    def hide(self, n: int):
        """Hide the oldest shown fragments, restored ones last, until about n tokens are freed: the LongMemEval
        reader's last resort when its own tool-call messages leave too little room for the fragments."""
        shown = sorted((u for u in self.store.order if u not in self.hidden), key=lambda u: u in self.restored)
        if not shown:
            raise RuntimeError("Sculptor: every fragment is hidden and the prompt still exceeds the cap")
        for u in shown:
            self.hidden.add(u)
            self._bump("hidden_to_fit")
            n -= self.tokens(u)
            if n <= 0:
                break

    def render(self, query: Query | None, allowance: int, ref_allowance: int | None = None):
        """query is unused: Sculptor selects nothing by similarity; the model edits the fragments itself."""
        allowance = max(allowance, 0)
        order = self.store.order
        before = dict(self.counts)
        stats = {"n_fragments": len(order)}
        if not order or allowance <= 0:
            self.room = 0
            return _finalize(lambda k, d: "", 0, allowance, stats, 0)
        perm = max(allowance, ref_allowance) if ref_allowance is not None else allowance
        self._enforce(perm)
        cost, n_drop = sum(self.tokens(u) for u in order), 0
        while cost > allowance and n_drop < len(order):  # a dip, or folded markers beyond the share: this render only
            cost -= self.tokens(order[n_drop])
            n_drop += 1
        stats["overflow_dropped"] = n_drop  # as Flow's renderer: _finalize adds its own drops
        r = _finalize(lambda k, d: F.join([self.block(u) for u in order[n_drop + d:]]), 0, allowance, stats,
                      len(order) - n_drop)
        self.hidden = set(order[:len(order) - r.stats["n_hist_shown"]])
        self.room = allowance - r.tokens
        r.stats.update(hidden=len(self.hidden), dip=allowance < perm, restored=len(self.restored),
                       **{s: sum(v == s for v in self.status.values()) for s in ("raw", "summarized", "folded")},
                       **{k: v - before.get(k, 0) for k, v in self.counts.items() if k.startswith("forced_")})
        return r

    # ---- tools ----
    def _uid(self, args) -> str | None:
        uid = self.uid_of.get(str(args.get("fragment_id", "")).strip())
        if uid is None:
            self._bump("invalid_ids")
        return uid

    def _change(self, uid: str, status: str, focus: str | None = None) -> bool:
        """Apply a status change (which also shows a hidden fragment) if what it adds to the memory text fits the
        room; what it frees goes back to the room."""
        delta = F.block_tokens(self.block(uid, status, focus)) - (0 if uid in self.hidden else self.tokens(uid))
        if delta > 0 and delta > self.room:  # an edit that shrinks the text always applies, whatever the room
            return False
        self.room -= delta
        self.status[uid] = status
        if focus:
            self.focus[uid] = focus
        self.hidden.discard(uid)
        return True

    def apply(self, name: str, args) -> str:
        """Run one tool call; the reply is the tool result the model sees."""
        self._bump(f"calls_{name}")
        if not isinstance(args, dict):
            self._bump("bad_args")
            return f"Error: the arguments of {name} are not a valid JSON object."
        if name in ("summarize_fragment", "fold_fragment", "restore_fragment"):
            uid = self._uid(args)
            if uid is None:
                return (f"Error: unknown fragment ID {args.get('fragment_id')!r}. Fragment IDs are shown in the "
                        "fragment headers.")
            fid = self.fid[uid]
            if name == "fold_fragment":
                if not self._change(uid, "folded"):  # only a hidden fragment's marker can fail to fit
                    return f"Not enough context budget to show fragment {fid}."
                self.restored.discard(uid)
                return f"Fragment {fid} folded."
            if name == "restore_fragment":
                if self.status[uid] == "raw" and uid not in self.hidden:
                    return f"Fragment {fid} is already shown in full."
                if not self._change(uid, "raw"):
                    self._bump("restores_denied")
                    return (f"Not enough context budget to restore fragment {fid}; fold or summarize other fragments "
                            "first.")
                self.restored.add(uid)
                self._bump("restores_granted")
                return f"Fragment {fid} restored to its original content."
            focus = str(args.get("focus") or "").strip() or SCULPTOR_FOCUS
            self._summarize(uid, focus)
            if not self._change(uid, "summarized", focus):
                self._bump("summaries_denied")
                return f"Not enough context budget to show a summary of fragment {fid}; fold other fragments first."
            self.restored.discard(uid)
            return f"Fragment {fid} summarized (focus: {focus})."
        if name == "search_context":
            return self._search(args)
        if name == "get_search_detail":
            hit = self.searches.get(str(args.get("search_id", "")).strip())
            if hit is None:
                self._bump("invalid_ids")
                return f"Error: unknown search ID {args.get('search_id')!r}."
            ext = _clamp(args.get("extended_context"), 500, 100, 2000)
            return f"[{args['search_id'].strip()}] {self._where(hit[0])}\n{self._snippet(hit, ext)}"
        self._bump("unknown_tool")
        return f"Error: unknown tool {name!r}."

    def _segments(self, uid: str) -> list[tuple[str, int, int]]:
        """(role, start, end) of each message in the fragment's raw text."""
        raw = self.store.notes[uid].raw
        heads = list(ROLE_HEADER[self.domain].finditer(raw))
        return [(ROLE_OF[m.group(1)], m.start(), heads[i + 1].start() if i + 1 < len(heads) else len(raw))
                for i, m in enumerate(heads)] or [("user", 0, len(raw))]

    def _where(self, uid: str) -> str:
        return f"fragment {self.fid[uid]} ({F.when(self.store.notes[uid])})"

    def _snippet(self, hit, ctx: int) -> str:
        uid, s0, s1, m0, m1 = hit
        raw = self.store.notes[uid].raw
        a, b = max(s0, m0 - ctx), min(s1, m1 + ctx)
        return ("..." if a > s0 else "") + raw[a:b].strip() + ("..." if b < s1 else "")

    def _search(self, args) -> str:
        query = str(args.get("query") or "")
        role = args.get("role", "user")
        if not query.strip():
            self._bump("bad_args")
            return "Error: empty search query."
        if role not in ("user", "assistant", "all"):
            self._bump("bad_args")
            return f"Error: role must be 'user', 'assistant' or 'all', not {role!r}."
        k, ctx = _clamp(args.get("max_results"), 10, 1, 50), _clamp(args.get("context_size"), 200, 50, 1000)
        pat = re.compile(re.escape(query), re.I)
        hits, total = [], 0
        for uid in self.store.order:
            raw = self.store.notes[uid].raw
            for r, s0, s1 in self._segments(uid):
                if role != "all" and r != role:
                    continue
                for m in pat.finditer(raw, s0, s1):
                    total += 1
                    if len(hits) < k:
                        hits.append((uid, s0, s1, m.start(), m.end()))
        self._bump("search_hits", total)
        if not hits:
            return f"No matches for {query!r} (role: {role})."
        out = [f"{total} match(es) for {query!r} (role: {role}); showing {len(hits)}:"]
        for h in hits:
            salt = 0
            while self.searches.get(sid := "s" + _sid(*h, salt)[:5], h) != h:
                salt += 1
            self.searches[sid] = h
            out.append(f"[{sid}] {self._where(h[0])}\n{self._snippet(h, ctx)}")
        return "\n\n".join(out)

    def stats(self) -> dict:
        return {"fragments": len(self.store), "restored": len(self.restored), **self.counts}


def _clamp(v, default: int, lo: int, hi: int) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


def parse_args(arguments: str):
    """Tool-call arguments as a dict, or None when they are not a JSON object."""
    try:
        a = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return None
    return a if isinstance(a, dict) else None
