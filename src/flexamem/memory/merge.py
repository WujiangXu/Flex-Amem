"""Merge fallback for history compression. When the short records no longer fit their allowance,
the oldest run of (up to) four consecutive top-level nodes at the lowest level is condensed into one
node that keeps references to its children, so any child can still be expanded later. Merges are
permanent within an episode; identical merge prompts are answered from the cache."""
from flexamem.llm.client import get_llm
from flexamem.memory import prompts as P
from flexamem.memory.store import NoteStore
from flexamem.render import format as F
from flexamem.types import MergeNode

GROUP = 4
MERGE_WORDS = 120  # about 1.5x one short record
MAX_WORDS = 150    # hard cap on a merge's text: an over-long reply must not make the history grow


class MergeState:
    def __init__(self):
        self.top: list[str] = []            # chronological top-level node ids (note uids / merge nids)
        self.nodes: dict[str, MergeNode] = {}
        self.synced = 0                      # number of store notes already placed in `top`
        self.n_merges = 0
        self.n_growing = 0                   # merges whose block is longer than its children together

    def sync(self, store: NoteStore):
        for u in store.order[self.synced:]:
            self.top.append(u)
        self.synced = len(store.order)

    def level(self, nid: str) -> int:
        return self.nodes[nid].level if nid in self.nodes else 0

    def leaves(self, nid: str) -> list[str]:
        if nid not in self.nodes:
            return [nid]
        return [u for c in self.nodes[nid].children for u in self.leaves(c)]

    def block(self, store: NoteStore, nid: str, digest: bool = True) -> str:
        """A node's block. Notes with a digest (flow_noevo_digest) show it after their record, and a merge the union of
        its leaves' digests after its text; digest=False leaves them out (the merge prompt's records)."""
        if nid not in self.nodes:
            return F.compressed(store.notes[nid], digest=digest)
        m = self.nodes[nid]
        text = F.merged(m, store.notes[f"u{m.first}"], store.notes[f"u{m.last}"])
        ds = [store.notes[u].digest for u in self.leaves(nid) if store.notes[u].digest]
        if not digest or not ds:
            return text
        called = list(dict.fromkeys(x for d in ds for x in d[0]))
        bound = list(dict.fromkeys(x for d in ds for x in d[1]))
        return f"{text} [{F.names_text(called, bound, cap=F.MERGE_DIGEST_NAMES)}]" if called or bound else text

    def _pick(self, top: list[str]) -> tuple[int, int] | None:
        if len(top) < 2:
            return None
        levels = [self.level(x) for x in top]
        for lv in sorted(set(levels)):
            i = 0
            while i < len(top):
                if levels[i] != lv:
                    i += 1
                    continue
                j = i
                while j < len(top) and levels[j] == lv and j - i < GROUP:
                    j += 1
                if j - i >= 2:
                    return i, j
                i = j
        return 0, min(GROUP, len(top))

    def merge_step(self, store: NoteStore) -> bool:
        """Condense one group in place. Returns False when nothing can be merged."""
        span = self._pick(self.top)
        if span is None:
            return False
        i, j = span
        kids = self.top[i:j]
        records = "\n\n".join(self.block(store, k, digest=False) for k in kids)
        text = get_llm().call("memory", P.FLOW_MERGE.format(words=MERGE_WORDS, records=records),
                              phase="merge", cache=True, children=kids).strip()
        words = text.split()
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS]) + " …"
        leaves = [u for k in kids for u in self.leaves(k)]
        first, last = store.notes[leaves[0]].order, store.notes[leaves[-1]].order
        nid = f"m{first}_{last}"
        self.nodes[nid] = MergeNode(nid=nid, children=kids, level=1 + max(self.level(k) for k in kids),
                                    text=text, first=first, last=last)
        self.top[i:j] = [nid]
        self.n_merges += 1
        if F.block_tokens(self.block(store, nid)) > sum(F.block_tokens(self.block(store, k)) for k in kids):
            self.n_growing += 1
        return True

    def depth(self) -> int:
        return max((self.level(x) for x in self.top), default=0)
