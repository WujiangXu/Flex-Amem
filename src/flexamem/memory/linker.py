"""Link a newly written note into the store (called in chronological order, one note at a time).

flow_link: our port of A-MEM's process_memory. Neighbours are shown as short records (not raw),
links are validated against the candidate ids and stored in both directions, and the linker may
rewrite only the retrieval fields of the notes it links; updated notes are re-embedded.
flow_link(evolve=False): the "w/o evolution" ablation (nearest-neighbour links, no LLM call).
flow_link(rewrite=False): the exploratory "w/o rewrites" variant (the LLM chooses links but rewrites no neighbour).
flow_link(full_embed=True): the post-hoc embedding fix (design deviation 29). The retrieval text of a note is embedded
whole, one window at a time, instead of being cut off at the embedder's 256 word pieces; a text that fits embeds as before.
amem_evolve: A-MEM's original evolution step, kept faithful except for the repairs noted inline.
"""
import numpy as np

from flexamem.llm.client import get_llm
from flexamem.llm.embed import embed, embed_one, embed_windows
from flexamem.llm.tokens import chunks
from flexamem.memory import prompts as P
from flexamem.memory.store import NoteStore
from flexamem.memory.writer import amem_embed_text, flow_embed_text
from flexamem.types import Note

K_NEIGHBORS = 5
K_KNN_LINKS = 3
CHUNK_TOKENS = 150  # o200k tokens; about 200 MiniLM word pieces, under its 256 limit


def _fmt_list(xs) -> str:
    return ", ".join(xs)


def _flow_vecs(notes: list[Note], full_embed: bool) -> list[tuple]:
    """(main vector, further window vectors or None) of each note's retrieval text."""
    texts = [flow_embed_text(n) for n in notes]
    if not full_embed:
        return [(e, None) for e in embed(texts)]
    return [(v[0], v[1:] if len(v) > 1 else None) for v in embed_windows(texts)]


def flow_link(store: NoteStore, note: Note, evolve: bool = True, rewrite: bool = True, full_embed: bool = False) -> dict:
    [(note.emb, note.chunk_embs)] = _flow_vecs([note], full_embed)
    q = note.emb if note.chunk_embs is None else np.vstack([note.emb[None], note.chunk_embs])
    cands = store.nearest(q, K_NEIGHBORS)
    stats = {"links": 0, "updates": 0}
    if cands and not evolve:
        store.add(note)
        for c in cands[:K_KNN_LINKS]:
            store.link(note.uid, c)
        stats["links"] = min(len(cands), K_KNN_LINKS)
        return stats
    if not cands:
        store.add(note)
        return stats
    lines = []
    for c in cands:
        n = store.notes[c]
        lines.append(f"[{c}] (time: {n.time or 'n/a'}; position: {n.span})\nshort: {n.short}\n"
                     f"keywords: {_fmt_list(n.keywords)}\ntags: {_fmt_list(n.tags)}\nretrieval_desc: {n.desc}")
    tmpl, schema = (P.FLOW_LINKER, P.FLOW_LINKER_SCHEMA) if rewrite else (P.FLOW_LINKER_LINKS, P.FLOW_LINKER_LINKS_SCHEMA)
    prompt = tmpl.format(new_id=note.uid, time=note.time or "n/a", span=note.span, short=note.short,
                         keywords=_fmt_list(note.keywords), tags=_fmt_list(note.tags), desc=note.desc,
                         candidates="\n\n".join(lines))
    r = get_llm().call("memory", prompt, phase="link", schema=schema, cache=True, unit=note.uid)
    store.add(note)
    linked = [c for c in dict.fromkeys(x.strip() for x in r["links"]) if c in cands]
    for c in linked:
        store.link(note.uid, c)
    changed = []
    for u in r.get("updates", []):
        c = u["id"].strip()
        if c not in linked or c in changed:
            continue
        n = store.notes[c]
        n.desc = u["retrieval_desc"].strip() or n.desc
        n.keywords = [k for k in (x.strip() for x in u["keywords"]) if k][:8] or n.keywords
        n.tags = [t for t in (x.strip() for x in u["tags"]) if t][:5] or n.tags
        changed.append(c)
    if changed:
        for c, (e, ch) in zip(changed, _flow_vecs([store.notes[c] for c in changed], full_embed)):
            store.set_emb(c, e, ch)
    stats.update(links=len(linked), updates=len(changed), invalid_links=len(r["links"]) - len(linked))
    return stats


def amem_evolve(store: NoteStore, note: Note) -> dict:
    note.emb = embed_one(amem_embed_text(note))
    # Repair: the embedder sees ~256 word pieces, a fraction of a unit's raw text, so the raw text is
    # also indexed in chunks (immutable, embedded once); a note matches through its best vector.
    note.chunk_embs = embed(chunks(note.raw, CHUNK_TOKENS)) if note.raw else None
    cands = store.nearest(note.emb, K_NEIGHBORS)
    stats = {"links": 0, "updates": 0, "evolved": False}
    if not cands:
        store.add(note)
        return stats
    neigh = ""
    for c in cands:
        n = store.notes[c]
        neigh += (f"memory_id:{c}\ttalk start time:{n.time}\tmemory content: {n.raw}\tmemory context: {n.desc}"
                  f"\tmemory keywords: {str(n.keywords)}\tmemory tags: {str(n.tags)}\n")
    prompt = P.AMEM_EVOLUTION.format(content=note.raw, context=note.desc, keywords=note.keywords,
                                     nearest_neighbors_memories=neigh, neighbor_number=len(cands))
    r = get_llm().call("memory", prompt, phase="link", schema=P.AMEM_EVOLUTION_SCHEMA, cache=True, unit=note.uid)
    changed, retag = [], False
    if r["should_evolve"]:
        stats["evolved"] = True
        for action in r["actions"]:
            if action == "strengthen":
                # Repair: suggested ids are validated against the neighbours (the original stores any
                # string); an empty tag list does not wipe the note's tags. Links stay one-way.
                for c in r["suggested_connections"]:
                    c = c.strip().removeprefix("memory_id:")
                    if c in cands and c not in note.links:
                        note.links.append(c)
                if r["tags_to_update"]:
                    note.tags, retag = list(r["tags_to_update"]), True
            elif action == "update_neighbor":
                ctxs, tagss = r["new_context_neighborhood"], r["new_tags_neighborhood"]
                for i in range(min(len(cands), len(tagss))):
                    n = store.notes[cands[i]]
                    n.tags = list(tagss[i])
                    if i < len(ctxs):
                        n.desc = ctxs[i]
                    changed.append(cands[i])
    store.add(note)
    if retag:
        store.set_emb(note.uid, embed_one(amem_embed_text(note)))
    changed = list(dict.fromkeys(changed))
    if changed:
        # Repair: the original never re-indexes evolved neighbours, so their updated context/tags are
        # invisible to retrieval.
        for c, e in zip(changed, embed([amem_embed_text(store.notes[c]) for c in changed])):
            store.set_emb(c, e)
    stats.update(links=len(note.links), updates=len(changed))
    return stats
