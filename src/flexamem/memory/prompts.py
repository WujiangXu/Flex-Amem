"""All memory-side prompts. Baseline prompts are copied verbatim from their sources (cited per
constant); Flow's writer/linker/merge prompts are our own and are shared by both domains."""

# ---------------------------------------------------------------- Flow (ours) --------------------
FLOW_WRITER = """You are writing a memory note for one unit of an ongoing interaction (for example a conversation with a user, or an agent's work log). The note is used in two ways. Its SHORT record stands in for the unit once the raw text no longer fits in context. Its retrieval fields are used later to find the unit again when its exact details are needed.

Return a JSON object with:
- "short": at most 80 words. A self-contained record of what this unit contains: the facts stated, decisions, actions and their outcomes, keeping the specific names, numbers, dates, identifiers, file paths and error messages most likely to matter later. State the content itself; do not describe the unit generically (not "the user talked about their trip" but what they said about it).
- "keywords": 3 to 8 specific terms (entities, identifiers, key concepts) that a later search for this unit would use.
- "tags": 2 to 5 broad categories.
- "retrieval_desc": one or two sentences on what detailed information the raw unit holds beyond the short record, so that a later need for those details can be matched to this unit.

Unit time: {time}
Unit position: {span}
Unit text:
{raw}"""

FLOW_WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "short": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "retrieval_desc": {"type": "string"},
    },
    "required": ["short", "keywords", "tags", "retrieval_desc"],
    "additionalProperties": False,
}

FLOW_LINKER = """You maintain a linked memory of an ongoing interaction. A new note has just been written. Decide which of the candidate earlier notes it should be linked to. Link two notes when recalling one makes the other useful: they concern the same entity, task, file or topic, or one continues, revises, corrects, contradicts or depends on the other. Do not link notes that are only generically similar.

For each earlier note you link, you may rewrite its retrieval fields ("retrieval_desc", "keywords", "tags") so that they also reflect what the new note reveals about it (for example, that a stated value was later changed, or that an attempted fix was later reverted). If nothing new is learned, repeat its current fields. You cannot change any short record or raw text.

New note {new_id} (time: {time}; position: {span})
short: {short}
keywords: {keywords}
tags: {tags}
retrieval_desc: {desc}

Candidate earlier notes:
{candidates}

Return a JSON object with "links" (the ids of candidates to link; may be empty) and "updates" (one entry per linked candidate whose fields you rewrite, each with "id", "retrieval_desc", "keywords", "tags")."""

FLOW_LINKER_SCHEMA = {
    "type": "object",
    "properties": {
        "links": {"type": "array", "items": {"type": "string"}},
        "updates": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "retrieval_desc": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "retrieval_desc", "keywords", "tags"],
            "additionalProperties": False,
        }},
    },
    "required": ["links", "updates"],
    "additionalProperties": False,
}

# Exploratory "w/o rewrites" variant (results/longmemeval_s/runs/expl_lme_links_dev60): FLOW_LINKER without the
# neighbour rewrites, so the linker only chooses links.
FLOW_LINKER_LINKS = """You maintain a linked memory of an ongoing interaction. A new note has just been written. Decide which of the candidate earlier notes it should be linked to. Link two notes when recalling one makes the other useful: they concern the same entity, task, file or topic, or one continues, revises, corrects, contradicts or depends on the other. Do not link notes that are only generically similar.

New note {new_id} (time: {time}; position: {span})
short: {short}
keywords: {keywords}
tags: {tags}
retrieval_desc: {desc}

Candidate earlier notes:
{candidates}

Return a JSON object with "links" (the ids of candidates to link; may be empty)."""

FLOW_LINKER_LINKS_SCHEMA = {
    "type": "object",
    "properties": {"links": {"type": "array", "items": {"type": "string"}}},
    "required": ["links"],
    "additionalProperties": False,
}

FLOW_MERGE = """Below are consecutive records from an earlier part of an ongoing interaction, in chronological order. Rewrite them as a single record of at most {words} words. Keep the facts, decisions, outcomes and specific identifiers (names, numbers, dates, file paths, error messages) most likely to matter later, keep the order of events clear, and drop redundancy. Output only the record.

{records}"""

# ---------------------------------------------------------------- A-MEM ---------------------------
# Verbatim from A-mem-sys/agentic_memory/memory_system.py (analyze_content, lines 180-208).
AMEM_ANALYZE = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context":
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """

# The original schema is non-strict; strict structured output needs every field required.
AMEM_ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "context": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords", "context", "tags"],
    "additionalProperties": False,
}

# Verbatim from memory_system.py (_evolution_system_prompt, lines 131-161).
AMEM_EVOLUTION = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories (each line starts with memory_id):
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Use the memory_id from the neighbors above. Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["memory_id_1", "memory_id_2", ...],
                                    "tags_to_update": ["tag_1",..."tag_n"],
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''

# Verbatim from memory_system.py (process_memory, lines 658-705).
AMEM_EVOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "should_evolve": {"type": "boolean"},
        "actions": {"type": "array", "items": {"type": "string"}},
        "suggested_connections": {"type": "array", "items": {"type": "string"}},
        "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
        "tags_to_update": {"type": "array", "items": {"type": "string"}},
        "new_tags_neighborhood": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
    "required": ["should_evolve", "actions", "suggested_connections", "tags_to_update",
                 "new_context_neighborhood", "new_tags_neighborhood"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- Rolling summary -----------------
# Verbatim OpenHands LLMSummarizingCondenser prompt (OpenHands, MIT License), as used in MemGym (Xu et al., arXiv
# 2605.20833).
SUMMARIZATION_PROMPT = """You are maintaining a context-aware state summary for an interactive agent.
You will be given a list of events corresponding to actions taken by the agent, and the most recent previous summary if one exists.
If the events being summarized contain ANY task-tracking, you MUST include a TASK_TRACKING section to maintain continuity.
When referencing tasks make sure to preserve exact task IDs and statuses.

Track:

USER_CONTEXT: (Preserve essential user requirements, goals, and clarifications in concise form)

TASK_TRACKING: {Active tasks, their IDs and statuses - PRESERVE TASK IDs}

COMPLETED: (Tasks completed so far, with brief results)
PENDING: (Tasks that still need to be done)
CURRENT_STATE: (Current variables, data structures, or relevant state)

For code-specific tasks, also include:
CODE_STATE: {File paths, function signatures, data structures}
TESTS: {Failing cases, error messages, outputs}
CHANGES: {Code edits, variable updates}
DEPS: {Dependencies, imports, external calls}
VERSION_CONTROL_STATUS: {Repository state, current branch, PR status, commit history}

PRIORITIZE:
1. Adapt tracking format to match the actual task type
2. Capture key user requirements and goals
3. Distinguish between completed and pending tasks
4. Keep all sections concise and relevant

SKIP: Tracking irrelevant details for the current task type

Example formats:

For code tasks:
USER_CONTEXT: Fix FITS card float representation issue
COMPLETED: Modified mod_float() in card.py, all tests passing
PENDING: Create PR, update documentation
CODE_STATE: mod_float() in card.py updated
TESTS: test_format() passed
CHANGES: str(val) replaces f"{val:.16G}"
DEPS: None modified
VERSION_CONTROL_STATUS: Branch: fix-float-precision, Latest commit: a1b2c3d

For other tasks:
USER_CONTEXT: Write 20 haikus based on coin flip results
COMPLETED: 15 haikus written for results [T,H,T,H,T,H,T,T,H,T,H,T,H,T,H]
PENDING: 5 more haikus needed
CURRENT_STATE: Last flip: Heads, Haiku count: 15/20"""


# Deviation (docs/method/design.md): the same retention instruction as Flow's writer, so the summary is not
# disadvantaged on the specific details both benchmarks ask about (the verbatim prompt targets tasks).
SUMMARY_RETENTION = ("Keep the specific names, numbers, dates, identifiers, file paths and error messages most "
                     "likely to matter later.")


def summarization_call(previous_summary: str, events: list[str]) -> str:
    p = SUMMARIZATION_PROMPT + "\n\n"
    p += f"<PREVIOUS SUMMARY>\n{previous_summary}\n</PREVIOUS SUMMARY>\n"
    p += "\n\n"
    for i, e in enumerate(events):
        p += f"<EVENT id={i}>\n{e}\n</EVENT>\n"
    return p + SUMMARY_RETENTION + "\nNow summarize the events using the rules above."


# ---------------------------------------------------------------- ReadAgent -----------------------
# Verbatim from ReadAgent (Lee et al., ICML 2024), method section; "article"/"page" wording kept
# for the gist prompt, adapted to interaction units for lookup.
READAGENT_GIST = "Please shorten the following passage.\n\nJust give me a shortened version. DO NOT explain your reason.\n\nPassage: \n\n{text}"

READAGENT_LOOKUP = """The following text is what you remember from reading an earlier interaction log and a {goal} related to it.

You may read 1 to {max_pages} page(s) of the log again to refresh your memory to prepare yourself for the {goal_short}.

Please respond with which page(s) you would like to read.

For example, if you only need to read Page 8, respond with "I want to look up Page [8] to ..."; if you would like to read Page 7 and 12, respond with "I want to look up Page [7, 12] to ..."; if you would like to read Page 2, 3, 7, 15 and 18, respond with "I want to look up Page [2, 3, 7, 15, 18] to ...".

DO NOT select more pages if you don't need to.

You don't need to {no_answer_yet} yet.

Text:

{gists}

{goal_label}:
{query}

Take a deep breath and tell me: Which page(s) would you like to read again?"""


# ---------------------------------------------------------------- MemGAS --------------------------
# Verbatim from MemGAS (Xu et al., ICLR 2026), Appendix I, Figures 8 (generation) and 9 (filter). On SWE the
# memory is named an interaction log instead of a dialogue and the question is the task with the latest
# progress (docs/method/design.md, deviation 26).
MEMGAS_NOUNS = {"lme": ("an user-AI assistant dialogue memory", "dialogue"),
                "swe": ("an AI coding agent's interaction log", "log")}
MEMGAS_SUMMARY = ("Below is {memory}. Please summarize the following {noun} as concisely as possible in a short "
                  "paragraph, extracting the main themes and key information.\n\n{text}")
MEMGAS_KEYWORDS = "Below is {memory}. Please extract the most relevant keywords, separated by semicolon.\n\n{text}"
MEMGAS_FILTER = {
    "lme": ("You are an intelligent dialog bot. You will be shown History Dialogs and corresponding multi-granular "
            "information. Filter the History Dialogs, summaries, and keywords to extract only the parts directly "
            "relevant to the Question. Preserve original tokens, do not paraphrase. Remove irrelevant turns, redundant "
            "info, and non-essential details.\nHistory Dialogs: {retrieved_texts}\nQuestion Date: {question_date}\n"
            "Question: {question}\nAnswer:"),
    "swe": ("You are an intelligent coding assistant. You will be shown an Earlier Interaction Log of an AI coding "
            "agent and corresponding multi-granular information. Filter the Earlier Interaction Log, summaries, and "
            "keywords to extract only the parts directly relevant to the next step of the Task. Preserve original "
            "tokens, do not paraphrase. Remove irrelevant steps, redundant info, and non-essential details.\n"
            "Earlier Interaction Log: {retrieved_texts}\nTask and latest progress: {question}\nAnswer:"),
}


# ---------------------------------------------------------------- Sculptor ------------------------
# Sculptor (Li et al., ICLR 2026): the unified system prompt (Appendix C.2, Figure 9) and the tool schemas (Appendix F,
# Figures 11-15), verbatim. fragment_context is not offered: the fragments are the shared units. The paper gives no
# summarizer prompt; SCULPTOR_SUMMARIZE is built from the summarize_fragment description, and a summary nobody asked
# for (the forced fallback, the LongMemEval build) takes "main points", one of the schema's own focus examples
# (docs/method/design.md, deviation 27).
SCULPTOR_SYSTEM = ("You are a helpful assistant. You can autonomously manage your own context: fold irrelevant "
                   "information, focus on useful details, summarize long texts to keep your context concise, and use "
                   "search tools to find key information in large documents.")
SCULPTOR_MANAGE = SCULPTOR_SYSTEM.split(". ", 1)[1]  # the second sentence, for the SWE memory message
SCULPTOR_SUMMARIZE = ("Summarize the following conversation fragment to compress its content while preserving key "
                      "information. Focus: {focus}.\n\n{text}")
SCULPTOR_FOCUS = "main points"


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


_FID = lambda what: {"type": "string", "description": f"ID of the fragment to {what} (e.g., 'f1a2b3')"}  # noqa: E731
SCULPTOR_TOOLS = [
    _tool("summarize_fragment", "Summarize a conversation fragment using LLM to compress content while preserving key "
          "information. Supports focus-oriented summarization.",
          {"fragment_id": _FID("summarize"),
           "focus": {"type": "string", "description": "Focus area for the summary (e.g., 'technical details', 'key "
                     "decisions', 'action items', 'main points', 'problems', 'solutions')"}},
          ["fragment_id", "focus"]),
    _tool("fold_fragment", "Fold (hide) a conversation fragment to reduce visible context length. The content is "
          "preserved and can be expanded later.", {"fragment_id": _FID("fold")}, ["fragment_id"]),
    _tool("restore_fragment", "Restore a fragment to its original content from ACM storage. Works for both summarized "
          "and folded fragments.", {"fragment_id": _FID("restore")}, ["fragment_id"]),
    _tool("search_context", "Search tool for finding exact text matches in conversation history.",
          {"query": {"type": "string", "description": "Exact text to search for in conversation history"},
           "role": {"type": "string", "enum": ["user", "assistant", "all"], "default": "user",
                    "description": "Filter by message role (default: user)"},
           "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50,
                           "description": "Maximum number of results to return"},
           "context_size": {"type": "integer", "default": 200, "minimum": 50, "maximum": 1000,
                            "description": "Context characters before/after match"}},
          ["query"]),
    _tool("get_search_detail", "Get detailed context for a search result by its ID. Retrieves extended context around "
          "the search match position.",
          {"search_id": {"type": "string", "description": "Search result ID from search_context (e.g., 's1a2b3')"},
           "extended_context": {"type": "integer", "default": 500, "minimum": 100, "maximum": 2000,
                                "description": "Number of characters to show before and after the match "
                                               "(default: 500)"}},
          ["search_id"]),
]


# ---------------------------------------------------------------- AFM ------------------------------
# Adaptive Focus Memory (Cruz 2025, arXiv:2511.12712), reproduced from the paper (decision 2026-10-11_locomo_afm). The
# paper describes the classifier (one call per message; CRITICAL, RELEVANT or TRIVIAL; "prompted to treat
# safety-sensitive facts (e.g., medical conditions) as CRITICAL") and the compressor (an LLM rewrite under a stated token
# budget) but prints neither prompt; these follow those descriptions and add nothing to them. The reference code was
# not consulted.
AFM_IMPORTANCE = """Classify the importance of the following past message for the rest of this conversation.
- CRITICAL: a safety-sensitive fact, such as a medical condition.
- RELEVANT: information relevant to the conversation.
- TRIVIAL: everything else.

Message:
{text}"""
AFM_IMPORTANCE_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["CRITICAL", "RELEVANT", "TRIVIAL"]}},
    "required": ["label"],
    "additionalProperties": False,
}
AFM_COMPRESS = """Rewrite the following message in at most {tokens} tokens. Keep the facts, constraints, names, numbers and identifiers most likely to matter later. Output only the rewritten message.

Message:
{text}"""
