"""LoCoMo's reader and judge prompts, in one place so they can be swapped. The set below is Mem0's; LongMemEval's
(bench/longmemeval/reader.py, bench/longmemeval/judge.py) are kept as LME_READER and LME_HISTORY_LEAD.

READER: the reader template. Fields available to str.format: {chat_history} (the memory text, also as {memories} and
{context}), {question}, {question_date} (the conversation's last session date), {speaker_a}, {speaker_b}. The memory
allowance is computed from the template actually set (bench/locomo/reader.py), so any template keeps the budget exact.
HISTORY_LEAD goes before the memory text.

JUDGE: the judge template, with {question}, {gold_answer} and {generated_answer} (also {answer} and {response}).
JUDGE_BY_CATEGORY: per-category overrides (default: LongMemEval's temporal-reasoning prompt for category 2, which
forgives off-by-one day counts). verdict(raw): the judge's reply -> correct or not (default: "yes" in it).

Templates go through str.format, so literal braces in a pasted prompt (a JSON example) must be doubled.

Set (2026-10-11, from the prompts of recent high-scoring LoCoMo papers): Mem0's prompts (Chhikara et al.,
arXiv 2504.19413, Appendix A), the paper that defined the LLM-as-a-Judge score J used by most LoCoMo results.
- READER: Mem0's results-generation prompt, instructions verbatim (PDF line breaks rejoined), with the single memory slot
  that TiMem (arXiv 2601.02845, ACL 2026 Findings, App. E.1.1) prints for it; TiMem, StructMem (ACL 2026) and Nemori
  reprint the same instructions word for word, and Memora (ICML 2026) reports using Mem0's answer prompt.
- JUDGE: Mem0's "Prompt Template for LLM as a Judge", verbatim, for every category (it covers time questions itself);
  verdict() reads its JSON label."""
import re

from flexamem.bench.longmemeval import judge as lme_judge
from flexamem.bench.longmemeval import reader as lme_reader

LME_READER, LME_HISTORY_LEAD = lme_reader.READ_PROMPT, lme_reader.HISTORY_LEAD  # the defaults before Mem0's
READER = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Relevant Memories:
{memories}

Question: {question}

Answer:"""
HISTORY_LEAD = ""

_SUFFIX = ("\n\nQuestion: {question}\n\nCorrect Answer: {gold_answer}\n\nModel Response: {generated_answer}\n\n"
           "Is the model response correct? Answer yes or no only.")
assert _SUFFIX.format(question="{q}", gold_answer="{a}", generated_answer="{r}") == lme_judge._SUFFIX  # verbatim
LME_JUDGE, LME_JUDGE_BY_CATEGORY = lme_judge._BASE + _SUFFIX, {2: lme_judge._TEMPORAL + _SUFFIX}  # defaults before
JUDGE = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""
JUDGE_BY_CATEGORY: dict = {}


def reader_prompt(q: dict, history: str) -> str:
    h = HISTORY_LEAD + history
    return READER.format(chat_history=h, memories=h, context=h, question=q["question"],
                         question_date=q["question_date"], speaker_a=q["speaker_a"], speaker_b=q["speaker_b"])


def judge_prompt(category: int, question: str, gold: str, response: str) -> str:
    return JUDGE_BY_CATEGORY.get(category, JUDGE).format(question=question, gold_answer=gold, generated_answer=response,
                                                         answer=gold, response=response)


def verdict(raw: str) -> bool:
    """Mem0's judge answers with a JSON label (after a one-sentence reason): the last CORRECT or WRONG in the reply."""
    hits = re.findall(r"\b(CORRECT|WRONG)\b", raw.upper())
    return bool(hits) and hits[-1] == "CORRECT"  # no label: wrong (the raw reply is kept beside the label)
