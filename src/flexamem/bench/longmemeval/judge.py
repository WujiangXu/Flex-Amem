"""LongMemEval answer checking (Wu et al., ICLR 2025). Instruction texts are transcribed from the
paper's arXiv source (figures/prompt_for_evaluation.tex); the Question/Answer/Response suffix and
the abstention template follow the official evaluate_qa.py. The paper's runs used two judges from model
families other than the reader's (configs/models/default.yaml: judge, judge2), instead of GPT-4o, and report
their agreement."""
from flexamem.llm.client import get_llm

_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
         "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
         "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
         "the response only contains a subset of the information required by the answer, answer no.")
_TEMPORAL = _BASE + (" In addition, do not penalize off-by-one errors for the number of days. If the question asks "
                     "for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
                     "predicting 19 days when the answer is 18), the model's response is still correct.")
_UPDATE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
           "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
           "information along with an updated answer, the response should be considered as correct as long as the "
           "updated answer is the required answer.")
_PREF = ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
         "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not "
         "need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes "
         "the user's personal information correctly.")
_SUFFIX = "\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only."
_PREF_SUFFIX = "\n\nQuestion: {q}\n\nRubric: {a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only."
_ABS = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if "
        "the model correctly identifies the question as unanswerable. The model could say that the information is "
        "incomplete, or some other information is given but the asked information is not.\n\nQuestion: {q}\n\n"
        "Explanation: {a}\n\nModel Response: {r}\n\nDoes the model correctly identify the question as unanswerable? "
        "Answer yes or no only.")


def judge_prompt(qtype: str, question: str, answer: str, response: str, abstention: bool) -> str:
    if abstention:
        return _ABS.format(q=question, a=answer, r=response)
    if qtype == "single-session-preference":
        return _PREF + _PREF_SUFFIX.format(q=question, a=answer, r=response)
    head = {"temporal-reasoning": _TEMPORAL, "knowledge-update": _UPDATE}.get(qtype, _BASE)
    return head + _SUFFIX.format(q=question, a=answer, r=response)


def judge(role: str, qtype: str, question: str, answer: str, response: str, abstention: bool, **meta) -> dict:
    out = get_llm().call(role, judge_prompt(qtype, question, answer, response, abstention), phase="judge",
                         cache=True, **meta)
    return {"label": verdict(out), "raw": out}


def verdict(raw: str) -> bool:
    return "yes" in raw.lower()
