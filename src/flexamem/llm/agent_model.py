"""mini-swe-agent model that routes through the configured OpenAI-compatible endpoint and records every agent API
response in the ledger (prompt tokens as billed by the API, used to cross-check our tiktoken count).
Recording happens per HTTP response in _query, so responses that end in a FormatError are counted too.
A context policy may offer tools of its own next to bash (Sculptor); their calls become actions
{"tool", "args", "tool_call_id"} that the agent runs in the policy instead of the environment. On AppWorld the action
tool is `python` (argument `code`) instead of bash; its calls become the same {"command", "tool_call_id"} actions."""
import json
import time

import litellm
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions

from flexamem.llm.client import REJECT_SENDS, REJECT_WAIT_S, LLMTransientError, get_llm
from flexamem.llm.tokens import billed_usage
from flexamem.llm.ledger import get_ledger
from flexamem.memory.segmenter import ACTION_TOOLS
from flexamem.methods.sculptor import parse_args


def make_agent_model(observation_template: str, format_error_template: str,
                     action_tool: str = "bash") -> "LedgerLitellmModel":
    llm = get_llm()
    spec = llm.role("agent")
    kwargs = {"api_base": llm.cfg["base_url"], "api_key": llm.api_key, "drop_params": True,
              "parallel_tool_calls": True}
    if spec.get("reasoning_effort"):
        kwargs["reasoning_effort"] = spec["reasoning_effort"]
    m = LedgerLitellmModel(model_name=f"openai/{spec['model']}", model_kwargs=kwargs,
                           cost_tracking="ignore_errors", observation_template=observation_template,
                           format_error_template=format_error_template)
    m.action_tool = action_tool
    return m


def parse_action_calls(tool_calls: list, tool: str, arg: str, format_error_template: str) -> list[dict]:
    """parse_toolcall_actions for an action tool other than bash: every call must be `tool` with argument `arg`, else
    FormatError with the same messages."""
    def error(msg):
        return FormatError({"role": "user", "content": Template(format_error_template, undefined=StrictUndefined)
                            .render(actions=[], error=msg), "extra": {"interrupt_type": "FormatError"}})
    if not tool_calls:
        raise error("No tool calls found in the response. Every response MUST include at least one tool call.")
    actions = []
    for tc in tool_calls:
        msg, args = "", {}
        try:
            args = json.loads(tc.function.arguments)
        except Exception as e:
            msg = f"Error parsing tool call arguments: {e}."
        if tc.function.name != tool:
            msg += f"Unknown tool '{tc.function.name}'."
        if not isinstance(args, dict) or not isinstance(args.get(arg), str):  # a list or number is the model's mistake
            msg += f"Missing '{arg}' argument in {tool} tool call."
        if msg:
            raise error(msg.strip())
        actions.append({"command": args[arg], "tool_call_id": tc.id})
    return actions


class LedgerLitellmModel(LitellmModel):
    last_usage: dict = {}
    # _query applies the rule of llm/client.py to rejected prompts (400), so mini-swe-agent's own 10 retries
    # (minisweagent/models/utils/retry.py) must not resend them. A final 400 ends the episode as a method error
    # (unresolved).
    abort_exceptions = [*LitellmModel.abort_exceptions, litellm.exceptions.BadRequestError, LLMTransientError]
    _rejected = 0
    tools: list = [BASH_TOOL]
    action_tool = "bash"

    def query(self, messages, tools=None, **kwargs):
        # Other errors (429, 5xx) still go to mini-swe-agent's retries, which call _query again; the count is per
        # prompt, so it is reset here and not in _query. One model instance serves one episode.
        self._rejected = 0
        self.tools = tools or [BASH_TOOL]
        return super().query(messages, **kwargs)

    def _query(self, messages, **kwargs):
        while True:
            t0 = time.time()
            try:  # upstream _query with this call's tools (upstream always sends [BASH_TOOL])
                r = litellm.completion(model=self.config.model_name, messages=messages, tools=self.tools,
                                       **(self.config.model_kwargs | kwargs))
                break
            except litellm.exceptions.ContextWindowExceededError:
                raise
            except litellm.exceptions.BadRequestError:
                self._rejected += 1
                if self._rejected >= REJECT_SENDS:
                    raise
                time.sleep(REJECT_WAIT_S)
        u = r.model_dump().get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        cdetails = u.get("completion_tokens_details") or {}
        get_ledger().record(phase="agent", model=self.config.model_name, latency_s=round(time.time() - t0, 3),
                            prompt_tokens=billed_usage(u)[0], completion_tokens=billed_usage(u)[1],
                            cached_tokens=details.get("cached_tokens", 0) or 0,
                            reasoning_tokens=cdetails.get("reasoning_tokens", 0) or 0, ok=True, cache_hit=False, usage_v=2)
        self.last_usage = u
        return r

    def _parse_actions(self, response) -> list[dict]:
        calls = response.choices[0].message.tool_calls or []
        if self.action_tool != "bash":  # AppWorld: python actions, plus Sculptor's context tools when offered
            own = {t["function"]["name"] for t in self.tools} - {self.action_tool}
            acts = [c for c in calls if c.function.name not in own]
            parsed = iter(parse_action_calls(acts, self.action_tool, ACTION_TOOLS[self.action_tool],
                                             self.config.format_error_template) if acts or not calls else [])
            return [{"tool": c.function.name, "args": parse_args(c.function.arguments), "tool_call_id": c.id}
                    if c.function.name in own else next(parsed) for c in calls]
        own = {t["function"]["name"] for t in self.tools} - {"bash"}
        if not any(c.function.name in own for c in calls):
            return super()._parse_actions(response)  # bash only, or an unknown tool: upstream's FormatError
        bash = iter(parse_toolcall_actions([c for c in calls if c.function.name not in own],
                                           format_error_template=self.config.format_error_template)
                    if any(c.function.name not in own for c in calls) else [])
        return [{"tool": c.function.name, "args": parse_args(c.function.arguments), "tool_call_id": c.id}
                if c.function.name in own else next(bash) for c in calls]

    def serialize(self) -> dict:
        d = super().serialize()
        d["info"]["config"]["model"]["model_kwargs"].pop("api_key", None)  # never write the key to trajectories
        return d
