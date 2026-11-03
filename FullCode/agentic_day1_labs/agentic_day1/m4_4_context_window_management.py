"""
Module 4, Concept 4. Context window management.

A long-running agent fills its context with tool output long before it runs out of things to
do. Past a point the model gets slower, more expensive and less accurate, then fails outright.
Managing the window means deciding, on every model call, what the model does not need to see.

Four techniques on the same log-heavy task, each measured in tokens per model call:
  0. Baseline       nothing managed; the context only grows
  1. Trim           keep the most recent messages (cheap, forgets everything older)
  2. Summarise      replace old history with a model-written summary (SummarizationMiddleware)
  3. Clear results  blank out old tool outputs but keep the call record (ContextEditingMiddleware)
  4. Offload        keep big outputs out of the context entirely: store them, return a short
                    preview and a handle, and give the agent a tool to read slices on demand

Techniques 1 and 2 rewrite stored state. 3 edits only what is sent to the model. 4 prevents
the problem at the source and is the one to reach for first with large tool results.

Prompts to try
  - Compare error patterns across payments-api, nginx, redis and auth-service on app-01 and tell me
    which service failed first.
"""
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools import tool
from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from lab_common import banner, llm

FIRST_ERROR = {"payments-api": "09:42:03", "nginx": "09:42:05", "redis": None, "auth-service": "09:47:30"}
ARTIFACTS = {}       # offloaded tool outputs, by handle
USAGE = []           # tokens sent on each model call of the current run


def fake_log(service):
    lines = [f"09:{m:02d}:{s:02d} INFO {service} request id={m * 60 + s} status=200 latency={20 + (m * s) % 40}ms"
             for m in range(38, 48) for s in range(0, 60, 12)]
    if FIRST_ERROR[service]:
        lines.append(f"{FIRST_ERROR[service]} ERROR {service} first failure: upstream unavailable")
    return "\n".join(sorted(lines))


@tool
def fetch_logs(service: str) -> str:
    """Return today's log for a service on app-01: payments-api, nginx, redis or auth-service."""
    return fake_log(service) if service in FIRST_ERROR else f"ERROR: unknown service '{service}'"


@tool
def read_artifact(handle: str, contains: str = "ERROR") -> str:
    """Return the lines of a stored tool output that contain a given text. Use the handle from the preview."""
    text = ARTIFACTS.get(handle)
    if text is None:
        return f"ERROR: unknown handle '{handle}'"
    return "\n".join(line for line in text.splitlines() if contains in line)[:2000] or f"no lines contain '{contains}'"


PROMPT = """You are a log analyst. Fetch the log of each service you are asked about, one at a time, and note the
time of its first ERROR line before moving on to the next. Finish with the order in which the services failed.
If a tool result is a preview with a handle, use read_artifact to look inside it instead of fetching again."""

TASK = ("Compare error patterns across payments-api, nginx, redis and auth-service on app-01 and tell me which "
        "service failed first.")


@wrap_model_call
def meter(request, handler):
    """Record the size of what is actually sent to the model. Listed last so it sees the other edits."""
    USAGE.append(count_tokens_approximately(request.messages))
    return handler(request)


# ---- 1. Trim: always keep the task, then as much recent history as fits in about 1,500 tokens.
#         The kept slice must start on a model turn so that no tool result is left without its call.
@before_model
def trim_history(state, runtime):
    task, rest = state["messages"][:1], state["messages"][1:]
    recent = trim_messages(rest, strategy="last", token_counter=count_tokens_approximately,
                           max_tokens=1500, start_on="ai", allow_partial=False)
    if len(recent) == len(rest):
        return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *task, *recent]}


# ---- 2. Summarise: when history passes 2,000 tokens, keep the last 4 messages and summarise the rest
summarise = SummarizationMiddleware(model=llm, trigger=("tokens", 2000), keep=("messages", 4))

# ---- 3. Clear old tool results: past 1,500 tokens, blank all but the most recent tool output
clear_results = ContextEditingMiddleware(edits=[
    ClearToolUsesEdit(trigger=1500, keep=1, placeholder="[log cleared from context; first-error time was noted above]")
])


# ---- 4. Offload: large results never enter the context
@wrap_tool_call
def offload_large_results(request, handler):
    result = handler(request)
    if isinstance(result, ToolMessage) and len(str(result.content)) > 1000:
        handle = f"artifact-{len(ARTIFACTS) + 1}"
        ARTIFACTS[handle] = str(result.content)
        lines = ARTIFACTS[handle].splitlines()
        preview = "\n".join(lines[:3] + ["..."] + lines[-2:])
        result = ToolMessage(
            content=f"Stored {len(lines)} lines as handle '{handle}'. Preview:\n{preview}\n"
                    f"Call read_artifact('{handle}', contains=...) to search inside it.",
            tool_call_id=result.tool_call_id, name=result.name)
    return result


VARIANTS = {
    "0. baseline": [],
    "1. trim": [trim_history],
    "2. summarise": [summarise],
    "3. clear old tool results": [clear_results],
    "4. offload large results": [offload_large_results],
}


def main():
    results = {}
    for name, middleware in VARIANTS.items():
        banner(name)
        USAGE.clear()
        ARTIFACTS.clear()
        agent = create_agent(
            model=llm, tools=[fetch_logs, read_artifact], system_prompt=PROMPT,
            middleware=[ModelCallLimitMiddleware(run_limit=14, exit_behavior="end"), *middleware, meter],
        )
        state = agent.invoke({"messages": [{"role": "user", "content": TASK}]})
        answer = state["messages"][-1].text
        results[name] = (list(USAGE), answer)
        print("tokens per model call:", USAGE)
        print("answer:", " ".join(answer.split())[:300])

    banner("Summary")
    print(f"{'technique':28}{'calls':>6}{'peak':>8}{'total':>9}")
    for name, (usage, _) in results.items():
        print(f"{name:28}{len(usage):>6}{max(usage):>8}{sum(usage):>9}")
    print("\nCorrect order: payments-api (09:42:03), nginx (09:42:05), auth-service (09:47:30); redis never failed.\n"
          "Check each answer against it. A technique that saves tokens by losing the answer has not saved anything.\n"
          "If a run ended on the model-call limit, the agent forgot which logs it had already read and\n"
          "fetched them again. That is the failure mode of trimming, and the reason the limit is there.")


if __name__ == "__main__":
    main()
