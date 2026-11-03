"""
Module 3, Concept 3. Tool-selection strategies.

Offering every tool on every call does not scale: each schema costs tokens, and accuracy
drops as near-duplicate tools pile up. Selection is also a safety control, because a tool the
model was never offered cannot be called.

Five strategies, each as middleware around the model call:
  A. Static        every tool, every time (the baseline, with its token cost)
  B. By role       filter on runtime context: contractors get read-only tools
  C. By state      staged unlocking: mutating tools appear only after a diagnostic has run
  D. By relevance  LLMToolSelectorMiddleware: a cheap model call picks the top-k tools first
  E. Forced        tool_choice pins the first step to a knowledge-base lookup

All of B to E work by replacing the tool list on the request: handler(request.override(tools=...)).

Prompts to try
  - payments-api on app-01 is down, please restart it.     (compare B as contractor and as admin)
  - Why is /var full on app-01?                            (C: watch the tool list grow after the first check)
  - Reset Asha's password and tell me her last login.      (D: identity tools should be selected, not storage)
"""
from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware, ModelCallLimitMiddleware, wrap_model_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from lab_common import banner, llm, run


def make_tool(name, description, reply, mutating=False):
    """Build a simulated single-argument tool. `mutating` is kept in metadata for the filters below."""
    def impl(target: str) -> str:
        return reply.format(target=target)
    impl.__name__ = name
    impl.__doc__ = ("MUTATING. " if mutating else "Read-only. ") + description + " `target` is a hostname or a user id."
    built = tool(impl)
    built.metadata = {"mutating": mutating, "domain": name.split("_")[0]}
    return built


CATALOG = [
    make_tool("kb_search", "Search the runbook knowledge base for a symptom.", "Runbook RB-12: if /var is full, rotate /var/log/payments, then restart payments-api. ({target})"),
    make_tool("service_status", "Return the state of the payments-api service on a host.", "payments-api on {target}: failed"),
    make_tool("service_logs", "Return recent payments-api log lines from a host.", "{target}: OSError [Errno 28] No space left on device"),
    make_tool("service_restart", "Restart payments-api on a host.", "payments-api on {target} restarted", mutating=True),
    make_tool("storage_usage", "Return disk usage per mount on a host.", "{target}: /var 100% used, / 46% used"),
    make_tool("storage_rotate_logs", "Archive old logs under /var/log on a host.", "{target}: 37G freed on /var", mutating=True),
    make_tool("storage_expand_volume", "Add 50 GB to the data volume of a host.", "{target}: volume expanded", mutating=True),
    make_tool("network_ping", "Check that a host answers on the network.", "{target}: reachable, 2 ms"),
    make_tool("network_open_port", "Open a firewall port on a host.", "{target}: port opened", mutating=True),
    make_tool("identity_last_login", "Return the last login time of a user.", "{target}: last login today 08:55"),
    make_tool("identity_reset_password", "Reset the password of a user and send them a reset link.", "{target}: reset link sent", mutating=True),
    make_tool("identity_disable_account", "Disable a user account.", "{target}: account disabled", mutating=True),
]

PROMPT = """You are an operations agent. Work only with the tools you are offered on this step; the set can change
between steps. If the tool you need is not offered, say what you would need and why, and stop there."""


@dataclass
class Caller:
    user_id: str
    role: Literal["admin", "contractor"]


@wrap_model_call
def log_offered(request, handler):
    """Print the tools the model is actually offered. Listed last, so it sees the other filters' result."""
    print(f"  [offered] {[t.name for t in request.tools if hasattr(t, 'name')]}")
    return handler(request)


# ---- B. Filter by role
@wrap_model_call
def tools_by_role(request, handler):
    if request.runtime.context.role != "admin":
        request = request.override(tools=[t for t in request.tools if not t.metadata["mutating"]])
    return handler(request)


# ---- C. Filter by state: no mutating tools until a read-only check has returned a result
@wrap_model_call
def unlock_after_diagnosis(request, handler):
    checked = any(isinstance(m, ToolMessage) for m in request.messages)
    if not checked:
        request = request.override(tools=[t for t in request.tools if not t.metadata["mutating"]])
    return handler(request)


# ---- E. Force the first step
@wrap_model_call
def kb_first(request, handler):
    if not any(isinstance(m, ToolMessage) for m in request.messages):
        request = request.override(tools=[t for t in request.tools if t.name == "kb_search"], tool_choice="kb_search")
    return handler(request)


def build(*middleware):
    return create_agent(model=llm, tools=CATALOG, system_prompt=PROMPT, context_schema=Caller,
                        middleware=[ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"), *middleware, log_offered])


def main():
    admin, contractor = Caller("ravi", "admin"), Caller("meera", "contractor")
    outage = "payments-api on app-01 is down, please restart it."

    banner("A. Static: the cost of offering everything")
    tokens = sum(count_tokens_approximately([str(convert_to_openai_tool(t))]) for t in CATALOG)
    print(f"{len(CATALOG)} tools cost roughly {tokens} tokens on every model call, before any conversation.")
    run(build(), outage, context=admin)

    banner("B. By role: contractor (read-only tools)")
    run(build(tools_by_role), outage, context=contractor)
    banner("B. By role: admin (all tools)")
    run(build(tools_by_role), outage, context=admin)

    banner("C. By state: mutating tools unlock after the first diagnostic result")
    run(build(unlock_after_diagnosis), "Why is /var full on app-01? Fix it if you can.", context=admin)

    banner("D. By relevance: a selector model picks at most 3 tools per request")
    selector = LLMToolSelectorMiddleware(model=llm, max_tools=3, always_include=["kb_search"])
    run(build(selector), "Reset Asha's password and tell me her last login.", context=admin)

    banner("E. Forced first step: knowledge base before anything else")
    run(build(kb_first), outage, context=admin)

    print("\nDiscuss: B and C are deterministic and auditable. D adds a model call and can drop the tool\n"
          "you needed. Which of your own tools should never depend on a model deciding to offer them?")


if __name__ == "__main__":
    main()
