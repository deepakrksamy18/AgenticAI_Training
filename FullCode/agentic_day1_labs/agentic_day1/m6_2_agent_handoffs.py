"""
Module 6, Concept 2. Agent-to-agent communication by handoff.

With a supervisor, every message goes through the centre. With handoffs, agents are peers:
the agent that is talking to the user can pass the conversation to a colleague, who then
talks to the user directly. It suits conversations that move between specialities.

What a handoff is, mechanically
  - A tool, transfer_to_<agent>, that the current agent can call.
  - The tool returns Command(goto=<agent>, graph=Command.PARENT, update=...): leave this
    agent, go to another node in the parent graph, and carry the conversation across.
  - The tool takes a `briefing` argument. That text is the protocol between the agents: what
    the user wants, what has been established, what is still open. A handoff with no briefing
    makes the next agent start again.

Also shown
  - Sticky routing: `active_agent` is kept in state, so the user's next message goes straight
    to whoever holds the conversation, with no trip back through the front desk.
  - Identity in context: the access rule is enforced in the tool, whichever agent calls it.

Conversation to try (one thread)
  ASHA  Hi, checkout seems broken, I think payments-api on app-01 is down.
  ASHA  OK. Can you restart it?                 (ops holds the conversation; a developer is denied)
  ASHA  Then please get me the permission.      (ops should hand off to access with a briefing)
"""
from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command

from lab_common import banner, llm, show_message

FLEET = {"app-01": {"payments-api": "failed"}}
TICKETS = []
AGENTS = ("front_desk", "ops", "access")


@dataclass
class Caller:
    user_id: str
    role: Literal["admin", "developer"]


class DeskState(MessagesState):
    active_agent: str


def make_handoff(target: str, when: str):
    @tool(f"transfer_to_{target}")
    def handoff(briefing: str, runtime: ToolRuntime[Caller]) -> Command:
        """placeholder"""
        note = ToolMessage(content=f"Conversation transferred to {target}. Briefing: {briefing}",
                           name=f"transfer_to_{target}", tool_call_id=runtime.tool_call_id)
        return Command(
            goto=target,
            graph=Command.PARENT,                                    # route in the team graph, not inside this agent
            update={"messages": [*runtime.state["messages"], note],  # carry the conversation across
                    "active_agent": target},
        )
    handoff.description = (f"Hand the conversation to the {target} agent. Use it when {when} `briefing` must say what "
                           "the user wants, what you have already established, and what is still open.")
    return handoff


@tool
def get_service_status(host: str, service: str) -> str:
    """Return the current state of a service on a host."""
    return f"{service} on {host}: {FLEET.get(host, {}).get(service, 'unknown')}"


@tool
def restart_service(host: str, service: str, runtime: ToolRuntime[Caller]) -> str:
    """Restart a service on a host. Changes the system. Administrators only."""
    if runtime.context.role != "admin":
        return f"ERROR[PERMISSION_DENIED]: role '{runtime.context.role}' may not restart services."
    FLEET[host][service] = "active"
    return f"{service} on {host} restarted"


@tool
def open_access_request(what: str, justification: str, runtime: ToolRuntime[Caller]) -> str:
    """Open a ticket asking an administrator to grant a permission or carry out a privileged action."""
    TICKETS.append({"id": f"REQ-{1001 + len(TICKETS)}", "by": runtime.context.user_id, "what": what, "why": justification})
    return f"opened {TICKETS[-1]['id']}"


SHARED = """

Working with colleagues
- You are one of three agents: front_desk, ops and access. The user sees one conversation and should not
  have to repeat themselves. Read the whole conversation, including any transfer briefing, before you reply.
- Transfer only when the request is outside your own tools. Call the transfer tool on its own, with no other
  tool call in the same turn, and do not also answer the user in that turn.
- After a transfer to you, carry on from the briefing. Do not greet the user again or ask for what it says.
"""

front_desk = create_agent(
    model=llm, name="front_desk",
    tools=[make_handoff("ops", "the user reports a fault or asks about the state of a service."),
           make_handoff("access", "the user asks for a permission or a privileged action they cannot perform.")],
    system_prompt="You are the front desk of an IT service team. Find out what the user needs, in one question at "
                  "most, and pass them to the right colleague. You have no tools of your own for fixing anything." + SHARED,
    middleware=[ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")],
)

ops = create_agent(
    model=llm, name="ops", context_schema=Caller,
    tools=[get_service_status, restart_service,
           make_handoff("access", "a tool returned PERMISSION_DENIED and the user wants the permission or wants an "
                                  "administrator to act."),
           make_handoff("front_desk", "the user changes to a subject that is neither a service fault nor access.")],
    system_prompt="You are the operations agent. Check a service before you change it and again afterwards. If a "
                  "tool returns ERROR[PERMISSION_DENIED], stop: tell the user what was denied and offer to pass "
                  "them to the access agent. Do not look for another way to make the change." + SHARED,
    middleware=[ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")],
)

access = create_agent(
    model=llm, name="access", context_schema=Caller,
    tools=[open_access_request,
           make_handoff("ops", "the user reports a new fault or asks about the state of a service.")],
    system_prompt="You are the access agent. You open access requests for administrators to review. Take the "
                  "justification from the conversation if it is there; ask only if it is missing. Open one request, "
                  "then give the user the ticket id." + SHARED,
    middleware=[ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")],
)

builder = StateGraph(DeskState, context_schema=Caller)
builder.add_node("front_desk", front_desk, destinations=("ops", "access"))
builder.add_node("ops", ops, destinations=("access", "front_desk"))
builder.add_node("access", access, destinations=("ops",))
# Sticky routing: each new user message goes to whoever currently holds the conversation.
builder.add_conditional_edges(START, lambda state: state.get("active_agent") or "front_desk", list(AGENTS))
desk = builder.compile(checkpointer=InMemorySaver())


def say(text, caller, config):
    print(f"\n  [user ] {text}")
    seen = len(desk.get_state(config).values.get("messages", [])) + 1
    for chunk in desk.stream({"messages": [{"role": "user", "content": text}]}, config, context=caller,
                             stream_mode="updates", subgraphs=True):
        namespace, update = chunk
        if not namespace:
            continue                                   # print agents' own steps, not the parent graph's echo
        agent = namespace[0].split(":")[0]
        for node_update in update.values():
            for msg in (node_update or {}).get("messages", []) if isinstance(node_update, dict) else []:
                if hasattr(msg, "type") and msg.type in ("ai", "tool"):
                    print(f"  <{agent}>", end="")
                    show_message(msg)
    state = desk.get_state(config).values
    print(f"  (conversation is now held by: {state.get('active_agent', 'front_desk')}; "
          f"{len(state['messages']) - seen} new messages)")


def main():
    banner("Handoffs between peer agents, one conversation")
    config, asha = {"configurable": {"thread_id": "desk-1"}}, Caller("asha", "developer")
    say("Hi, checkout seems broken, I think payments-api on app-01 is down.", asha, config)
    say("OK. Can you restart it?", asha, config)
    say("Then please get me the permission, the release is tonight.", asha, config)
    print("\nFleet:", FLEET, "| tickets:", TICKETS)
    print("\nDiscuss: read each briefing. Could the receiving agent act on it without the transcript?\n"
          "What would you lose by passing only the briefing and not the whole conversation?")


if __name__ == "__main__":
    main()
