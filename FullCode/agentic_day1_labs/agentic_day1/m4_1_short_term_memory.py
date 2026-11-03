"""
Module 4, Concept 1. Short-term memory: state that lives for one conversation.

An agent has no memory between calls unless you give it one. In LangChain the short-term
memory is the graph state, saved by a checkpointer and keyed by thread_id.

This lab covers
  1. No checkpointer: every call starts from nothing.
  2. Checkpointer + thread_id: the conversation continues. Two threads do not see each other.
  3. State beyond messages: a custom field (the open ticket) that tools read and write with
     Command(update=...). Facts the agent must not lose belong in state, not in chat history.
  4. Inspecting memory: get_state and get_state_history.
  5. Time travel: resume from an earlier checkpoint to explore a different branch.

InMemorySaver loses everything when the process ends. In production use a database-backed
checkpointer (langgraph-checkpoint-sqlite or langgraph-checkpoint-postgres); the agent code
does not change.

Prompts to try (same thread, in order)
  - My laptop ws-114 cannot reach the VPN since this morning.
  - Raise it to P2, the whole team is blocked.
  - What is my ticket number and its priority?
"""
from typing import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lab_common import banner, clip, llm, run

PROMPT = """You are a service-desk agent in an ongoing conversation with one user.

- The open ticket for this conversation is kept by the system, not by you. Before you state a ticket id or a
  priority, read it with get_ticket. Do not rely on your recollection of earlier turns for these values.
- Open at most one ticket per conversation. If one is open, update it.
"""


class DeskState(AgentState):
    """Agent state = messages (inherited) + anything else the conversation must keep."""
    ticket: NotRequired[dict]


@tool
def open_ticket(summary: str, priority: str, runtime: ToolRuntime) -> Command:
    """Open a ticket for this conversation. Priority is P1 to P4."""
    if runtime.state.get("ticket"):
        return Command(update={"messages": [ToolMessage("ERROR: a ticket is already open; use update_priority.",
                                                         tool_call_id=runtime.tool_call_id, status="error")]})
    ticket = {"id": "TCK-3001", "summary": summary, "priority": priority}
    # A tool that changes state returns a Command. The ToolMessage is still required: it answers the tool call.
    return Command(update={"ticket": ticket,
                           "messages": [ToolMessage(f"opened {ticket}", tool_call_id=runtime.tool_call_id)]})


@tool
def update_priority(priority: str, runtime: ToolRuntime) -> Command:
    """Change the priority of the open ticket. Priority is P1 to P4."""
    ticket = {**runtime.state.get("ticket", {}), "priority": priority}
    return Command(update={"ticket": ticket,
                           "messages": [ToolMessage(f"priority is now {priority}", tool_call_id=runtime.tool_call_id)]})


@tool
def get_ticket(runtime: ToolRuntime) -> str:
    """Return the ticket that is open in this conversation, if any."""
    return str(runtime.state.get("ticket") or "no ticket is open in this conversation")


TOOLS = [open_ticket, update_priority, get_ticket]


def main():
    banner("1. No checkpointer: the second call knows nothing about the first")
    stateless = create_agent(model=llm, tools=TOOLS, system_prompt=PROMPT, state_schema=DeskState)
    run(stateless, "My laptop ws-114 cannot reach the VPN since this morning.")
    run(stateless, "What is my ticket number?")

    banner("2-3. Checkpointer and thread_id: the conversation and the ticket persist")
    agent = create_agent(model=llm, tools=TOOLS, system_prompt=PROMPT, state_schema=DeskState,
                         checkpointer=InMemorySaver())
    asha = {"configurable": {"thread_id": "asha-vpn"}}
    ravi = {"configurable": {"thread_id": "ravi-printer"}}
    run(agent, "My laptop ws-114 cannot reach the VPN since this morning.", config=asha)
    run(agent, "Raise it to P2, the whole team is blocked.", config=asha)
    run(agent, "What is my ticket number and its priority?", config=asha)

    print("\nA different thread is a different memory:")
    run(agent, "What is my ticket number?", config=ravi)

    banner("4. Inspecting memory")
    snapshot = agent.get_state(asha)
    print("ticket in state :", snapshot.values.get("ticket"))
    print("messages stored :", len(snapshot.values["messages"]))
    history = list(agent.get_state_history(asha))
    print(f"checkpoints     : {len(history)} (one per step, newest first)")
    for snap in history[:6]:
        last = snap.values["messages"][-1] if snap.values.get("messages") else None
        print(f"  next={str(snap.next):12} ticket={snap.values.get('ticket')}  last={clip(getattr(last, 'content', ''), 60)!r}")

    banner("5. Time travel: branch from the moment before the priority changed")
    before = next((s for s in history if (s.values.get("ticket") or {}).get("priority") not in (None, "P2")
                   and not s.next), None) or history[-1]
    print("resuming from a checkpoint where the ticket was:", before.values.get("ticket"))
    # Invoking with an old checkpoint's config forks the thread there. The original history is kept.
    run(agent, "Actually close this as resolved, the VPN is back. What was the final priority?", config=before.config)
    print("ticket on the main line is still:", agent.get_state(asha).values.get("ticket"))


if __name__ == "__main__":
    main()
