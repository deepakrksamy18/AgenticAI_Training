"""
Module 5, Concept 1. LangGraph: the runtime underneath create_agent.

`create_agent` gives you one fixed graph: model, tools, loop. LangGraph is what you use when
you want to draw the graph yourself: some steps decided by code, some by a model, some by a
person. The rule of thumb for this audience: use a workflow where the path is known, and put
an agent only where the path depends on what is found.

One incident-intake workflow, showing the primitives in the order you meet them:
  State + reducers     a typed dict; Annotated[list, operator.add] lets parallel nodes append
  Nodes                plain functions: state in, partial update out
  A model in a node    triage uses create_agent with structured output
  Conditional edges    code decides the next node from state
  Send                 fan out one node per affected service, in parallel
  interrupt            pause for a human decision and resume with Command(resume=...)
  Checkpointer         makes the pause possible and the run resumable
  Streaming            stream_mode="updates" shows each node as it finishes

            +--> check_service --+
  triage ---+--> check_service --+--> decide --> approval --> remediate --> report
            +--> check_service --+         \\____________________________/

Reports to try
  - Checkout is failing with 502s for all customers and the login page is slow.
  - One user says the wiki looks odd.            (low severity: should skip approval and remediation)
"""
import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from lab_common import banner, llm

HEALTH = {"payments-api": "failed", "auth-service": "degraded", "nginx": "active", "wiki": "active"}
Service = Literal["payments-api", "auth-service", "nginx", "wiki"]


class Triage(BaseModel):
    severity: Literal["high", "low"] = Field(description="high if customers or revenue are affected, otherwise low")
    services: list[Service] = Field(description="Every service the report implicates. Include nginx for any web error.")
    summary: str = Field(description="One sentence restating the problem without speculation about the cause")


class IncidentState(TypedDict, total=False):
    report: str
    triage: dict
    checks: Annotated[list[dict], operator.add]     # reducer: parallel check_service nodes all append here
    approved: bool
    actions: Annotated[list[str], operator.add]
    final_report: str


TRIAGE_PROMPT = """You triage incoming incident reports for an operations team. Classify the report; do not try
to diagnose it. List a service only if the report gives a reason to suspect it."""

REPORT_PROMPT = """You write the closing note for an incident. Use only the facts you are given: the triage, the
health checks and the actions taken. State what was found, what was done and what is still open, in under 120
words. If remediation was not approved, say that plainly."""

triage_agent = create_agent(model=llm, tools=[], system_prompt=TRIAGE_PROMPT, response_format=ToolStrategy(Triage))
report_agent = create_agent(model=llm, tools=[], system_prompt=REPORT_PROMPT)


def triage(state: IncidentState):
    out = triage_agent.invoke({"messages": [{"role": "user", "content": state["report"]}]})
    return {"triage": out["structured_response"].model_dump()}


def fan_out(state: IncidentState):
    """Conditional edge that returns Send objects: one check_service run per service, in parallel."""
    return [Send("check_service", {"service": s}) for s in state["triage"]["services"]] or "decide"


def check_service(payload: dict):
    """Deterministic work. This node receives the Send payload, not the whole state."""
    service = payload["service"]
    return {"checks": [{"service": service, "state": HEALTH.get(service, "unknown")}]}


def decide(state: IncidentState) -> Literal["approval", "report"]:
    """Routing in code: remediation needs a high severity AND a service that is actually unhealthy."""
    unhealthy = [c for c in state.get("checks", []) if c["state"] in ("failed", "degraded")]
    return "approval" if unhealthy and state["triage"]["severity"] == "high" else "report"


def approval(state: IncidentState):
    """Pause here. Everything before this point is saved; nothing after it has run."""
    unhealthy = [c["service"] for c in state["checks"] if c["state"] != "active"]
    answer = interrupt({"question": "Restart these services?", "services": unhealthy})
    return {"approved": str(answer).lower().startswith("approve")}


def remediate(state: IncidentState):
    if not state.get("approved"):
        return {"actions": ["remediation was not approved; nothing was changed"]}
    done = []
    for check in state["checks"]:
        if check["state"] != "active":
            HEALTH[check["service"]] = "active"
            done.append(f"restarted {check['service']}")
    return {"actions": done}


def report(state: IncidentState):
    facts = (f"Triage: {state['triage']}\nHealth checks: {state.get('checks', [])}\n"
             f"Actions: {state.get('actions', ['none; remediation was not needed'])}")
    out = report_agent.invoke({"messages": [{"role": "user", "content": facts}]})
    return {"final_report": out["messages"][-1].text}


builder = StateGraph(IncidentState)
for node in (triage, check_service, approval, remediate, report):
    builder.add_node(node.__name__, node)
builder.add_node("decide", lambda state: {})                    # a join point after the parallel checks
builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", fan_out, ["check_service", "decide"])
builder.add_edge("check_service", "decide")
builder.add_conditional_edges("decide", decide, ["approval", "report"])
builder.add_edge("approval", "remediate")
builder.add_edge("remediate", "report")
builder.add_edge("report", END)
graph = builder.compile(checkpointer=InMemorySaver())


def stream(payload, config):
    """Print each node's update as it completes. Returns True if the run paused for a human."""
    paused = False
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                print(f"  [PAUSED ] {update[0].value}")
                paused = True
            else:
                print(f"  [{node:13}] {update}")
    return paused


def main():
    banner("The graph, as Mermaid (paste into mermaid.live to draw it)")
    print(graph.get_graph().draw_mermaid())

    banner("Run 1: high severity, pauses for approval")
    config = {"configurable": {"thread_id": "incident-1"}}
    paused = stream({"report": "Checkout is failing with 502s for all customers and the login page is slow."}, config)
    if paused:
        print("\n  state is saved; next node:", graph.get_state(config).next)
        print("  ... a human reviews, minutes or days later ...\n")
        stream(Command(resume="approve"), config)
    print("\nFinal report:\n" + (graph.get_state(config).values.get("final_report") or "(none)"))
    print("Service health now:", HEALTH)

    banner("Run 2: low severity, no approval, nothing changed")
    config = {"configurable": {"thread_id": "incident-2"}}
    stream({"report": "One user says the wiki looks odd."}, config)
    print("\nFinal report:\n" + (graph.get_state(config).values.get("final_report") or "(none)"))

    print("\nDiscuss: only two of the six nodes call a model. Which of your current 'agents' are really\n"
          "workflows with one or two model steps in them?")


if __name__ == "__main__":
    main()
