"""
Module 6, Concept 1. Supervisor / worker orchestration as an explicit graph.

In m2_5 the supervisor was a ReAct agent and the workers were its tools: quick to build, but
the delegation loop was hidden inside one agent. Here the same team is a LangGraph graph, so
every hop is a visible, checkpointed step that you can bound, audit and interrupt.

  START -> supervisor --(route)--> diagnostics --+
               ^   \\--(route)--> provisioning --+
               |___________________________________|      supervisor -> END when it decides FINISH

What the explicit version adds
  - The supervisor is a router with structured output (Route). It cannot do the work itself.
  - Workers return to the supervisor through a shared WORKLOG in state. The supervisor never
    sees a worker's internal messages, only its typed report.
  - MAX_ROUNDS is enforced in code. The supervisor is told how many rounds are left.
  - Identity travels in runtime context to the worker that enforces permissions.

Requests to try
  ASHA  Checkout is throwing 502s on app-01. Fix it, then install git and wireshark on app-01.
  RAVI  Install docker-desktop on ws-114 and check that nginx on app-01 is healthy.
  ASHA  Please email the customers an apology.            (no worker covers it: FINISH and say so)
"""
import json
import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from it_env import ENV, reset_env
from lab_common import banner, clip, llm
from m2_1_react_troubleshooter import troubleshooter
from m2_2_stop_and_report_guardrails import ASHA, RequestContext, provisioner

MAX_ROUNDS = 6

SUPERVISOR_PROMPT = """You coordinate an IT service team. You do none of the work yourself. On each turn you see the
user's request and the worklog of jobs already done, and you decide the single next step.

Workers
- diagnostics: finds and repairs faults on application hosts (service down, errors, disk, logs).
- provisioning: installs or verifies catalogue software on a host, or opens an access request.

Routing
- Send one job at a time. Write the instruction so that it stands on its own: the worker sees only that
  text, so include the host, the service or package, and the symptom.
- Do not describe the requester's role or permissions; the system passes identity to the workers.
- A worker's status is final. Never resend a job that came back STOPPED or ESCALATED, in any wording.
- Choose FINISH when every part of the request has an entry in the worklog, when what remains is outside
  both workers' abilities, or when no rounds are left.

Final answer (only with FINISH)
- Give each job's outcome with the worker's status and error_type exactly as reported, then the reason in
  plain language. Say plainly which parts of the request were not done and why.
"""


class Route(BaseModel):
    next: Literal["diagnostics", "provisioning", "FINISH"]
    instruction: str = Field(default="", description="Self-contained job for the worker. Empty when FINISH.")
    final_answer: str = Field(default="", description="Reply to the user. Only when next is FINISH.")


class TeamState(TypedDict, total=False):
    request: str
    worklog: Annotated[list[dict], operator.add]
    instruction: str
    answer: str


router = create_agent(model=llm, tools=[], system_prompt=SUPERVISOR_PROMPT, response_format=ToolStrategy(Route))


def supervisor(state: TeamState) -> Command[Literal["diagnostics", "provisioning", "__end__"]]:
    worklog = state.get("worklog", [])
    rounds_left = MAX_ROUNDS - len(worklog)
    brief = (f"Request:\n{state['request']}\n\nWorklog:\n{json.dumps(worklog, indent=2) if worklog else '(empty)'}"
             f"\n\nRounds left: {rounds_left}")
    route = router.invoke({"messages": [{"role": "user", "content": brief}]})["structured_response"]
    if route.next == "FINISH" or rounds_left <= 0:
        answer = route.final_answer or f"Stopped after {MAX_ROUNDS} rounds. Worklog: {worklog}"
        print(f"  [supervisor ] FINISH")
        return Command(goto=END, update={"answer": answer})
    print(f"  [supervisor ] -> {route.next}: {clip(route.instruction, 150)}")
    return Command(goto=route.next, update={"instruction": route.instruction})


def diagnostics(state: TeamState) -> Command[Literal["supervisor"]]:
    out = troubleshooter.invoke({"messages": [{"role": "user", "content": state["instruction"]}]})
    report = out.get("structured_response")
    entry = {"worker": "diagnostics", "job": state["instruction"],
             **(report.model_dump(include={"status", "root_cause", "actions_taken", "verified"}) if report
                else {"status": "ESCALATED", "root_cause": "worker returned no report"})}
    print(f"  [diagnostics ] {entry['status']}: {clip(entry.get('root_cause', ''), 120)}")
    return Command(goto="supervisor", update={"worklog": [entry]})


def provisioning(state: TeamState, runtime: Runtime[RequestContext]) -> Command[Literal["supervisor"]]:
    # The caller's identity goes to the worker in context, not in the instruction text.
    out = provisioner.invoke({"messages": [{"role": "user", "content": state["instruction"]}]}, context=runtime.context)
    report = out.get("structured_response")
    entry = {"worker": "provisioning", "job": state["instruction"],
             **(report.model_dump(include={"status", "error_type", "reason"}) if report
                else {"status": "STOPPED", "error_type": "TOOL_FAILURE", "reason": "worker returned no report"})}
    print(f"  [provisioning] {entry['status']} / {entry['error_type']}: {clip(entry['reason'], 120)}")
    return Command(goto="supervisor", update={"worklog": [entry]})


builder = StateGraph(TeamState, context_schema=RequestContext)
builder.add_node("supervisor", supervisor)
builder.add_node("diagnostics", diagnostics)
builder.add_node("provisioning", provisioning)
builder.add_edge(START, "supervisor")          # all other edges come from the Command each node returns
team = builder.compile()


def main():
    banner("Supervisor / worker graph")
    reset_env()
    request = "Checkout is throwing 502s on app-01. Fix it, then install git and wireshark on app-01."
    print(f"  [user        ] {request}")
    result = team.invoke({"request": request}, context=ASHA, config={"recursion_limit": 40})
    print("\n--- Answer ---\n" + result["answer"])
    print("\n--- Worklog ---")
    for entry in result.get("worklog", []):
        print(" ", entry)
    print("\npayments-api:", ENV["hosts"]["app-01"]["services"]["payments-api"]["state"],
          "| installed on app-01:", ENV["hosts"]["app-01"]["installed"])
    print("\nDiscuss: the two independent installs ran one after the other. What would change if the\n"
          "supervisor could return several jobs and the graph dispatched them in parallel with Send?")


if __name__ == "__main__":
    main()
