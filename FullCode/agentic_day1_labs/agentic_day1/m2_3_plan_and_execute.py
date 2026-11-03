"""Module 2, Lab 3. Plan-and-Execute: workstation onboarding.

Pattern: a planner writes the whole plan first, an executor carries out one step at a time,
and a replanner looks at the results and decides whether to continue, change the plan or finish.

Use it for long tasks with many independent steps, where the plan should be visible before
anything runs. It loses to ReAct when each step depends on the previous result.

Two versions on the same task:
  3a. TodoListMiddleware on a single agent. The plan lives in state["todos"].
  3b. An explicit LangGraph graph with three agents:
        planner    no tools; its prompt lists what the executor can do
        executor   the Lab 2 provisioner, unchanged, so the permission gate still applies
        replanner  consulted when a step is stopped or the plan runs out; decides what happens next

Each executor step is a full agent run (several model calls), so the plan should have one step per
package. Progress is printed before every model call. If the run still pauses for a long time between
lines, the Azure deployment is rate limiting: lab_common sets a 60 second timeout and two retries.

The caller is a developer, so the docker-desktop step is stopped. The replanner must swap in
one access-request step without retrying the install and without abandoning the other steps.

More prompts to try with plan_execute:
  RAVI  the same onboarding task                       (no blocked step)
  ASHA  Set up ws-114 with vscode, wireshark and slack. (completed, policy violation, not found)
  ASHA  Install git on ws-114 and on ws-999.            (unknown host mid-plan)
  ASHA  Make ws-114 ready for the data team.            (underspecified: does the planner invent a list?)
"""
import json
import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from it_env import AUDIT_LOG, ENV, TICKETS, reset_env
from lab_common import banner, llm, run
from m2_2_stop_and_report_guardrails import (
    ASHA,
    PROVISIONING_PROMPT,
    PROVISIONING_TOOLS,
    RequestContext,
    make_caller_prompt,
    permission_gate,
    provisioner,
)


# ---- 3a. One agent that keeps its own todo list
ONBOARDING_TASK = (
    "Set up workstation ws-114 for a new backend developer. They need vscode, git, postgresql-client "
    "and docker-desktop. Verify each install and finish with a summary of what is ready and what is not."
)

todo_agent = create_agent(
    model=llm,
    tools=PROVISIONING_TOOLS,
    middleware=[make_caller_prompt(PROVISIONING_PROMPT), TodoListMiddleware(), permission_gate,
                ModelCallLimitMiddleware(run_limit=30, exit_behavior="end")],
    context_schema=RequestContext,
)


# ---- 3b. Planner, executor and replanner as a graph
PLANNER_PROMPT = """You are the planner for an IT service desk. You turn a request into an ordered list of steps
for an executor agent. You do not carry out any step yourself.

The executor handles one package at a time. Given "Install <package> on <host>" it looks the package up in the
catalogue, installs it and verifies the install by itself, as one job. It can also open an access request for
an administrator. It can do nothing else, so plan only steps it can perform.

Writing steps
- One step per package: "Install <package> on <host> and verify it". Do not write separate steps for the
  catalogue lookup, the install and the verification; that triples the work and the executor does all three
  anyway. A request for four packages is a plan of four steps.
- Name the package and the host in every step, because the executor sees each step on its own without the
  rest of the plan.
- Put steps that are likely to succeed first, so that one blocked step does not hold up the others.
- Do not plan around permissions. You do not know the caller's role; the executor checks it.
- Do not add steps the request did not ask for.
"""

REPLANNER_PROMPT = """You are the replanner for an IT service desk. After each executed step you receive the
original request, the results so far and the steps still planned. Decide what happens next.

- You are consulted when a step was stopped or when no steps remain. If steps remain and the results so far
  do not change them, return decision "continue" with the remaining steps unchanged.
- If a step ended STOPPED with PERMISSION_DENIED or POLICY_VIOLATION, never plan that install again in any
  form. For PERMISSION_DENIED, add exactly one step that opens an access request for that package and host,
  unless one has already been opened. Then carry on with the remaining steps. One blocked package does not
  stop the others.
- If a step ended STOPPED with NOT_FOUND or TOOL_FAILURE, drop it and note it for the final report.
- When no steps remain, return decision "finish" with a final_report for the requester: what is ready, what is
  not and why, and any ticket ids. Report stopped steps as stopped.
"""


class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered, self-contained steps")


class Replan(BaseModel):
    decision: Literal["continue", "finish"]
    remaining_steps: list[str] = Field(default_factory=list, description="Steps still to run, in order")
    final_report: str = Field(default="", description="Only when decision is finish")


planner = create_agent(model=llm, tools=[], system_prompt=PLANNER_PROMPT, response_format=ToolStrategy(Plan))
replanner = create_agent(model=llm, tools=[], system_prompt=REPLANNER_PROMPT, response_format=ToolStrategy(Replan))


class PlanExecuteState(TypedDict, total=False):
    task: str
    plan: list[str]
    past_steps: Annotated[list[dict], operator.add]
    final_report: str


MAX_STEPS = 12


def plan_node(state: PlanExecuteState):
    out = planner.invoke({"messages": [{"role": "user", "content": state["task"]}]})
    steps = out["structured_response"].steps
    print("PLAN")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    return {"plan": steps}


def execute_node(state: PlanExecuteState, runtime: Runtime[RequestContext]):
    step = state["plan"][0]
    done, total = len(state.get("past_steps", [])) + 1, len(state.get("past_steps", [])) + len(state["plan"])
    print(f"\nEXECUTE  step {done} of {total}: {step}")       # printed before the model is called, so progress is visible
    instruction = (f"Carry out this one step and nothing else: {step}\n"
                   "If the step is only a catalogue lookup, a verification or an access request, report COMPLETED "
                   "when that step itself succeeded.")
    out = run(provisioner, instruction, context=runtime.context) or {}     # run() prints each tool call as it happens
    report = out.get("structured_response")
    result = {"step": step, "status": report.status, "error_type": report.error_type,
              "detail": report.reason, "done": report.what_was_done} if report else \
             {"step": step, "status": "STOPPED", "error_type": "TOOL_FAILURE", "detail": "executor returned no report", "done": []}
    print(f"         -> {result['status']} / {result['error_type']}")
    return {"past_steps": [result], "plan": state["plan"][1:]}


def replan_node(state: PlanExecuteState):
    if len(state["past_steps"]) >= MAX_STEPS:
        return {"plan": [], "final_report": f"Stopped after {MAX_STEPS} steps without finishing. Results: {state['past_steps']}"}
    last = state["past_steps"][-1]
    if state["plan"] and last["status"] == "COMPLETED":
        return {}                     # nothing to reconsider: carry on without spending a model call
    print("REPLAN   consulting the replanner" + (" (last step was stopped)" if last["status"] != "COMPLETED" else " (plan finished)"))
    brief = (f"Original request:\n{state['task']}\n\nResults so far:\n{json.dumps(state['past_steps'], indent=2)}"
             f"\n\nSteps still planned:\n{json.dumps(state['plan'], indent=2)}")
    decision = replanner.invoke({"messages": [{"role": "user", "content": brief}]})["structured_response"]
    if decision.decision == "finish" or not decision.remaining_steps:
        return {"plan": [], "final_report": decision.final_report or "Finished with no report."}
    if decision.remaining_steps != state["plan"]:
        print(f"REPLAN   {decision.remaining_steps}")
    return {"plan": decision.remaining_steps}


builder = StateGraph(PlanExecuteState, context_schema=RequestContext)
builder.add_node("plan", plan_node)
builder.add_node("execute", execute_node)
builder.add_node("replan", replan_node)
builder.add_edge(START, "plan")
builder.add_edge("plan", "execute")
builder.add_edge("execute", "replan")
builder.add_conditional_edges("replan", lambda s: END if s.get("final_report") else "execute", [END, "execute"])
plan_execute = builder.compile()


def main():
    banner("3a. TodoListMiddleware")
    reset_env()
    state = run(todo_agent, ONBOARDING_TASK, context=ASHA)
    print("\n--- Final todo list ---")
    for todo in state.get("todos", []):
        print(f"[{todo['status']:>11}] {todo['content']}")
    print("Installed on ws-114:", ENV["hosts"]["ws-114"]["installed"])

    banner("3b. Planner / executor / replanner graph")
    reset_env()
    result = plan_execute.invoke({"task": ONBOARDING_TASK}, context=ASHA, config={"recursion_limit": 60})
    print("\n--- Final report ---\n" + result["final_report"])
    print("\nInstalled on ws-114:", ENV["hosts"]["ws-114"]["installed"])
    print("Tickets:", TICKETS)
    print("Audit log:", AUDIT_LOG)


if __name__ == "__main__":
    main()
