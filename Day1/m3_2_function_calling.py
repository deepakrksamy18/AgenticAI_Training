"""
Module 3, Concept 2. Function calling: the loop underneath every agent.

`create_agent` hides four things. This lab does them by hand first, then shows the same
behavior in the framework, so participants know what to look for when an agent misbehaves.

  1. bind_tools: the tool schemas travel with every model request.
  2. The model replies with tool_calls (name, args, id) instead of text.
  3. Your code runs each call and returns a ToolMessage carrying the same tool_call_id.
  4. Repeat until the model replies with text, or a step limit is reached.

Also covered
  - Parallel tool calls: several calls in one model turn, and how to switch that off.
  - tool_choice: auto, required, or one named tool.
  - Errors as data: a failed tool returns an error message to the model, which can then recover.
  - The framework version: create_agent with ToolRetryMiddleware for transient failures.

Prompts to try
  - Is payments-api healthy on app-01 and app-02?          (expect two calls in one turn)
  - Restart payments-api on app-03.                         (unknown host: the error goes back to the model)
  - What is the capital of France?                          (no tool needed: with tool_choice="required" it must still call one)
"""
import json

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from lab_common import banner, clip, llm, run

FLEET = {"app-01": {"payments-api": "failed"}, "app-02": {"payments-api": "active"}}
ATTEMPTS = {"count": 0}


@tool
def get_service_status(host: str, service: str) -> str:
    """Return the current state of a service on a host."""
    if host not in FLEET:
        raise ValueError(f"unknown host '{host}'. Known hosts: {sorted(FLEET)}")
    return f"{service} on {host}: {FLEET[host].get(service, 'not installed')}"


@tool
def restart_service(host: str, service: str) -> str:
    """Restart a service on a host. Changes the system."""
    if host not in FLEET:
        raise ValueError(f"unknown host '{host}'. Known hosts: {sorted(FLEET)}")
    FLEET[host][service] = "active"
    return f"{service} on {host} restarted and is active"


@tool
def get_cpu_load(host: str) -> str:
    """Return the 5-minute CPU load for a host. The metrics service is unreliable."""
    ATTEMPTS["count"] += 1
    if ATTEMPTS["count"] < 3:
        raise TimeoutError("metrics service timed out")
    return f"{host}: load 0.42"


TOOLS = [get_service_status, restart_service]
SYSTEM = ("You are an operations assistant. Use the tools to check before you answer. If a tool reports an "
          "error, read it, correct the call if you can, and otherwise tell the user what failed.")


def manual_loop(question, *, model, tools, max_steps=6):
    """The agent loop written out. Returns the final text."""
    by_name = {t.name: t for t in tools}
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    for step in range(1, max_steps + 1):
        ai = model.invoke(messages)                      # (2) the model answers or asks for tools
        messages.append(ai)
        if not ai.tool_calls:
            return ai.text
        print(f"  step {step}: {len(ai.tool_calls)} tool call(s) in this turn")
        for call in ai.tool_calls:                       # (3) run every call the model asked for
            print("Tool call plan: ", call["name"], call["args"])
            try:
                result, status = by_name[call["name"]].invoke(call["args"]), "success"
            except Exception as err:                     # errors are data for the model, not a crash
                result, status = f"ERROR: {err}", "error"
            print(f"    {call['name']}({json.dumps(call['args'])}) -> {clip(result, 120)}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"], status=status))
    return "Stopped: step limit reached without a final answer."   # (4) always bound the loop


def main():
    banner("1. The loop by hand, parallel tool calls allowed")
    model = llm.bind_tools(TOOLS)                        # (1) schemas now travel with each request
    print("Answer:", manual_loop("Is payments-api healthy on app-01 and app-02?", model=model, tools=TOOLS))

    banner("2. Same question, one call per turn (parallel_tool_calls=False)")
    serial = llm.bind_tools(TOOLS, parallel_tool_calls=False)
    print("Answer:", manual_loop("Is payments-api healthy on app-01 and app-02?", model=serial, tools=TOOLS))
    print("Count the steps in 1 and 2. Parallel calls save model round trips; serial calls let each\n"
          "result inform the next call. Mutating tools are usually safer in serial.")

    banner("3. A tool error goes back to the model")
    print("Answer:", manual_loop("Restart payments-api on app-03.", model=model, tools=TOOLS))

    banner("4. tool_choice")
    question = [HumanMessage("What is the capital of France?")]
    for choice in ("auto", "required", "get_service_status"):
        ai = llm.bind_tools(TOOLS, tool_choice=choice).invoke(question)
        print(f"  tool_choice={choice!r:22} -> tool_calls={[c['name'] for c in ai.tool_calls]} text={clip(ai.text, 60)!r}")
    print("'required' and a named tool force a call even when none is needed. Use them for a fixed first\n"
          "step or for structured output, and go back to 'auto' afterwards.")

    banner("5. The same loop in the framework, with retries for a flaky tool")
    agent = create_agent(
        model=llm,
        tools=[*TOOLS, get_cpu_load],
        system_prompt=SYSTEM,
        middleware=[ToolRetryMiddleware(max_retries=3, tools=["get_cpu_load"], initial_delay=0.2, jitter=False)],
    )
    ATTEMPTS["count"] = 0
    run(agent, "What is the CPU load on app-02?")
    print(f"get_cpu_load was attempted {ATTEMPTS['count']} time(s); the model saw only the successful result.")


if __name__ == "__main__":
    main()
