"""
Module 5, Concept 5. OpenAI Agents SDK: agents, handoffs and guardrails.

The SDK's core idea is a small set of primitives: an Agent (instructions + tools), a handoff
(one agent passes the conversation to another), a guardrail (a check that can stop a run),
and a Runner that drives the loop. There is no graph to draw; routing is done by handoffs.

This lab: a triage agent that hands off to an operations agent or an access agent, plus an
input guardrail that stops requests containing credentials before any model sees them.

Maps to what you built in LangChain
  Agent(instructions, tools)       ~  create_agent(system_prompt, tools)
  handoffs=[agent_a, agent_b]      ~  handoff tools between agents in a LangGraph graph (see m6_2)
  @input_guardrail                 ~  before_agent / before_model middleware that can end the run
  Runner.run(..., context=...)     ~  agent.invoke(..., context=...)

The Anthropic (Claude) Agent SDK follows the same shape, an agent loop with tools, subagents
and hooks, but it needs a Claude model endpoint, so it is not run against this Azure deployment.

Use a separate virtual environment for each framework in this module; their dependencies conflict.
    pip install -U openai-agents
"""
import asyncio
import os
import re
from dataclasses import dataclass

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    function_tool,
    input_guardrail,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

CONFIG = {
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "<your-deployment-name>"),
    "base_url": os.getenv("AZURE_OPENAI_BASE_URL", "https://<your-resource>.openai.azure.com/openai/v1/"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", "<your-api-key>"),
}

set_tracing_disabled(True)      # tracing uploads to the OpenAI platform, which an Azure key cannot use
model = OpenAIChatCompletionsModel(
    model=CONFIG["deployment"],
    openai_client=AsyncOpenAI(base_url=CONFIG["base_url"], api_key=CONFIG["api_key"]),
)

FLEET = {"app-01": {"payments-api": "failed"}}
TICKETS = []


@dataclass
class Caller:
    user_id: str
    role: str


@function_tool
def get_service_status(host: str, service: str) -> str:
    """Return the current state of a service on a host."""
    return f"{service} on {host}: {FLEET.get(host, {}).get(service, 'unknown')}"


@function_tool
def restart_service(ctx: RunContextWrapper[Caller], host: str, service: str) -> str:
    """Restart a service on a host. Changes the system."""
    # ctx is injected by the Runner and is not part of the schema the model sees.
    if ctx.context.role != "admin":
        return f"ERROR[PERMISSION_DENIED]: role '{ctx.context.role}' may not restart services."
    FLEET[host][service] = "active"
    return f"{service} on {host} restarted"


@function_tool
def open_access_request(ctx: RunContextWrapper[Caller], what: str, justification: str) -> str:
    """Open a ticket asking an administrator to grant access or perform a privileged action."""
    TICKETS.append({"id": f"REQ-{1001 + len(TICKETS)}", "by": ctx.context.user_id, "what": what, "why": justification})
    return f"opened {TICKETS[-1]['id']}"


@input_guardrail
async def no_credentials(ctx: RunContextWrapper[Caller], agent: Agent, user_input) -> GuardrailFunctionOutput:
    """Deterministic check that runs before the agent. A tripwire stops the run with an exception."""
    text = user_input if isinstance(user_input, str) else str(user_input)
    found = bool(re.search(r"(password|passwd|api[_ -]?key|token)\s*(is|=|:)\s*\S+", text, re.I))
    return GuardrailFunctionOutput(output_info={"credential_found": found}, tripwire_triggered=found)


ops_agent = Agent[Caller](
    name="Operations agent",
    handoff_description="Diagnoses and restarts services on application hosts.",
    instructions=("You fix service faults. Check a service before you change it and again afterwards. If a tool "
                  "returns ERROR[PERMISSION_DENIED], stop and say what was denied; do not look for another way."),
    tools=[get_service_status, restart_service],
    model=model,
)

access_agent = Agent[Caller](
    name="Access agent",
    handoff_description="Handles requests for permissions, accounts and privileged actions the caller cannot do.",
    instructions="You open access requests. Ask for a justification if none was given, then open one request.",
    tools=[open_access_request],
    model=model,
)

triage_agent = Agent[Caller](
    name="Triage agent",
    instructions=("You are the front desk. Decide which specialist should take the request and hand off to them. "
                  "Do not try to solve the request yourself."),
    handoffs=[ops_agent, access_agent],
    input_guardrails=[no_credentials],
    model=model,
)


async def ask(text, caller):
    print(f"\n[user:{caller.user_id}/{caller.role}] {text}")
    try:
        result = await Runner.run(triage_agent, text, context=caller, max_turns=10)
    except InputGuardrailTripwireTriggered:
        print("[blocked] The request contained a credential, so it was stopped before reaching a model.")
        return
    print(f"[{result.last_agent.name}] {result.final_output}")


async def main():
    await ask("payments-api on app-01 is down. Please restart it.", Caller("ravi", "admin"))
    FLEET["app-01"]["payments-api"] = "failed"
    await ask("payments-api on app-01 is down. Please restart it.", Caller("asha", "developer"))
    await ask("I need permission to restart services on app-01 for the release tonight.", Caller("asha", "developer"))
    await ask("Log in to app-01 for me, my password is Monsoon#2291.", Caller("asha", "developer"))
    print("\nFleet:", FLEET, "| tickets:", TICKETS)


if __name__ == "__main__":
    asyncio.run(main())
