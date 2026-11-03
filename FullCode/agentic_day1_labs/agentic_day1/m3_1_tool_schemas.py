"""
Module 3, Concept 1. Tool schemas.

The model never sees your function. It sees a name, a description and a JSON schema, and it
writes arguments to fit them. The schema is therefore a prompt, and most "the agent called the
tool wrongly" bugs are schema bugs.

This lab covers
  1. Three ways to define a schema: inferred from the signature, parsed from the docstring,
     and an explicit Pydantic model with enums, constraints and a nested object.
  2. Printing the exact JSON the model receives.
  3. Validation: what happens when the model sends arguments that break the schema.
  4. Injected arguments (ToolRuntime) that the tool gets but the model never sees.
  5. An A/B run: the same request against a vague tool and a well-specified one.

Prompts to try in part 5
  - Checkout is down for everyone on app-01 since 09:42, payments-api keeps crashing. Log it.
  - Meera can't open VS Code on ws-114, not urgent.
  - Something is weird with the network.        (too little information: does it ask or guess?)
"""
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from lab_common import banner, llm, run

TICKETS = []


# ---- 1a. Schema inferred from the signature. The docstring becomes the tool description.
@tool
def get_service_status(host: str, service: str) -> str:
    """Return the current state of a service on a host."""
    return f"{service} on {host}: active"


# ---- 1b. Argument descriptions parsed from a Google-style docstring.
@tool(parse_docstring=True)
def read_logs(host: str, service: str, lines: int = 20) -> str:
    """Return the most recent log lines for a service.

    Args:
        host: Hostname exactly as it appears in the inventory, for example app-01.
        service: The systemd unit name without the .service suffix.
        lines: How many lines to return, newest last. Keep it small; logs are long.
    """
    return f"(last {lines} lines of {service} on {host})"


# ---- 1c. Explicit Pydantic schema: enums, constraints, a nested object, and field descriptions
#          that tell the model how to choose a value, not only what the value is.
class Impact(BaseModel):
    users_affected: int = Field(ge=0, description="Best estimate of people affected. Use 0 if unknown.")
    business_critical: bool = Field(description="True if revenue, payments or customer-facing service is affected.")


class CreateTicketArgs(BaseModel):
    title: str = Field(max_length=80, description="One line naming the symptom and the system. No names of people.")
    category: Literal["outage", "degradation", "access", "software", "hardware"] = Field(
        description="outage = service unavailable. degradation = slow or partly failing. access = permissions or "
                    "login. software = install or application fault on one machine. hardware = physical device."
    )
    priority: Literal["P1", "P2", "P3", "P4"] = Field(
        description="P1 = business-critical outage now. P2 = many users or a workaround exists. "
                    "P3 = one user blocked. P4 = request or cosmetic issue."
    )
    affected_host: str | None = Field(default=None, description="Inventory hostname if one is known, otherwise omit.")
    impact: Impact


@dataclass
class Caller:
    user_id: str


@tool(args_schema=CreateTicketArgs)
def create_ticket(title: str, category: str, priority: str, impact: Impact,
                  affected_host: str | None = None, runtime: ToolRuntime[Caller] = None) -> str:
    """Open a service-desk ticket. Call it once per distinct problem, after you know the symptom and its scope."""
    # `runtime` is injected by the framework. It is not in the schema, so the model cannot set the
    # requester's identity. That value comes from the application.
    ticket = {"id": f"TCK-{2001 + len(TICKETS)}", "requested_by": runtime.context.user_id, "title": title,
              "category": category, "priority": priority, "affected_host": affected_host, "impact": impact.model_dump()}
    TICKETS.append(ticket)
    return json.dumps(ticket)


# ---- 5. The vague version of the same tool, for the A/B comparison.
@tool
def ticket(a: str, b: str, c: str) -> str:
    """Make a ticket."""
    TICKETS.append({"id": f"TCK-{2001 + len(TICKETS)}", "a": a, "b": b, "c": c})
    return json.dumps(TICKETS[-1])


SERVICE_DESK_PROMPT = """You are a service-desk intake agent. You turn a user's report into a ticket.

- Open a ticket only when you know the symptom and roughly who is affected. If the report is too thin to
  choose a category or a priority, ask one question and wait for the answer.
- Choose priority from the stated impact. Urgency in the user's tone is not impact.
- After opening a ticket, tell the user the ticket id and the priority you set, with the reason in one line.
"""


def show_schema(t):
    print(json.dumps(convert_to_openai_tool(t), indent=2))


def main():
    banner("1-2. What the model receives for each definition style")
    for t in (get_service_status, read_logs, create_ticket):
        print(f"\n--- {t.name} ---")
        show_schema(t)
    print("\nNote: `runtime` does not appear in create_ticket's schema.")

    banner("3. Validation happens before your function runs")
    bad_args = {"title": "Checkout down", "category": "emergency", "priority": "P0",
                "impact": {"users_affected": -5, "business_critical": "yes"}}
    try:
        CreateTicketArgs(**bad_args)
    except ValidationError as err:
        for e in err.errors():
            print(f"  {'.'.join(map(str, e['loc']))}: {e['msg']}")
    print("Inside an agent this error text goes back to the model as the tool result, and the model retries\n"
          "with corrected arguments. A precise schema turns a silent bad write into a recoverable error.")

    report = "Checkout is down for everyone on app-01 since 09:42, payments-api keeps crashing. Log it."
    banner("5a. Vague tool: ticket(a, b, c)")
    TICKETS.clear()
    run(create_agent(model=llm, tools=[ticket], system_prompt=SERVICE_DESK_PROMPT), report)
    print("Stored:", TICKETS)

    banner("5b. Well-specified tool: create_ticket")
    TICKETS.clear()
    agent = create_agent(model=llm, tools=[create_ticket], system_prompt=SERVICE_DESK_PROMPT, context_schema=Caller)
    run(agent, report, context=Caller(user_id="asha"))
    print("Stored:", json.dumps(TICKETS, indent=2))
    print("\nCompare: which fields can a downstream system rely on in 5a? Which in 5b?")


if __name__ == "__main__":
    main()
