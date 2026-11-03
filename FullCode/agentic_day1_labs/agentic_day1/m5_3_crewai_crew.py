"""
Module 5, Concept 3. CrewAI: roles, tasks and a process.

CrewAI's core idea is the job description. You describe each agent as a role with a goal,
describe each task with its expected output, and pick a process that orders the tasks. Most
of the prompt engineering happens in those descriptions, and the framework assembles them.

This lab: a two-agent crew run sequentially. An SRE diagnoses and fixes a fault with tools,
then a communications writer turns the SRE's result into a customer-facing status update.

Maps to what you built in LangChain
  Agent(role, goal, backstory, tools)   ~  create_agent with a system prompt and tools
  Task(description, expected_output)    ~  the user message plus the output contract
  Task(context=[earlier_task])          ~  passing one agent's result into the next agent's input
  Process.sequential / hierarchical     ~  a fixed pipeline / a supervisor graph

Use a separate virtual environment for each framework in this module; their dependencies conflict.
    pip install -U crewai
"""
import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

CONFIG = {
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "<your-deployment-name>"),
    "base_url": os.getenv("AZURE_OPENAI_BASE_URL", "https://<your-resource>.openai.azure.com/openai/v1/"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", "<your-api-key>"),
}

# The "openai/" prefix selects the OpenAI-compatible client; base_url points it at Azure.
llm = LLM(model=f"openai/{CONFIG['deployment']}", base_url=CONFIG["base_url"], api_key=CONFIG["api_key"])

FLEET = {"app-01": {"payments-api": "failed"}}


@tool("get_service_status")
def get_service_status(host: str, service: str) -> str:
    """Return the current state of a service on a host."""
    return f"{service} on {host}: {FLEET.get(host, {}).get(service, 'unknown')}"


@tool("restart_service")
def restart_service(host: str, service: str) -> str:
    """Restart a service on a host. Changes the system."""
    if host not in FLEET:
        return f"ERROR: unknown host '{host}'"
    FLEET[host][service] = "active"
    return f"{service} on {host} restarted"


sre = Agent(
    role="Site reliability engineer",
    goal="Restore failed services safely and leave an accurate record of what was done",
    backstory="You have run production systems for years. You check before you change anything, and you "
              "check again afterwards, because a fix nobody observed is a guess.",
    tools=[get_service_status, restart_service],
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Incident communications writer",
    goal="Tell customers what happened in plain language without overstating or speculating",
    backstory="You write status-page updates. You use only facts the engineers have confirmed.",
    llm=llm,
    verbose=True,
)

fix_task = Task(
    description="payments-api on {host} is reported down. Check it, restore it, and confirm the result.",
    expected_output="Three short lines: state before, action taken, state after. If it could not be restored, "
                    "say so and say what was tried.",
    agent=sre,
)

update_task = Task(
    description="Write a status-page update for customers about the payments incident, based on the "
                "engineer's record.",
    expected_output="At most 60 words. What was affected, that it is resolved (only if the record confirms "
                    "it), and no internal hostnames.",
    agent=writer,
    context=[fix_task],          # the writer receives the SRE's output
)

crew = Crew(agents=[sre, writer], tasks=[fix_task, update_task], process=Process.sequential)


def main():
    result = crew.kickoff(inputs={"host": "app-01"})
    print("\n--- Engineer's record ---\n" + str(fix_task.output))
    print("\n--- Status-page update ---\n" + str(result))
    print("\nFleet:", FLEET)


if __name__ == "__main__":
    main()
