"""
Module 5, Concept 2. AutoGen (AgentChat): agents that work by talking to each other.

AutoGen's core idea is the conversation. You define agents, put them in a team, and a
termination condition decides when the conversation is over. Compare with LangGraph, where
you draw the control flow yourself: here the control flow is the turn-taking rule.

This lab: an operator agent with tools and a reviewer agent without tools, in a round-robin
team. The reviewer checks that the operator verified its fix and replies APPROVE to end it.

Maps to what you built in LangChain
  AssistantAgent(tools=..., system_message=...)   ~  create_agent(tools=..., system_prompt=...)
  RoundRobinGroupChat / SelectorGroupChat         ~  a LangGraph graph of agents
  TextMentionTermination | MaxMessageTermination  ~  ModelCallLimitMiddleware and END edges

Use a separate virtual environment for each framework in this module; their dependencies conflict.
    pip install -U autogen-agentchat "autogen-ext[openai]"
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

CONFIG = {
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "<your-deployment-name>"),
    "base_url": os.getenv("AZURE_OPENAI_BASE_URL", "https://<your-resource>.openai.azure.com/openai/v1/"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", "<your-api-key>"),
}

model_client = OpenAIChatCompletionClient(
    model=CONFIG["deployment"],
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    # A deployment name is not a model name AutoGen recognises, so describe its capabilities.
    model_info={"vision": False, "function_calling": True, "json_output": True,
                "structured_output": True, "family": "unknown"},
)

FLEET = {"app-01": {"payments-api": "failed"}}


async def get_service_status(host: str, service: str) -> str:
    """Return the current state of a service on a host."""
    return f"{service} on {host}: {FLEET.get(host, {}).get(service, 'unknown')}"


async def restart_service(host: str, service: str) -> str:
    """Restart a service on a host. Changes the system."""
    if host not in FLEET:
        return f"ERROR: unknown host '{host}'"
    FLEET[host][service] = "active"
    return f"{service} on {host} restarted"


operator = AssistantAgent(
    name="operator",
    model_client=model_client,
    tools=[get_service_status, restart_service],
    reflect_on_tool_use=True,          # after tool results, make one more model call to write a reply
    system_message=(
        "You are an operations engineer. Check the state of a service before you change it, and check again "
        "after any change. Report what you checked, what you changed and the state you observed afterwards. "
        "If the reviewer asks for something, do it and report again."
    ),
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message=(
        "You review an operations engineer's report. You have no tools and you change nothing. Approve only if "
        "the report shows a check before the change and a check after it that confirms a healthy state. If it "
        "does, reply with the single word APPROVE. If not, say exactly which check is missing."
    ),
)

team = RoundRobinGroupChat(
    [operator, reviewer],
    termination_condition=TextMentionTermination("APPROVE", sources=["reviewer"]) | MaxMessageTermination(10),
)


async def main():
    result = await team.run(task="payments-api on app-01 is reported down. Bring it back and confirm.")
    for message in result.messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        print(f"[{message.source:>8}] {' '.join(content.split())[:260]}")
    print("\nStopped because:", result.stop_reason)
    print("Fleet:", FLEET)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
