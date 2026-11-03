# =====================================================================
# Ex1 - agno agents with built-in and custom tools
# Version: client environment (agno, FARM gateway: AzureOpenAI model)
#
# Install once:
#   pip install agno openai ddgs python-dotenv
#
# Run:
#   python Ex1_agent_demo_farm.py
#
# The DuckDuckGo part needs internet access to duckduckgo.com. If the
# client network blocks it, the first agent fails and the rest still runs.
# =====================================================================

import os
from datetime import datetime
from textwrap import dedent

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.tools import tool
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

# ---------------------------------------------------------------------
# Connection settings for the FARM gateway, read from a .env file:
#   FARM_BASE_URL, FARM_API_KEY, FARM_DEPLOYMENT_NAME, MODEL_NAME, FARM_API_VERSION
# ---------------------------------------------------------------------
CONFIG = {
    "base_url": os.getenv("FARM_BASE_URL"),
    "api_key": os.getenv("FARM_API_KEY"),
    "chat_deployment": os.getenv("FARM_DEPLOYMENT_NAME"),
    "chat_model": os.getenv("MODEL_NAME"),
    "api_version": os.getenv("FARM_API_VERSION"),
}

missing = [k for k, v in CONFIG.items() if not v]
if missing:
    raise ValueError(f"Missing .env values: {missing}")

# The FARM gateway needs this extra header on every request
FARM_HEADERS = {"genaiplatform-farm-subscription-key": CONFIG["api_key"]}

chat_model = AzureOpenAI(
    id=CONFIG["chat_model"],                     # underlying model, e.g. "gpt-4o"
    api_key=CONFIG["api_key"],
    azure_endpoint=CONFIG["base_url"],
    azure_deployment=CONFIG["chat_deployment"],  # Azure deployment name
    api_version=CONFIG["api_version"],
    default_headers=FARM_HEADERS,
)


# ─── Agent 1: a built-in web search tool ─────────────────────────────────────
agent = Agent(
    model=chat_model,
    tools=[DuckDuckGoTools()],
    markdown=True,
    description="You're an enthusiastic news reporter with flair!",
)

agent.print_response("Who is the current Delhi CM", stream=True)
agent.print_response("Who was the Delhi CM in 2015", stream=True)


# ─── Agent 2: custom tools ───────────────────────────────────────────────────
@tool(name="web_search", description="Search the web for information", show_result=True)
def web_search(query: str) -> str:
    """Web search tool"""
    # stub: integrate an actual search API here
    return f"[Search results for {query}]"


@tool(name="calculator", description="Compute math expressions", show_result=True)
def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


agent2 = Agent(
    model=chat_model,
    tools=[web_search, calculator],              # with web search and calculator
    description=dedent("""
        You are a helpful research assistant. Use tools when needed,
        and think step-by-step before answering.
    """),
    markdown=True,                               # allow formatted output
)

print(agent2.run("What is the latest news on telecom domain?").content)

# The whole response object: content plus tool calls, metrics and more
print(agent2.run("What is 12345 * (67 + 89)?"))

# Only the answer text
print(agent2.run("What is 12345 * (67 + 89)?").content)
