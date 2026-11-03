"""Shared setup for the Agentic AI Deep Dive labs (Day 1, Modules 2 to 6).
Version: client environment (FARM gateway).

Every LangChain lab file imports `llm` and the trace helpers from here, so the FARM
connection is configured in one place. The values come from a .env file in this folder.

    pip install -U langchain langchain-openai langgraph python-dotenv

Run any lab directly, for example:  python m2_1_react_troubleshooter.py
"""
import json
import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.types import Command

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

# The chat model. AzureChatOpenAI is LangChain's Azure client: it takes the same values as
# agno's AzureOpenAI in the FARM example (endpoint, deployment, API version, key, header).
llm = AzureChatOpenAI(
    azure_endpoint=CONFIG["base_url"],
    azure_deployment=CONFIG["chat_deployment"],
    model=CONFIG["chat_model"],                    # underlying model, e.g. "gpt-4o"
    api_version=CONFIG["api_version"],
    api_key=CONFIG["api_key"],
    default_headers=FARM_HEADERS,
    timeout=60,       # fail a stalled request after a minute instead of the client default of ten
    max_retries=2,    # then retry twice; a rate-limited deployment surfaces as an error, not a silent wait
)

# Embeddings are only needed by m4_3. Set EMBEDDING_DEPLOYMENT_NAME (and EMBEDDING_MODEL_NAME
# if it is not text-embedding-3-small) to use it.
embeddings = None
if os.getenv("EMBEDDING_DEPLOYMENT_NAME"):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=CONFIG["base_url"],
        azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
        model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        api_version=CONFIG["api_version"],
        api_key=CONFIG["api_key"],
        default_headers=FARM_HEADERS,
        check_embedding_ctx_length=False,          # send plain text
    )


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def clip(text, width=320):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def show_message(msg):
    """Print one message from an agent trace."""
    if msg.type == "ai":
        if msg.text:
            print(f"  [model] {clip(msg.text)}")
        for call in msg.tool_calls:
            print(f"  [call ] {call['name']}({json.dumps(call['args'])})")
    elif msg.type == "tool":
        content = msg.text or msg.content          # MCP tools return content blocks; .text flattens them
        flag = "ERROR" if msg.status == "error" or str(content).startswith("ERROR[") else "ok"
        print(f"  [tool ] {msg.name} -> {flag}: {clip(content)}")


def run(agent, prompt, *, context=None, config=None):
    """Run an agent, print a step-by-step trace, and return the final state.

    `prompt` is a user message, or a `Command` when resuming after an interrupt.
    """
    payload = prompt if isinstance(prompt, Command) else {"messages": [{"role": "user", "content": prompt}]}
    if not isinstance(prompt, Command):
        print(f"  [user ] {clip(prompt)}")
    final = None
    for mode, chunk in agent.stream(payload, config=config, context=context, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
            continue
        for node, update in chunk.items():
            if node == "__interrupt__":
                for intr in update:
                    value = intr.value if isinstance(intr.value, dict) else {"question": intr.value}
                    for req in value.get("action_requests", []):
                        print(f"  [PAUSED] approval needed for {req['name']}({req['args']})")
                continue
            if not isinstance(update, dict):
                continue
            messages = update.get("messages") or []
            for msg in messages if isinstance(messages, list) else []:
                if hasattr(msg, "type"):
                    show_message(msg)
    return final


def show_report(state):
    """Pretty-print the structured response of a finished agent run."""
    report = (state or {}).get("structured_response")
    if report is None:
        print("No structured response (the run was cut short or interrupted).")
    else:
        print(report.model_dump_json(indent=2))
    return report
