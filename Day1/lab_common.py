"""Shared setup for the Agentic AI Deep Dive labs (Day 1, Modules 2 to 6).

Every LangChain lab file imports `llm` and the trace helpers from here, so the Azure
connection is configured in one place. Set the environment variables below or edit CONFIG.

    pip install -U langchain langchain-openai langgraph

Run any lab directly, for example:  python m2_1_react_troubleshooter.py
"""
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CONFIG = {
    "deployment": os.getenv("FARM_DEPLOYMENT_NAME", ""),
    "model": os.getenv("MODEL_NAME", ""),
    "base_url": os.getenv("FARM_BASE_URL", ""),
    "api_key": os.getenv("FARM_API_KEY", ""),
    "api_version": os.getenv("FARM_API_VERSION", ""),
    # Only the vector-store lab (m4_3) needs an embedding deployment.
    "embedding_deployment": os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"),
}

required_settings = ("deployment", "model", "base_url", "api_key", "api_version")
missing_settings = [name for name in required_settings if not CONFIG[name]]
if missing_settings:
    raise RuntimeError("Missing Bosch Model Farm settings in Day1/.env: " + ", ".join(missing_settings))

llm = AzureChatOpenAI(
    model=CONFIG["model"],
    azure_deployment=CONFIG["deployment"],
    azure_endpoint=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    api_version=CONFIG["api_version"],
    http_client=httpx.Client(headers={"genaiplatform-farm-subscription-key": CONFIG["api_key"]}),
    timeout=60,       # fail a stalled request after a minute instead of the client default of ten
    max_retries=2,    # then retry twice; a rate-limited deployment surfaces as an error, not a silent wait
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
