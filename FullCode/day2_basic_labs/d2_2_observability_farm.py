# =====================================================================
# Day 2 - Observability: seeing what an agent actually did
# Version: client environment (agno, FARM gateway: AzureOpenAI model)
#
# What you will learn, one step at a time:
#   STEP 1  Measure one model call: time, tokens and cost
#   STEP 2  Watch an agent run live with agno's debug mode
#   STEP 3  Record every step of a run in a trace file
#   STEP 4  Summarise the trace file
#   STEP 5  See a failure in the trace
#   STEP 6  Hosted observability tools
#
# Install once:
#   pip install agno openai python-dotenv
#
# Run:
#   python d2_2_observability_farm.py
# =====================================================================

import os
import json
import time
from datetime import datetime

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure import AzureOpenAI

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

FARM_HEADERS = {"genaiplatform-farm-subscription-key": CONFIG["api_key"]}

chat_model = AzureOpenAI(
    id=CONFIG["chat_model"],
    api_key=CONFIG["api_key"],
    azure_endpoint=CONFIG["base_url"],
    azure_deployment=CONFIG["chat_deployment"],
    api_version=CONFIG["api_version"],
    default_headers=FARM_HEADERS,
)

# Prices are only examples. Replace them with the prices on your own contract.
PRICE_PER_1000_INPUT_TOKENS = 0.0025     # dollars
PRICE_PER_1000_OUTPUT_TOKENS = 0.0100    # dollars

TRACE_FILE = "agent_trace.jsonl"         # one line of JSON per event


def cost_in_dollars(input_tokens, output_tokens):
    return (input_tokens / 1000 * PRICE_PER_1000_INPUT_TOKENS
            + output_tokens / 1000 * PRICE_PER_1000_OUTPUT_TOKENS)


# =====================================================================
# STEP 1: Measure one model call
# =====================================================================
# Three numbers matter for every model call:
#   time    - how long the user waited
#   tokens  - how much text went in and came out
#   cost    - tokens multiplied by the price

print("\n========== STEP 1: measure one model call ==========\n")

writer = Agent(model=chat_model)

start = time.time()
response = writer.run("In one sentence, what is a late payment fee?")
seconds = time.time() - start

input_tokens = response.metrics.input_tokens if response.metrics else 0
output_tokens = response.metrics.output_tokens if response.metrics else 0

print("Answer      :", response.content)
print("Time        :", round(seconds, 2), "seconds")
print("Tokens in   :", input_tokens)
print("Tokens out  :", output_tokens)
print("Cost (est.) : $", round(cost_in_dollars(input_tokens, output_tokens), 5))


# =====================================================================
# STEP 2: Watch an agent run live with agno's debug mode
# =====================================================================
# An agent makes several model calls and tool calls for one question.
# debug_mode=True makes agno print every message, tool call and metric
# while the run happens. It is the quickest way to see inside an agent.

print("\n========== STEP 2: debug mode ==========\n")

# A tiny fake customer database for the tools
ACCOUNTS = {
    "C-1001": {"name": "Asha", "plan": "UNL-599", "on_time_months": 14, "late_fees_last_6_months": 0},
    "C-1002": {"name": "Ravi", "plan": "CAP-299", "on_time_months": 3, "late_fees_last_6_months": 2},
}


def get_account(customer_id: str) -> str:
    """Look up a customer account by its id, for example C-1001."""
    time.sleep(0.3)                                   # pretend this is a slow database
    account = ACCOUNTS.get(customer_id)
    if account is None:
        return f"ERROR: no customer with id {customer_id}"
    return json.dumps(account)


def check_late_fee_waiver(customer_id: str) -> str:
    """Check whether a customer's late payment fee can be waived under the policy."""
    account = ACCOUNTS.get(customer_id)
    if account is None:
        return f"ERROR: no customer with id {customer_id}"
    if account["late_fees_last_6_months"] > 0:
        return "NOT WAIVABLE: repeat late fee within six months."
    if account["on_time_months"] >= 12:
        return "WAIVABLE: first late fee and at least twelve months of on-time payments."
    return "NOT WAIVABLE: fewer than twelve months of on-time payments."


SUPPORT_PROMPT = ("You are a telecom billing support agent. Use the tools to check the account and the "
                  "policy before you answer. Never guess an account detail.")

debug_agent = Agent(
    model=chat_model,
    tools=[get_account, check_late_fee_waiver],
    instructions=SUPPORT_PROMPT,
    debug_mode=True,                 # print everything as it happens
)

debug_agent.run("Customer C-1001 was charged a late fee this month. Can we waive it?")

# Debug output is good for one run on your screen. It is too noisy to keep
# for every run, and it disappears when the window closes. So next we keep
# a short, permanent record of each run instead.


# =====================================================================
# STEP 3: Record every step of a run in a trace file
# =====================================================================
# After a run, agno gives us the details: every tool call with its inputs,
# result and time, and the token totals. We write them to a trace file,
# one line of JSON per step. Open the file in Notepad to see it.

print("\n========== STEP 3: record a trace ==========\n")

support_agent = Agent(
    model=chat_model,
    tools=[get_account, check_late_fee_waiver],
    instructions=SUPPORT_PROMPT,
)

if os.path.exists(TRACE_FILE):
    os.remove(TRACE_FILE)                     # start each lab run with an empty trace


def run_and_trace(run_name, question):
    """Run the agent, print what happened, and append each step to the trace file."""
    start = time.time()
    response = support_agent.run(question)
    seconds = round(time.time() - start, 2)

    print("USER :", question)
    events = []
    for tool in response.tools or []:
        tool_seconds = round(tool.metrics.duration, 2) if tool.metrics and tool.metrics.duration else 0
        result = str(tool.result)
        print(f"TOOL : {tool.tool_name}({tool.tool_args}) -> {result[:80]}  [{tool_seconds}s]")
        events.append({"type": "tool", "name": tool.tool_name, "args": tool.tool_args,
                       "result": result[:300], "seconds": tool_seconds, "error": result.startswith("ERROR")})

    model_calls = sum(1 for m in (response.messages or []) if m.role == "assistant")
    metrics = response.metrics
    events.append({"type": "model", "calls": model_calls, "seconds": seconds,
                   "input_tokens": metrics.input_tokens if metrics else 0,
                   "output_tokens": metrics.output_tokens if metrics else 0})
    print("AGENT:", response.content)

    with open(TRACE_FILE, "a") as f:
        for event in events:
            event["run"] = run_name
            event["time"] = datetime.now().isoformat(timespec="seconds")
            f.write(json.dumps(event) + "\n")


run_and_trace("late-fee-C-1001", "Customer C-1001 was charged a late fee this month. Can we waive it?")


# =====================================================================
# STEP 4: Summarise the trace file
# =====================================================================
# Because the trace is saved, we can add up the numbers afterwards.
# This is what a monitoring dashboard does, just at a bigger scale.

print("\n========== STEP 4: summarise the trace file ==========\n")


def summarise_trace(run_name):
    with open(TRACE_FILE) as f:
        events = [json.loads(line) for line in f]
    events = [e for e in events if e["run"] == run_name]
    model = [e for e in events if e["type"] == "model"]
    tools = [e for e in events if e["type"] == "tool"]
    tokens_in = sum(e["input_tokens"] for e in model)
    tokens_out = sum(e["output_tokens"] for e in model)
    print(f"Run '{run_name}'")
    print("  model calls :", sum(e["calls"] for e in model))
    print("  tool calls  :", len(tools), "| failed:", sum(e["error"] for e in tools))
    print("  tokens      :", tokens_in, "in,", tokens_out, "out")
    print("  cost (est.) : $", round(cost_in_dollars(tokens_in, tokens_out), 5))
    print("  total time  :", sum(e["seconds"] for e in model), "seconds")
    if tools:
        slowest = max(tools, key=lambda e: e["seconds"])
        print("  slowest tool:", slowest["name"], "-", slowest["seconds"], "seconds")


summarise_trace("late-fee-C-1001")


# =====================================================================
# STEP 5: See a failure in the trace
# =====================================================================
# Customer C-9999 does not exist. The tool returns an error. Without a
# trace you would only see the agent's final sentence and might not
# notice anything went wrong. With a trace, the failed step is obvious.

print("\n========== STEP 5: a failure in the trace ==========\n")

run_and_trace("late-fee-C-9999", "Customer C-9999 says they were charged a late fee. Can we waive it?")
print()
summarise_trace("late-fee-C-9999")


# =====================================================================
# STEP 6: Hosted observability tools
# =====================================================================

print("""
========== STEP 6: hosted observability tools ==========

Our trace file is home-made, so it works anywhere, including behind the
FARM gateway. Hosted tools such as Langfuse, Arize Phoenix or LangSmith
record the same information automatically and give you screens to search
runs, compare prompts and watch cost over time. agno can send its runs to
these tools; check with your platform team which one is approved before
sending any customer data outside the company.
""")
