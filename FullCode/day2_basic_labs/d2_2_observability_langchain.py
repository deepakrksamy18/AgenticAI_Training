# =====================================================================
# Day 2 - Observability: seeing what an agent actually did
# Version: LangChain, using ChatOpenAI
#
# What you will learn, one step at a time:
#   STEP 1  Measure one model call: time, tokens and cost
#   STEP 2  Trace an agent run: every model call and tool call, live
#   STEP 3  Save the trace to a file and summarise it
#   STEP 4  See a failure in the trace
#   STEP 5  The same run in LangSmith (optional)
#
# Install once:
#   pip install langchain langchain-openai python-dotenv
#   pip install langsmith          (only for STEP 5)
#
# Run:
#   python d2_2_observability_langchain.py
# =====================================================================

import os
import json
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# STEP 5 needs this switched on BEFORE LangChain is imported.
# If LANGSMITH_API_KEY is set, every run is also sent to LangSmith.
# Get a free key at https://smith.langchain.com and set it with:
#   set "LANGSMITH_API_KEY=lsv2_..."
# ---------------------------------------------------------------------
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "agentic-day2-observability")
    print("LangSmith tracing is ON. Project:", os.environ["LANGSMITH_PROJECT"])
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    print("LangSmith tracing is OFF (no LANGSMITH_API_KEY). Steps 1 to 4 still work.")

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler

CONFIG = {
    "base_url": os.getenv("AZURE_OPENAI_BASE_URL", ""),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
}

missing = [k for k, v in CONFIG.items() if not v]
if missing:
    raise ValueError(f"Missing settings: {missing}. Set them in .env or with 'set' in the terminal.")

llm = ChatOpenAI(
    model=CONFIG["deployment"],
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
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

start = time.time()
response = llm.invoke("In one sentence, what is a late payment fee?")
seconds = time.time() - start

usage = response.usage_metadata or {}          # the token counts the service reports
input_tokens = usage.get("input_tokens", 0)
output_tokens = usage.get("output_tokens", 0)

print("Answer      :", response.content)
print("Time        :", round(seconds, 2), "seconds")
print("Tokens in   :", input_tokens)
print("Tokens out  :", output_tokens)
print("Cost (est.) : $", round(cost_in_dollars(input_tokens, output_tokens), 5))


# =====================================================================
# STEP 2: Trace an agent run
# =====================================================================
# An agent makes several model calls and tool calls for one question.
# A "callback handler" is a small class that LangChain notifies at every
# step. We use it to print each step and write it to a trace file.

print("\n========== STEP 2: trace an agent run ==========\n")

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


class TraceLogger(BaseCallbackHandler):
    """Prints every model call and tool call, and writes each one to the trace file."""

    def __init__(self, run_name):
        self.run_name = run_name
        self.started = {}                     # start time of each step, by its id

    def write(self, event):
        event["run"] = self.run_name
        event["time"] = datetime.now().isoformat(timespec="seconds")
        with open(TRACE_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")

    # --- model calls ---
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.started[run_id] = time.time()
        print("  [model] thinking ...")

    def on_llm_end(self, response, *, run_id, **kwargs):
        seconds = round(time.time() - self.started.pop(run_id, time.time()), 2)
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        tools_asked = [call["name"] for call in message.tool_calls]
        print(f"  [model] done in {seconds}s, tokens in/out {usage.get('input_tokens', 0)}/"
              f"{usage.get('output_tokens', 0)}, asked for tools: {tools_asked or 'none'}")
        self.write({"type": "model", "seconds": seconds, "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0), "tools_asked": tools_asked})

    # --- tool calls ---
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.started[run_id] = time.time()
        print(f"  [tool ] {serialized.get('name')} called with {input_str}")

    def on_tool_end(self, output, *, run_id, **kwargs):
        seconds = round(time.time() - self.started.pop(run_id, time.time()), 2)
        result = str(getattr(output, "content", output))
        print(f"  [tool ] returned in {seconds}s: {result[:100]}")
        self.write({"type": "tool", "seconds": seconds, "result": result[:300],
                    "error": result.startswith("ERROR")})

    def on_tool_error(self, error, *, run_id, **kwargs):
        print(f"  [tool ] FAILED: {error}")
        self.write({"type": "tool", "seconds": 0, "result": str(error), "error": True})


support_agent = create_agent(
    model=llm,
    tools=[get_account, check_late_fee_waiver],
    system_prompt="You are a telecom billing support agent. Use the tools to check the account and the "
                  "policy before you answer. Never guess an account detail.",
)

if os.path.exists(TRACE_FILE):
    os.remove(TRACE_FILE)                     # start each lab run with an empty trace

question = "Customer C-1001 was charged a late fee this month. Can we waive it?"
print("USER :", question)
start = time.time()
result = support_agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"callbacks": [TraceLogger("late-fee-C-1001")], "run_name": "late-fee-C-1001"},
)
print("AGENT:", result["messages"][-1].content)
print("Whole run took", round(time.time() - start, 2), "seconds")


# =====================================================================
# STEP 3: Save the trace to a file and summarise it
# =====================================================================
# The trace file is plain text: one JSON line per step. Open it in Notepad.
# Because it is saved, we can add up the numbers afterwards. This is
# what a monitoring dashboard does, just at a bigger scale.

print("\n========== STEP 3: summarise the trace file ==========\n")


def summarise_trace(run_name):
    with open(TRACE_FILE) as f:
        events = [json.loads(line) for line in f if json.loads(line)["run"] == run_name]
    model_calls = [e for e in events if e["type"] == "model"]
    tool_calls = [e for e in events if e["type"] == "tool"]
    tokens_in = sum(e["input_tokens"] for e in model_calls)
    tokens_out = sum(e["output_tokens"] for e in model_calls)
    slowest = max(events, key=lambda e: e["seconds"]) if events else None
    print(f"Run '{run_name}'")
    print("  model calls :", len(model_calls))
    print("  tool calls  :", len(tool_calls), "| failed:", sum(e["error"] for e in tool_calls))
    print("  tokens      :", tokens_in, "in,", tokens_out, "out")
    print("  cost (est.) : $", round(cost_in_dollars(tokens_in, tokens_out), 5))
    if slowest:
        print("  slowest step:", slowest["type"], "-", slowest["seconds"], "seconds")


summarise_trace("late-fee-C-1001")


# =====================================================================
# STEP 4: See a failure in the trace
# =====================================================================
# Customer C-9999 does not exist. The tool returns an error. Without a
# trace you would only see the agent's final sentence and might not
# notice anything went wrong. With a trace, the failed step is obvious.

print("\n========== STEP 4: a failure in the trace ==========\n")

question = "Customer C-9999 says they were charged a late fee. Can we waive it?"
print("USER :", question)
result = support_agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"callbacks": [TraceLogger("late-fee-C-9999")], "run_name": "late-fee-C-9999"},
)
print("AGENT:", result["messages"][-1].content)
print()
summarise_trace("late-fee-C-9999")


# =====================================================================
# STEP 5: The same runs in LangSmith (optional)
# =====================================================================
# Our trace file is home-made. LangSmith is a hosted tool that records
# the same information automatically, with screens to browse each run.

print("\n========== STEP 5: LangSmith ==========\n")

if os.environ.get("LANGSMITH_TRACING") == "true":
    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()           # short scripts can end before the traces are uploaded
    print("Open https://smith.langchain.com, choose the project",
          os.environ["LANGSMITH_PROJECT"], "and open the runs named late-fee-C-1001 and late-fee-C-9999.")
    print("Click a run to see each model call, the exact prompt, each tool call, tokens and time.")
else:
    print("Set LANGSMITH_API_KEY and run again to send these runs to LangSmith.")
