"""Module 2, Lab 1. ReAct: system troubleshooting agent.

Pattern: reason, act, observe, repeat. The model picks the next tool from everything it has
seen so far, so the path is not known in advance. `create_agent` is this loop.

What this lab adds to the basic loop
  1. A scratchpad tool (note_hypothesis) makes the reasoning visible. With native tool calling
     most models return tool calls and no text, so a "Thought:" instruction is unreliable.
  2. The prompt separates read-only from MUTATING tools: diagnose, then change, then verify.
  3. Middleware bounds the loop: a cap on model calls and a cap on restarts.
  4. The answer is a typed object: response_format=ToolStrategy(IncidentReport).

Test prompts (the first two run in main):
  - Customers are getting 502 errors on checkout. The payments API runs on app-01. Find out why and fix it.
  - Just restart payments-api on app-01, it is down.            (restart fails; does it pivot to diagnosis?)
  - Redis on app-01 is using too much memory and that is why checkout is failing. Fix it.   (wrong hypothesis)
  - payments-api on app-07 is down.                              (unknown host: expect ESCALATED)
  - Free up space on app-01 by rotating /var/lib/docker.         (tool returns a policy error)

Check in the report: root_cause should reach the DEBUG-logging deploy ("disk full" is one hop
short), and the log-level revert belongs under follow_ups because no tool can do it.
"""
from typing import Literal

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from it_env import HYPOTHESES, get_host, reset_env
from lab_common import banner, llm, run, show_report


@tool
def note_hypothesis(hypothesis: str, next_check: str) -> str:
    """Record the hypothesis you are currently testing and the next check you will run. Changes nothing."""
    HYPOTHESES.append({"hypothesis": hypothesis, "next_check": next_check})
    return "noted"


@tool
def get_service_status(host: str, service: str) -> str:
    """Read-only. Return the current state of a service on a host."""
    h = get_host(host)
    if h is None:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'"
    svc = h["services"].get(service)
    if svc is None:
        return f"ERROR[NOT_FOUND]: no service '{service}' on {host}. Known: {sorted(h['services'])}"
    return f"{service}: {svc['state']} ({svc['detail']})"


@tool
def read_logs(host: str, service: str, lines: int = 20) -> str:
    """Read-only. Return the most recent log lines for a service on a host."""
    h = get_host(host)
    if h is None or service not in h["logs"]:
        return f"ERROR[NOT_FOUND]: no logs for '{service}' on '{host}'"
    return "\n".join(h["logs"][service][-lines:])


@tool
def check_disk(host: str) -> str:
    """Read-only. Return disk usage per mount point and the largest directories."""
    h = get_host(host)
    if h is None:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'"
    mounts = ", ".join(f"{m} {p}% used" for m, p in h["disk"].items())
    return f"Mounts: {mounts}. Largest directories: {h['big_dirs']}"


@tool
def check_memory(host: str) -> str:
    """Read-only. Return memory usage for a host."""
    h = get_host(host)
    return h["memory"] if h else f"ERROR[NOT_FOUND]: unknown host '{host}'"


@tool
def list_recent_changes(host: str) -> str:
    """Read-only. Return recent deployments and maintenance on a host."""
    h = get_host(host)
    if h is None:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'"
    return "\n".join(h["changes"]) or "no recorded changes"


@tool
def rotate_logs(host: str, path: str) -> str:
    """MUTATING. Compress and archive log files under a /var/log directory to free disk space."""
    h = get_host(host)
    if h is None:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'"
    if not path.startswith("/var/log/"):
        return "ERROR[POLICY_VIOLATION]: only directories under /var/log may be rotated"
    if path.rstrip("/") not in h["big_dirs"]:
        return f"ERROR[NOT_FOUND]: no such log directory '{path}'"
    if path.rstrip("/") == "/var/log/payments":
        h["disk"]["/var"] = 41
        h["big_dirs"]["/var/log/payments"] = "0.4G"
        return "rotated /var/log/payments: 37.6G archived to cold storage. /var is now 41% used"
    return f"rotated {path}: freed less than 1G"


@tool
def restart_service(host: str, service: str) -> str:
    """MUTATING. Restart a service on a host."""
    h = get_host(host)
    if h is None or service not in h["services"]:
        return f"ERROR[NOT_FOUND]: no service '{service}' on '{host}'"
    if service == "payments-api" and h["disk"]["/var"] >= 98:
        h["logs"][service].append("10:05:11 FATAL startup aborted: [Errno 28] No space left on device")
        return "ERROR[TOOL_FAILURE]: payments-api failed to start: [Errno 28] No space left on device"
    h["services"][service] = {"state": "active", "detail": "running, started just now"}
    h["logs"][service].append("10:06:40 INFO startup complete, listening on :8443")
    return f"{service} restarted and is active"


TROUBLESHOOTING_TOOLS = [
    note_hypothesis, get_service_status, read_logs, check_disk,
    check_memory, list_recent_changes, rotate_logs, restart_service,
]


TROUBLESHOOTER_PROMPT = """You are a site-reliability agent that diagnoses and repairs faults on Linux application hosts.

How you work
- Start from the symptom the user reports and form one hypothesis at a time.
- Before your first diagnostic, and again every time your hypothesis changes, call note_hypothesis with the
  hypothesis and the check that would confirm it or rule it out. Choose the cheapest check that can do either.
- Follow the evidence along the chain of cause. A failed service is a symptom, so keep asking what made it fail
  until you reach something that changed.

Changing the system
- Tools marked MUTATING change the host. Call one only after at least two read-only results point to the same
  cause, and only when the change addresses that cause. Restarting a service is not a diagnosis.
- After every change, check again with a read-only tool. Report a fix as verified only if you saw the healthy
  state yourself after making the change.

When to stop
- If a tool returns ERROR[...] twice for the same call, stop repeating it. Explain what the error tells you and
  try a different line of investigation, or escalate.
- If you cannot find the cause, say what you ruled out and escalate. A clear "not resolved" is a good outcome;
  an invented root cause is not.

Reporting
- Quote the log line or metric behind each claim.
- You can only do what your tools do. Anything else that is needed, such as a configuration change or a
  permanent fix, goes in follow_ups for a human. Do not describe an action as done unless a tool did it.
"""


class IncidentReport(BaseModel):
    status: Literal["RESOLVED", "MITIGATED", "ESCALATED"] = Field(description="RESOLVED only if verified after the fix")
    symptom: str
    root_cause: str = Field(description="The earliest cause in the chain that the evidence supports")
    evidence: list[str] = Field(description="Quoted log lines or metrics, one per item")
    actions_taken: list[str] = Field(description="Only changes actually made through tools")
    verified: bool = Field(description="True only if a read-only check after the fix showed a healthy state")
    follow_ups: list[str] = Field(description="Work a human still needs to do")


troubleshooter = create_agent(
    model=llm,
    tools=TROUBLESHOOTING_TOOLS,
    system_prompt=TROUBLESHOOTER_PROMPT,
    response_format=ToolStrategy(IncidentReport),
    middleware=[
        ModelCallLimitMiddleware(run_limit=18, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="restart_service", run_limit=3),
    ],
)


def main():
    banner("ReAct troubleshooting: main scenario")
    reset_env()
    state = run(troubleshooter, "Customers are getting 502 errors on checkout. The payments API runs on app-01. Find out why and fix it.")
    print("\n--- Incident report ---")
    incident = show_report(state)
    print("\n--- Hypothesis trail ---")
    for i, h in enumerate(HYPOTHESES, 1):
        print(f"{i}. {h['hypothesis']}  ->  {h['next_check']}")

    banner("Instruction to act without diagnosing")
    reset_env()
    state = run(troubleshooter, "Just restart payments-api on app-01, it is down.")
    show_report(state)


if __name__ == "__main__":
    main()
