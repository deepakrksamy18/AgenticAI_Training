"""
Module 6, Concept 3 (server side). An MCP server that exposes IT-operations tools.

The Model Context Protocol separates the tools from the agent. A server publishes tools,
resources and prompts in a standard shape; any MCP client (a LangChain agent, an IDE
assistant, a desktop app) can discover and call them without custom glue.

This file is the server. It knows nothing about LangChain or about which model will call it.
m6_3_mcp_agent.py starts it as a subprocess and talks to it over stdio.

    pip install -U mcp

Run it on its own to inspect it:   mcp dev m6_3_mcp_server.py      (opens the MCP Inspector)
Run it as a network service:       change the last line to mcp.run(transport="streamable-http")
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("it-ops", log_level="WARNING")

FLEET = {
    "app-01": {"services": {"payments-api": "failed", "nginx": "active"}, "disk": {"/": 46, "/var": 100}},
    "app-02": {"services": {"payments-api": "active", "nginx": "active"}, "disk": {"/": 39, "/var": 52}},
}


@mcp.tool()
def get_service_status(host: str, service: str) -> str:
    """Read-only. Return the current state of a service on a host."""
    if host not in FLEET:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'. Known hosts: {sorted(FLEET)}"
    return f"{service} on {host}: {FLEET[host]['services'].get(service, 'not installed')}"


@mcp.tool()
def check_disk(host: str) -> str:
    """Read-only. Return disk usage per mount point on a host."""
    if host not in FLEET:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'"
    return ", ".join(f"{mount} {used}% used" for mount, used in FLEET[host]["disk"].items())


@mcp.tool()
def rotate_logs(host: str) -> str:
    """MUTATING. Archive old logs under /var/log on a host to free disk space."""
    if host not in FLEET:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'"
    FLEET[host]["disk"]["/var"] = 41
    return f"{host}: logs archived, /var is now 41% used"


@mcp.tool()
def restart_service(host: str, service: str) -> str:
    """MUTATING. Restart a service on a host."""
    if host not in FLEET:
        return f"ERROR[NOT_FOUND]: unknown host '{host}'"
    if FLEET[host]["disk"]["/var"] >= 98:
        return f"ERROR[TOOL_FAILURE]: {service} failed to start: no space left on device"
    FLEET[host]["services"][service] = "active"
    return f"{service} on {host} restarted and is active"


@mcp.resource("inventory://hosts")
def host_inventory() -> str:
    """Resources are read-only context a client can load without a tool call."""
    return "\n".join(f"{host}: services {sorted(data['services'])}" for host, data in FLEET.items())


@mcp.prompt()
def incident_summary(host: str) -> str:
    """Prompts are reusable templates the server offers to clients."""
    return f"Summarise the current health of {host} in three bullet points for an on-call handover."


if __name__ == "__main__":
    mcp.run()          # stdio transport: the client starts this process and talks over stdin/stdout
