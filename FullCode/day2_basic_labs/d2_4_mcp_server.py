# =====================================================================
# Day 2 - MCP: a very simple MCP server (stdio)
#
# MCP (Model Context Protocol) lets you put tools in a separate program
# called a SERVER. Any MCP CLIENT, such as our agents, Claude Desktop or
# an IDE, can then discover the tools and call them.
#
# This server offers three telecom billing tools from our labs.
# It does not use any AI model. It only answers tool calls.
#
# You do NOT run this file yourself. The client files start it for you:
#   python d2_4_mcp_client_langchain.py
#   python d2_4_mcp_client_farm.py
#
# "stdio" means the client starts this program and the two talk through
# the program's standard input and output, like a private phone line.
#
# Install once:
#   pip install mcp
#   (works with mcp version 1 and version 2; the LangChain client needs version 1)
# =====================================================================

# The MCP library renamed its server class in version 2.
# This works with either version.
try:
    from mcp.server.fastmcp import FastMCP as MCPServer      # mcp version 1.x
except ImportError:
    from mcp.server.mcpserver import MCPServer               # mcp version 2.x

# Give the server a name. Clients see this name.
mcp = MCPServer("telecom-billing", log_level="WARNING")

# A tiny fake customer database
ACCOUNTS = {
    "C-1001": {"name": "Asha", "plan": "UNL-599", "on_time_months": 14, "late_fees_last_6_months": 0},
    "C-1002": {"name": "Ravi", "plan": "CAP-299", "on_time_months": 3, "late_fees_last_6_months": 2},
}
CREDITS_GIVEN = []


# Each function marked with @mcp.tool() becomes a tool.
# The function name, the type hints and the docstring are what the client
# sees, exactly like the tools in our earlier labs.

@mcp.tool()
def get_account(customer_id: str) -> str:
    """Look up a customer account by its id, for example C-1001."""
    account = ACCOUNTS.get(customer_id)
    if account is None:
        return f"ERROR: no customer with id {customer_id}"
    return str(account)


@mcp.tool()
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


@mcp.tool()
def apply_credit(customer_id: str, amount_dollars: float, reason: str) -> str:
    """Apply a credit to a customer's bill. Credits above 20 dollars need a supervisor and are refused."""
    if customer_id not in ACCOUNTS:
        return f"ERROR: no customer with id {customer_id}"
    if amount_dollars > 20:
        return "REFUSED: credits above 20 dollars need a supervisor."
    CREDITS_GIVEN.append({"customer_id": customer_id, "amount": amount_dollars, "reason": reason})
    return f"Credit of {amount_dollars} dollars applied to {customer_id}. Credits applied so far: {len(CREDITS_GIVEN)}"


# Start the server and wait for a client to talk to it over stdio.
if __name__ == "__main__":
    mcp.run()
