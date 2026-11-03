# =====================================================================
# Day 2 - MCP: a client that uses the tools of an MCP server
# Version: LangChain, using ChatOpenAI
#
# What you will learn, one step at a time:
#   STEP 1  Start the server and ask it which tools it has (no AI)
#   STEP 2  Call one tool ourselves, directly (still no AI)
#   STEP 3  Give the MCP tools to an agent and let it decide
#
# The server is d2_4_mcp_server.py. Keep it in the same folder.
# This file starts it for you.
#
# Install once:
#   pip install langchain langchain-openai langchain-mcp-adapters "mcp<2" python-dotenv
#
# Run:
#   python d2_4_mcp_client_langchain.py
#
# About "async": MCP keeps a connection open to the server while we work.
# Python code that waits on a connection is written as an "async def"
# function and started with asyncio.run(...). Read each function from top
# to bottom like the other labs; "await" just means "wait for the answer".
# =====================================================================
# ! pip install  "mcp[cli]" anyio langchain-mcp-adapters
import os
import sys
import asyncio

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

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

# How to start the server: run "python d2_4_mcp_server.py" with the same Python as this file.
SERVER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_4_mcp_server.py")
server = StdioServerParameters(command=sys.executable, args=[SERVER_FILE])


# =====================================================================
# STEP 1 and STEP 2: talk to the server directly, without any AI
# =====================================================================
# This shows that MCP is just a protocol: ask for the list of tools,
# then call one by name with arguments. An agent does the same thing,
# except the model chooses which tool to call.

async def talk_to_server_directly():
    print("\n========== STEP 1: which tools does the server have? ==========\n")

    async with stdio_client(server) as (read, write):          # starts the server program
        async with ClientSession(read, write) as session:      # opens the MCP conversation
            await session.initialize()                          # the two sides say hello

            tool_list = await session.list_tools()
            for tool in tool_list.tools:
                print("Tool:", tool.name)
                print("   what it does :", tool.description)
                schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
                print("   inputs       :", list(schema.get("properties", {}).keys()))

            print("\n========== STEP 2: call one tool ourselves ==========\n")

            result = await session.call_tool("get_account", {"customer_id": "C-1001"})
            print("get_account('C-1001') returned:", result.content[0].text)


asyncio.run(talk_to_server_directly())


# =====================================================================
# STEP 3: give the MCP tools to an agent
# =====================================================================
# load_mcp_tools turns every server tool into a normal LangChain tool.
# The agent code is exactly the same as in our earlier labs. The only
# difference is where the tools come from.

async def agent_with_mcp_tools():
    print("\n========== STEP 3: an agent using the MCP tools ==========\n")

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print("Tools given to the agent:", [tool.name for tool in tools])

            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt="You are a telecom billing support agent. Use the tools to check accounts and "
                              "policies before you answer. If a tool refuses an action, tell the user why.",
            )

            questions = [
                "Customer C-1001 was charged a late fee. Can we waive it?",
                "Please give customer C-1002 a 50 dollar credit for the network outage.",
            ]
            for question in questions:
                result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
                print("USER :", question)
                for message in result["messages"]:
                    for call in getattr(message, "tool_calls", None) or []:
                        print("TOOL :", call["name"], call["args"])
                print("AGENT:", result["messages"][-1].content)
                print("-" * 70)


asyncio.run(agent_with_mcp_tools())

# What to notice:
#   - This file has no tool code at all. The tools live in the server.
#   - Change a tool in d2_4_mcp_server.py, run this file again, and the
#     agent uses the new version. Nothing here needs to change.
#   - The 50 dollar credit is refused by the SERVER, not by the prompt.
#     Rules that must always hold belong in the tool code.
