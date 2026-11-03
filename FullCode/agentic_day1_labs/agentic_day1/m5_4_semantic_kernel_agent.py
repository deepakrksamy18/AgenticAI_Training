"""
Module 5, Concept 4. Semantic Kernel: plugins, a kernel and filters.

Semantic Kernel's core idea is the kernel: a container that holds AI services and plugins
(classes whose methods are exposed as functions). An agent is a thin layer over a kernel.
It comes from the Microsoft stack, so it is the one most Azure teams meet first, and it has
C# and Java versions with the same concepts.

This lab: an operations plugin, a ChatCompletionAgent, a thread for multi-turn memory, and a
function-invocation filter that blocks a privileged function for a non-admin caller.

Maps to what you built in LangChain
  @kernel_function methods on a plugin class   ~  @tool functions
  ChatCompletionAgent(instructions, plugins)   ~  create_agent(system_prompt, tools)
  ChatHistoryAgentThread                       ~  a checkpointer thread
  function-invocation filter                   ~  wrap_tool_call middleware

Use a separate virtual environment for each framework in this module; their dependencies conflict.
    pip install -U semantic-kernel
"""
import asyncio
import os
from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext
from semantic_kernel.functions import FunctionResult, kernel_function

CONFIG = {
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "<your-deployment-name>"),
    "base_url": os.getenv("AZURE_OPENAI_BASE_URL", "https://<your-resource>.openai.azure.com/openai/v1/"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", "<your-api-key>"),
}

FLEET = {"app-01": {"payments-api": "failed"}}
CALLER = {"user_id": "asha", "role": "developer"}      # set by your application, never by the model
AUDIT_LOG = []


class OpsPlugin:
    """Methods decorated with @kernel_function become functions the model can call.

    Do not name a parameter `service`, `kernel` or `arguments`: Semantic Kernel injects its own
    objects under those names instead of asking the model for a value.
    """

    @kernel_function(description="Return the current state of a service on a host.")
    def get_service_status(self, host: Annotated[str, "Inventory hostname"],
                           service_name: Annotated[str, "Service name"]) -> str:
        return f"{service_name} on {host}: {FLEET.get(host, {}).get(service_name, 'unknown')}"

    @kernel_function(description="Restart a service on a host. Changes the system. Administrators only.")
    def restart_service(self, host: Annotated[str, "Inventory hostname"],
                        service_name: Annotated[str, "Service name"]) -> str:
        FLEET[host][service_name] = "active"
        return f"{service_name} on {host} restarted"


kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(
    ai_model_id=CONFIG["deployment"],
    async_client=AsyncOpenAI(base_url=CONFIG["base_url"], api_key=CONFIG["api_key"]),
))


@kernel.filter(FilterTypes.FUNCTION_INVOCATION)
async def permission_filter(context: FunctionInvocationContext, next):
    """Runs around every function call. Deny by returning a result without calling next()."""
    if context.function.name == "restart_service" and CALLER["role"] != "admin":
        AUDIT_LOG.append({"user": CALLER["user_id"], "function": context.function.name, "decision": "DENIED"})
        context.result = FunctionResult(
            function=context.function.metadata,
            value=f"ERROR[PERMISSION_DENIED]: role '{CALLER['role']}' may not restart services.",
        )
        return
    await next(context)


agent = ChatCompletionAgent(
    kernel=kernel,
    name="ops_agent",
    plugins=[OpsPlugin()],
    instructions=(
        "You are an operations assistant. Check a service before you change it. If a function returns "
        "ERROR[PERMISSION_DENIED], stop: do not retry and do not suggest another way to get the same result. "
        "Tell the user what was denied and that an administrator can do it."
    ),
)


async def ask(text, thread):
    print(f"\n[user ] {text}")
    response = await agent.get_response(messages=text, thread=thread)
    print(f"[agent] {response.content}")
    return response.thread


async def main():
    thread = ChatHistoryAgentThread()
    thread = await ask("Is payments-api running on app-01?", thread)
    thread = await ask("Restart it.", thread)                 # 'it' resolves from the thread; the filter denies
    print("\nFleet:", FLEET, "| audit:", AUDIT_LOG)

    CALLER.update(user_id="ravi", role="admin")
    thread = await ask("I'm the administrator now. Restart it and confirm.", thread)
    print("\nFleet:", FLEET, "| audit:", AUDIT_LOG)


if __name__ == "__main__":
    asyncio.run(main())
