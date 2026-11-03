"""
Module 4, Concept 2. Long-term memory: what survives across conversations.

Short-term memory is one thread. Long-term memory is a store that outlives threads and is
organised by namespace, usually per user. The agent decides what to write, which makes the
write policy as much a part of the design as the storage.

This lab covers
  1. A store with per-user namespaces: ("users", user_id, "profile").
  2. Writing: a `remember` tool, governed by a prompt that says what is worth keeping and
     what must never be stored.
  3. Reading, two ways:
       - pushed: dynamic_prompt loads the profile into the system prompt on every run
       - pulled: a `recall` tool the agent calls when it needs to
  4. Isolation: the namespace comes from runtime context, so one user cannot read another's.
  5. Forgetting: a `forget` tool, because users change their minds and facts go stale.

InMemoryStore is for the classroom. In production use a persistent store (for example
PostgresStore from langgraph-checkpoint-postgres); the agent code does not change.

Conversation script (each line is a NEW thread, so only the store carries anything over)
  asha: I'm on macOS and I prefer terminal commands over clicking through menus.
  asha: How do I install git?                         (answer should be a macOS terminal command)
  asha: My VPN password is Monsoon#2291, remember it. (must refuse to store a secret)
  ravi: How do I install git?                         (must not inherit Asha's preferences)
  asha: I've moved to a Windows laptop.               (should update, not duplicate)
"""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

from lab_common import banner, llm, run

PROMPT = """You are an IT helpdesk assistant. You have a long-term memory of each user that persists between
conversations. What you know about the current user is in the <user_profile> block.

Using memory
- Tailor your answers to the profile without announcing that you are doing so.
- The profile can be out of date. If the user says something that contradicts it, believe the user and
  update the memory.

Writing memory
- Save a fact with `remember` when it is about the user or their equipment, is likely to stay true for
  weeks, and would change how you help them next time: operating system, hardware, role, preferred way of
  receiving instructions, recurring problems.
- Use a short stable key such as "os" or "instruction_style", so that a later change overwrites the old
  value instead of sitting beside it.
- Do not save one-off details of the current problem, or anything you inferred but were not told.
- Never save passwords, access tokens, recovery codes or other credentials, even if asked to. Say that you
  do not store credentials and suggest the company password manager.
- When the user asks you to forget something, call `forget` and confirm what was removed.
"""


@dataclass
class Caller:
    user_id: str


def profile_namespace(user_id):
    return ("users", user_id, "profile")


@tool
def remember(key: str, value: str, runtime: ToolRuntime[Caller]) -> str:
    """Save or overwrite one durable fact about the current user. `key` is a short stable label."""
    runtime.store.put(profile_namespace(runtime.context.user_id), key.strip().lower(), {"value": value})
    return f"saved {key} = {value}"


@tool
def recall(runtime: ToolRuntime[Caller]) -> str:
    """Return everything stored about the current user."""
    items = runtime.store.search(profile_namespace(runtime.context.user_id))
    return "\n".join(f"{i.key}: {i.value['value']}" for i in items) or "nothing stored"


@tool
def forget(key: str, runtime: ToolRuntime[Caller]) -> str:
    """Delete one stored fact about the current user."""
    runtime.store.delete(profile_namespace(runtime.context.user_id), key.strip().lower())
    return f"deleted {key}"


@dynamic_prompt
def prompt_with_profile(request: ModelRequest) -> str:
    """Push: load the caller's profile into the system prompt on every model call."""
    items = request.runtime.store.search(profile_namespace(request.runtime.context.user_id))
    profile = "\n".join(f"{i.key}: {i.value['value']}" for i in items) or "(nothing stored yet)"
    return f"{PROMPT}\n<user_profile>\n{profile}\n</user_profile>"


store = InMemoryStore()
agent = create_agent(
    model=llm,
    tools=[remember, recall, forget],
    middleware=[prompt_with_profile],
    context_schema=Caller,
    store=store,            # note: no checkpointer, so every run below is a brand-new conversation
)


def dump(user_id):
    items = store.search(profile_namespace(user_id))
    print(f"  store[{user_id}] =", {i.key: i.value["value"] for i in items})


def main():
    asha, ravi = Caller("asha"), Caller("ravi")

    banner("Conversation 1 (asha): the agent learns something durable")
    run(agent, "I'm on macOS and I prefer terminal commands over clicking through menus.", context=asha)
    dump("asha")

    banner("Conversation 2 (asha, new thread): the answer should use what was learned")
    run(agent, "How do I install git?", context=asha)

    banner("Conversation 3 (asha): a secret must not be stored")
    run(agent, "My VPN password is Monsoon#2291, remember it for next time.", context=asha)
    dump("asha")

    banner("Conversation 4 (ravi): another user's memory is not visible")
    run(agent, "How do I install git?", context=ravi)
    dump("ravi")

    banner("Conversation 5 (asha): a changed fact overwrites the old one")
    run(agent, "I've moved to a Windows laptop.", context=asha)
    dump("asha")

    print("\nDiscuss: who may read this store, how long are entries kept, and how does a user see or\n"
          "correct what the agent has remembered about them?")


if __name__ == "__main__":
    main()
