"""
Module 4, Concept 3. Vector stores as agent memory.

Key-value memory answers "what is stored under this key". A vector store answers "what have
we seen that is like this", which is what episodic memory needs: the agent recalls past
incidents that resemble the current one even when no word matches exactly.

This lab covers
  1. Indexing past incident reports with embeddings and metadata.
  2. Retrieval as a tool: the agent decides when to search and writes its own query.
  3. Scores and a relevance floor, so weak matches are not presented as precedent.
  4. Write-back: after resolving a new incident the agent stores it, and a later, differently
     worded report finds it. This uses the LangGraph store with a semantic index.

Needs an embedding deployment: set AZURE_OPENAI_EMBEDDING_DEPLOYMENT (see lab_common.CONFIG).

Prompts to try
  - Checkout requests are failing with 502 on app-01. Have we seen this before?
  - Users in the Pune office cannot print since the weekend.
  - The coffee machine on floor 3 is broken.          (nothing relevant: the agent should say so)
"""
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore

from lab_common import CONFIG, banner, llm, run

embeddings = OpenAIEmbeddings(
    model=CONFIG["embedding_deployment"],
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    check_embedding_ctx_length=False,   # send plain text; needed for non-OpenAI-hosted deployments
)

PAST_INCIDENTS = [
    ("INC-101", "payments", "Checkout returned 502 Bad Gateway. payments-api crash-looped because /var was full of DEBUG "
     "logs after a deploy enabled verbose logging. Fix: rotated /var/log/payments, restarted, reverted log level."),
    ("INC-087", "payments", "Card payments timed out for ten minutes. The upstream bank gateway rotated its TLS "
     "certificate and our trust store was stale. Fix: updated the CA bundle and restarted payments-api."),
    ("INC-064", "network", "Pune office lost access to shared printers after a weekend switch firmware upgrade reset "
     "the print VLAN. Fix: restored VLAN 40 on the floor switches."),
    ("INC-052", "identity", "Contractors could not log in on Monday. Their accounts had expired at quarter end. "
     "Fix: extended the accounts and added an expiry report to the weekly review."),
    ("INC-033", "storage", "Nightly backup failed on app-02 because the data volume reached 100 percent. "
     "Fix: expanded the volume by 50 GB and added a disk alert at 80 percent."),
]
RELEVANCE_FLOOR = 0.35   # tune for your embedding model: print the scores first, then choose

incident_index = InMemoryVectorStore(embeddings)
resolved_store = None    # created in main(); holds incidents the agent resolves during this session


@tool
def search_past_incidents(query: str, area: str | None = None) -> str:
    """Search previous incident reports for ones that resemble a symptom.

    Describe the symptom and the system in the query, as you would to a colleague. `area` optionally
    restricts the search to one of: payments, network, identity, storage.
    """
    flt = (lambda doc: doc.metadata["area"] == area) if area else None
    hits = incident_index.similarity_search_with_score(query, k=3, filter=flt)
    strong = [(doc, score) for doc, score in hits if score >= RELEVANCE_FLOOR]
    if not strong:
        return "No past incident is similar enough to be useful."
    return "\n".join(f"[{doc.metadata['id']}] similarity {score:.2f}: {doc.page_content}" for doc, score in strong)


@tool
def record_resolution(incident_id: str, summary: str, runtime: ToolRuntime) -> str:
    """Store a newly resolved incident so that it can be found in future. Summary: symptom, cause and fix."""
    runtime.store.put(("incidents", "resolved"), incident_id, {"text": summary})
    return f"stored {incident_id}"


@tool
def search_recent_resolutions(query: str, runtime: ToolRuntime) -> str:
    """Search incidents resolved recently by this team, including ones not yet in the main archive."""
    hits = runtime.store.search(("incidents", "resolved"), query=query, limit=2)
    return "\n".join(f"[{h.key}] similarity {h.score:.2f}: {h.value['text']}" for h in hits) or "nothing recorded yet"


PROMPT = """You are an incident analyst. You help engineers by recalling how similar problems were handled.

- Search before you answer. Write the query yourself from the symptom; do not paste the user's sentence.
- Check both the archive and the recent resolutions.
- Cite the incident id for every precedent you use, and say how the current case resembles it and how it
  differs. A precedent is a lead to check, not a diagnosis.
- If the search finds nothing similar, say that there is no precedent. Do not stretch a weak match.
- When the user tells you how an incident was resolved, store it with record_resolution.
"""


def main():
    global resolved_store
    banner("1. Index the incident archive")
    incident_index.add_documents([Document(page_content=text, metadata={"id": iid, "area": area})
                                  for iid, area, text in PAST_INCIDENTS])
    dims = len(embeddings.embed_query("probe"))
    print(f"indexed {len(PAST_INCIDENTS)} incidents, embedding size {dims}")

    banner("2. Look at raw scores before trusting a threshold")
    for query in ("502 errors on checkout", "office printers unreachable", "coffee machine broken"):
        hits = incident_index.similarity_search_with_score(query, k=2)
        print(f"  {query!r}: " + ", ".join(f"{d.metadata['id']} {s:.2f}" for d, s in hits))

    resolved_store = InMemoryStore(index={"embed": embeddings, "dims": dims, "fields": ["text"]})
    agent = create_agent(model=llm, tools=[search_past_incidents, search_recent_resolutions, record_resolution],
                         system_prompt=PROMPT, store=resolved_store)

    banner("3. Retrieval as a tool")
    run(agent, "Checkout requests are failing with 502 on app-01. Have we seen this before?")
    run(agent, "The coffee machine on floor 3 is broken.")

    banner("4. Write-back, then recall with different wording")
    run(agent, "For the record: INC-140 today, the wiki was unreachable because its database ran out of "
               "connections after a reporting job leaked them. We restarted the job with a connection limit.")
    run(agent, "Confluence pages won't load and the DB shows too many clients. Anything recent like that?")

    print("\nDiscuss: what should be embedded (the whole report or a summary), who curates what is written\n"
          "back, and what happens when a stored 'fix' later turns out to be wrong?")


if __name__ == "__main__":
    main()
