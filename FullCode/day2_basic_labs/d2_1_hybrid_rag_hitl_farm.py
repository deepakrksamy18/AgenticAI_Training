# =====================================================================
# Day 2 - Hybrid search RAG with a human in the loop
# Version: client environment (agno, FARM gateway: AzureOpenAI model and embedder)
#
# What you will learn, one step at a time:
#   STEP 1  The knowledge base: a few telecom billing policies
#   STEP 2  Keyword search (BM25): finds exact words and codes
#   STEP 3  Similarity search (FAISS): finds meaning, saved to disk
#   STEP 4  Hybrid search: combine both result lists
#   STEP 5  Why hybrid helps: a score card on test questions
#   STEP 6  RAG: answer a question from the retrieved policies
#   STEP 7  Human in the loop: a supervisor approves, edits or rejects
#
# Install once:
#   pip install agno openai faiss-cpu rank_bm25 numpy python-dotenv
#
# Run:
#   python d2_1_hybrid_rag_hitl_farm.py
# =====================================================================

import os
import re
import json
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.knowledge.embedder.azure_openai import AzureOpenAIEmbedder

load_dotenv()

# ---------------------------------------------------------------------
# Connection settings for the FARM gateway, read from a .env file:
#   FARM_BASE_URL, FARM_API_KEY, FARM_DEPLOYMENT_NAME, MODEL_NAME, FARM_API_VERSION,
#   EMBEDDING_DEPLOYMENT_NAME, EMBEDDING_MODEL_NAME (optional)
# ---------------------------------------------------------------------
CONFIG = {
    "base_url": os.getenv("FARM_BASE_URL"),
    "api_key": os.getenv("FARM_API_KEY"),
    "chat_deployment": os.getenv("FARM_DEPLOYMENT_NAME"),
    "chat_model": os.getenv("MODEL_NAME"),
    "embed_deployment": os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    "api_version": os.getenv("FARM_API_VERSION"),
}

missing = [k for k, v in CONFIG.items() if not v]
if missing:
    raise ValueError(f"Missing .env values: {missing}")

# The FARM gateway needs this extra header on every request
FARM_HEADERS = {"genaiplatform-farm-subscription-key": CONFIG["api_key"]}

chat_model = AzureOpenAI(
    id=CONFIG["chat_model"],
    api_key=CONFIG["api_key"],
    azure_endpoint=CONFIG["base_url"],
    azure_deployment=CONFIG["chat_deployment"],
    api_version=CONFIG["api_version"],
    default_headers=FARM_HEADERS,
)

embedder = AzureOpenAIEmbedder(
    id=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),   # the model your embedding deployment runs
    api_key=CONFIG["api_key"],
    azure_endpoint=CONFIG["base_url"],
    azure_deployment=CONFIG["embed_deployment"],
    api_version=CONFIG["api_version"],
    client_params={"default_headers": FARM_HEADERS},
)

# An agent with no tools: we use it only to send a prompt and get text back.
writer = Agent(model=chat_model)


# Two small helpers. Everything below uses only these two functions.
def embed(text):
    """Turn text into a list of numbers that describe its meaning."""
    return embedder.get_embedding(text)


def ask_llm(prompt):
    """Send one prompt to the model and return the reply as text."""
    return writer.run(prompt).content


# ---------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------
DATA_FOLDER = "rag_data"
os.makedirs(DATA_FOLDER, exist_ok=True)
INDEX_FILE = os.path.join(DATA_FOLDER, "billing_policies.faiss")
DOCS_FILE = os.path.join(DATA_FOLDER, "billing_policies.json")
REVIEW_LOG = os.path.join(DATA_FOLDER, "review_log.json")


# =====================================================================
# STEP 1: The knowledge base
# =====================================================================
# Ten short billing policies. Each one is a "chunk": the unit we search.
# For a long PDF you would first cut it into chunks with a text splitter
# (see Ex3_rag_similarity_keyword_hybrid.py). Here each policy is short
# enough to be one chunk already.
#
# Notice that some policies contain CODES (ERR-4019, UNL-599, PROMO-NEW15).
# Codes are where keyword search shines.

print("\n========== STEP 1: the knowledge base ==========\n")

documents = [
    {"id": "DOC-01", "title": "Duplicate charges",
     "text": "Duplicate charge complaints occur when a customer is billed twice for the same monthly plan fee, "
             "usually because of a payment-gateway retry. The duplicate charge is reversed within 5 to 7 business "
             "days once it is verified against the transaction log. If it stayed unresolved for more than one "
             "billing cycle, the customer also gets a 10 percent goodwill credit on that bill."},
    {"id": "DOC-02", "title": "International roaming disputes",
     "text": "If a customer is charged roaming rates while roaming was switched off and no roaming pack was active, "
             "agents check the tower-connection logs. Charges confirmed as accidental roaming are fully waived. If a "
             "roaming pack was active and the data cap was exceeded, only the overage is billable, with a one-time "
             "50 percent reduction."},
    {"id": "DOC-03", "title": "Mid-cycle plan changes",
     "text": "When a customer upgrades or downgrades in the middle of a billing cycle, the bill shows prorated "
             "charges: the old plan for the days before the change and the new plan for the rest. The first bill "
             "after a change often looks higher. No refund is due, because the prorated total is correct."},
    {"id": "DOC-04", "title": "Late payment fees",
     "text": "A late payment fee of 9 dollars is applied when the balance is not cleared by the due date. A first "
             "late fee is waived on request for customers with twelve months of on-time payments. Repeat late fees "
             "within six months are not waivable. Autopay gives a 2 dollar monthly discount."},
    {"id": "DOC-05", "title": "Data overage",
     "text": "Data overage is billed at 10 dollars per extra gigabyte on capped plans. Customers who were not "
             "warned at 90 percent usage because of a system error can have the overage credited."},
    {"id": "DOC-06", "title": "Autopay failures and error codes",
     "text": "Autopay error ERR-4012 means the bank declined the saved card. Autopay error ERR-4019 means the saved "
             "card has expired and the customer must add a new card. If our own payment system was down on the "
             "charge date, the late fee and any reconnection fee are fully refunded."},
    {"id": "DOC-07", "title": "Paper bill fee",
     "text": "A 3 dollar fee is charged each month to accounts that receive mailed statements instead of e-bills. "
             "If the customer never asked for mailed statements, switch them to e-billing, waive this month's fee "
             "and refund up to three earlier months."},
    {"id": "DOC-08", "title": "Reconnection fee",
     "text": "A reconnection fee of 20 dollars applies when service is restored after a suspension for non-payment. "
             "It is waived if the suspension came from our billing error. For hardship cases it can be split into "
             "two instalments."},
    {"id": "DOC-09", "title": "Promotional discounts",
     "text": "The new-customer promotion PROMO-NEW15 gives a 15 dollar monthly credit for 12 months. When it ends, "
             "the bill rises by 15 dollars. The charge is valid, but retention agents may offer a loyalty discount."},
    {"id": "DOC-10", "title": "Plan catalogue",
     "text": "Plan UNL-599 costs 599 rupees a month and includes unlimited calls, unlimited data at full speed and "
             "100 SMS a day. Plan CAP-299 costs 299 rupees a month with 2 GB of data a day and unlimited calls."},
]

for doc in documents:
    print(doc["id"], "-", doc["title"])


# =====================================================================
# STEP 2: Keyword search (BM25)
# =====================================================================
# BM25 is the classic search-engine method. It scores a document higher
# when it contains the words of the question, especially rare words.
# It knows nothing about meaning: "debited" and "charged" are different words.

print("\n========== STEP 2: keyword search (BM25) ==========\n")


def split_into_words(text):
    """Lower-case the text and split it into words. Keeps codes like err-4019 together."""
    return re.findall(r"[a-z0-9\-]+", text.lower())


bm25 = BM25Okapi([split_into_words(doc["title"] + " " + doc["text"]) for doc in documents])


def keyword_search(question, top_k=3):
    scores = bm25.get_scores(split_into_words(question))
    best = np.argsort(scores)[::-1][:top_k]                 # positions with the highest scores first
    return [documents[i]["id"] for i in best if scores[i] > 0]


print("Question: What does autopay error ERR-4019 mean?")
print("BM25 top 3:", keyword_search("What does autopay error ERR-4019 mean?"))


# =====================================================================
# STEP 3: Similarity search (FAISS), saved to disk
# =====================================================================
# Similarity search compares MEANING using embeddings (lab m4_3).
# We save the index to disk so the next run does not have to rebuild it.

print("\n========== STEP 3: similarity search (FAISS) ==========\n")

START_FRESH = True      # set to False to reuse the index saved by an earlier run

if not START_FRESH and os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE) as f:
        documents = json.load(f)
    print("Loaded the index from disk:", index.ntotal, "documents")
else:
    vectors = np.array([embed(doc["title"] + ". " + doc["text"]) for doc in documents], dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "w") as f:
        json.dump(documents, f, indent=2)
    print("Built the index and saved it to", INDEX_FILE)


def similarity_search(question, top_k=3):
    query = np.array([embed(question)], dtype="float32")
    faiss.normalize_L2(query)
    scores, positions = index.search(query, top_k)
    return [documents[p]["id"] for p in positions[0] if p != -1]


print("Question: My card was debited two times for one subscription")
print("FAISS top 3:", similarity_search("My card was debited two times for one subscription"))


# =====================================================================
# STEP 4: Hybrid search: combine both result lists
# =====================================================================
# We run BOTH searches and merge the two ranked lists with
# "reciprocal rank fusion": a document gets points for being near the top
# of either list, and more points for being high in both.
#
#     points = 1 / (60 + position in keyword list) + 1 / (60 + position in similarity list)
#
# LangChain's EnsembleRetriever (used in Ex3) does the same thing for you.

print("\n========== STEP 4: hybrid search ==========\n")


def hybrid_search(question, top_k=3):
    points = {}
    for result_list in [keyword_search(question, top_k=5), similarity_search(question, top_k=5)]:
        for position, doc_id in enumerate(result_list, start=1):
            points[doc_id] = points.get(doc_id, 0) + 1 / (60 + position)
    ranked = sorted(points, key=points.get, reverse=True)
    return ranked[:top_k]


print("Question: What does autopay error ERR-4019 mean?")
print("Hybrid top 3:", hybrid_search("What does autopay error ERR-4019 mean?"))


# =====================================================================
# STEP 5: Why hybrid helps: a score card
# =====================================================================
# For each test question we know which document holds the answer.
# A method "finds it" if that document is in its top 3 results.

print("\n========== STEP 5: score card ==========\n")

test_questions = [
    ("What does autopay error ERR-4019 mean?",                    "DOC-06"),   # an exact code
    ("What is included in plan UNL-599?",                          "DOC-10"),   # an exact code
    ("My card was debited two times for one subscription",        "DOC-01"),   # different words, same meaning
    ("Is there a charge if I get my statements by post?",          "DOC-07"),   # different words, same meaning
    ("I was charged for roaming but roaming was turned off",       "DOC-02"),   # both kinds of clue
]

print(f"{'Question':58} {'Keyword':>8} {'Similar':>8} {'Hybrid':>8}")
totals = {"keyword": 0, "similarity": 0, "hybrid": 0}
for question, answer_doc in test_questions:
    found = {
        "keyword": answer_doc in keyword_search(question),
        "similarity": answer_doc in similarity_search(question),
        "hybrid": answer_doc in hybrid_search(question),
    }
    for method in totals:
        totals[method] += found[method]
    marks = ["yes" if found[m] else "NO" for m in ["keyword", "similarity", "hybrid"]]
    print(f"{question[:58]:58} {marks[0]:>8} {marks[1]:>8} {marks[2]:>8}")

print(f"{'Found (out of ' + str(len(test_questions)) + ')':58} "
      f"{totals['keyword']:>8} {totals['similarity']:>8} {totals['hybrid']:>8}")

# What to look for:
#   - Keyword search wins on codes (ERR-4019, UNL-599) but can miss a question
#     that uses different words ("debited", "by post").
#   - Similarity search wins on meaning but can miss exact codes.
#   - Hybrid gets the best of both. Your exact numbers depend on the
#     embedding model, so run it and discuss what you see.


# =====================================================================
# STEP 6: RAG: answer a question from the retrieved policies
# =====================================================================
# RAG = Retrieve the right policies, then ask the model to answer
# using ONLY those policies.

print("\n========== STEP 6: answer a question with RAG ==========\n")

EXAMPLE_QUESTION = "I was charged roaming fees but roaming was switched off. Will I get a refund?"

print("Type a billing question for the assistant and press Enter.")
print("Example:", EXAMPLE_QUESTION)
print("Press Enter without typing anything to use the example.")
question = input("Your question: ").strip() or EXAMPLE_QUESTION
print("\nUsing the question:", question)

source_ids = hybrid_search(question, top_k=3)
sources = [doc for doc in documents if doc["id"] in source_ids]

print("\nPolicies found by hybrid search:")
for doc in sources:
    print(" ", doc["id"], "-", doc["title"])

context = "\n\n".join(f"[{doc['id']}] {doc['title']}: {doc['text']}" for doc in sources)

RAG_PROMPT = f"""You are a telecom billing support assistant.
Answer the customer's question using ONLY the policies below.
Mention the policy id, for example [DOC-02], for each fact you use.
If the policies do not answer the question, reply exactly:
"I don't have that information in the billing policy."

Policies:
{context}

Customer question:
{question}

Answer:"""

draft_answer = ask_llm(RAG_PROMPT)
print("\nDraft answer from the assistant:\n")
print(draft_answer)


# =====================================================================
# STEP 7: Human in the loop: a supervisor approves, edits or rejects
# =====================================================================
# The assistant's answer is only a DRAFT. A person checks it before it
# reaches the customer. This is the simplest form of human in the loop:
# the program stops and waits for a decision.

print("\n========== STEP 7: supervisor review ==========\n")

print("You are the supervisor. Read the draft answer above and the policies it used.")
print("Then choose what happens next:")
print("   a  = APPROVE: send the draft to the customer as it is")
print("   e  = EDIT:    you will type a corrected answer, and that is sent instead")
print("   r  = REJECT:  nothing is sent; you will type the reason")

decision = ""
while decision not in ["a", "e", "r"]:
    decision = input("Type a, e or r and press Enter: ").strip().lower()
    if decision not in ["a", "e", "r"]:
        print("Please type only the letter a, e or r.")

if decision == "a":
    final_answer = draft_answer
    note = "approved as drafted"
elif decision == "e":
    print("Type the corrected answer on one line and press Enter:")
    final_answer = input("Corrected answer: ").strip() or draft_answer
    note = "edited by supervisor"
else:
    final_answer = None
    note = input("Why are you rejecting it? Type a short reason: ").strip() or "no reason given"

if final_answer:
    print("\nSENT TO CUSTOMER:\n" + final_answer)
else:
    print("\nNothing was sent. Reason:", note)

# Keep a record of every decision. This is the audit trail that lets you
# check later who approved what, and how often the assistant needed fixing.
log = []
if os.path.exists(REVIEW_LOG):
    with open(REVIEW_LOG) as f:
        log = json.load(f)
log.append({
    "time": datetime.now().isoformat(timespec="seconds"),
    "question": question,
    "sources": source_ids,
    "draft_answer": draft_answer,
    "decision": {"a": "approved", "e": "edited", "r": "rejected"}[decision],
    "note": note,
    "final_answer": final_answer,
})
with open(REVIEW_LOG, "w") as f:
    json.dump(log, f, indent=2)
print("\nDecision saved to", REVIEW_LOG, "- reviews recorded so far:", len(log))
