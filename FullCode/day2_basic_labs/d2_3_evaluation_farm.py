# =====================================================================
# Day 2 - Evaluation: measuring whether the RAG assistant is good enough
# Version: client environment (agno, FARM gateway: AzureOpenAI model and embedder)
#
# What you will learn, one step at a time:
#   STEP 1  Rebuild the RAG assistant from d2_1 (same data, same search)
#   STEP 2  Write a test set: questions with known right answers
#   STEP 3  Evaluate retrieval: did search find the right policy?
#   STEP 4  Evaluate answers with simple rules: are the key facts there?
#   STEP 5  Evaluate answers with an LLM judge: is it faithful and correct?
#   STEP 6  The report card, saved to a CSV file, and a release decision
#
# Install once:
#   pip install agno openai faiss-cpu rank_bm25 numpy python-dotenv
#
# Run:
#   python d2_3_evaluation_farm.py
# =====================================================================

import os
import re
import csv
import json

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


# =====================================================================
# STEP 1: Rebuild the RAG assistant from d2_1
# =====================================================================
# This is the same knowledge base and the same hybrid search as d2_1,
# without the explanations. See d2_1 for how each part works.

print("\n========== STEP 1: the RAG assistant ==========\n")

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


def split_into_words(text):
    return re.findall(r"[a-z0-9\-]+", text.lower())


bm25 = BM25Okapi([split_into_words(d["title"] + " " + d["text"]) for d in documents])
vectors = np.array([embed(d["title"] + ". " + d["text"]) for d in documents], dtype="float32")
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)


def keyword_search(question, top_k=3):
    scores = bm25.get_scores(split_into_words(question))
    return [documents[i]["id"] for i in np.argsort(scores)[::-1][:top_k] if scores[i] > 0]


def similarity_search(question, top_k=3):
    query = np.array([embed(question)], dtype="float32")
    faiss.normalize_L2(query)
    _, positions = index.search(query, top_k)
    return [documents[p]["id"] for p in positions[0] if p != -1]


def hybrid_search(question, top_k=3):
    points = {}
    for result_list in [keyword_search(question, 5), similarity_search(question, 5)]:
        for position, doc_id in enumerate(result_list, start=1):
            points[doc_id] = points.get(doc_id, 0) + 1 / (60 + position)
    return sorted(points, key=points.get, reverse=True)[:top_k]


REFUSAL = "I don't have that information in the billing policy."


def rag_answer(question):
    """Retrieve with hybrid search, then answer from the retrieved policies only."""
    source_ids = hybrid_search(question)
    sources = [d for d in documents if d["id"] in source_ids]
    context = "\n\n".join(f"[{d['id']}] {d['title']}: {d['text']}" for d in sources)
    prompt = f"""You are a telecom billing support assistant.
Answer the customer's question using ONLY the policies below.
If the policies do not answer the question, reply exactly: "{REFUSAL}"

Policies:
{context}

Customer question:
{question}

Answer:"""
    return ask_llm(prompt), source_ids, context


print("RAG assistant ready with", len(documents), "policies")


# =====================================================================
# STEP 2: Write a test set
# =====================================================================
# A test set is a list of questions where WE know the right answer.
# For each question we write down:
#   right_doc   - the policy that holds the answer (None if no policy does)
#   must_say    - short facts the answer must contain
#
# Two questions are deliberately NOT covered by the policies. A good
# assistant must refuse them instead of inventing an answer.

print("\n========== STEP 2: the test set ==========\n")

test_set = [
    {"question": "I was billed twice for the same month. How long does the refund take?",
     "right_doc": "DOC-01", "must_say": ["5 to 7 business days"]},
    {"question": "What does autopay error ERR-4019 mean?",
     "right_doc": "DOC-06", "must_say": ["expired"]},
    {"question": "Can my first late payment fee be waived?",
     "right_doc": "DOC-04", "must_say": ["twelve months"]},
    {"question": "How much is the reconnection fee?",
     "right_doc": "DOC-08", "must_say": ["20 dollars"]},
    {"question": "How much does plan UNL-599 cost?",
     "right_doc": "DOC-10", "must_say": ["599"]},
    {"question": "How do I reset my voicemail PIN?",
     "right_doc": None, "must_say": [REFUSAL]},
    {"question": "Which phone models are on sale this week?",
     "right_doc": None, "must_say": [REFUSAL]},
]

for i, test in enumerate(test_set, start=1):
    print(f"{i}. {test['question']}  ->  expected policy: {test['right_doc'] or 'none (must refuse)'}")


# =====================================================================
# STEP 3: Evaluate retrieval
# =====================================================================
# If search does not find the right policy, the model cannot answer well,
# however good it is. So we measure search on its own first.
# Hit rate = how many questions had the right policy in the top 3.

print("\n========== STEP 3: retrieval hit rate ==========\n")

answerable = [t for t in test_set if t["right_doc"]]
for name, search in [("keyword", keyword_search), ("similarity", similarity_search), ("hybrid", hybrid_search)]:
    hits = sum(t["right_doc"] in search(t["question"]) for t in answerable)
    print(f"{name:10}  found the right policy for {hits} of {len(answerable)} questions")


# =====================================================================
# STEP 4: Evaluate answers with simple rules
# =====================================================================
# The cheapest check: does the answer contain the facts it must contain?
# Rules are fast, free and always give the same result. They cannot judge
# wording, so we add a second check in STEP 5.

print("\n========== STEP 4: rule checks ==========\n")

results = []
for test in test_set:
    answer, source_ids, context = rag_answer(test["question"])
    rule_pass = all(fact.lower() in answer.lower() for fact in test["must_say"])
    results.append({
        "question": test["question"],
        "right_doc": test["right_doc"] or "none",
        "found_doc": (test["right_doc"] in source_ids) if test["right_doc"] else "n/a",
        "answer": answer,
        "context": context,
        "rule_pass": rule_pass,
    })
    print(("PASS" if rule_pass else "FAIL"), "-", test["question"])
    if not rule_pass:
        print("       expected to contain:", test["must_say"])
        print("       answer was        :", answer[:150])


# =====================================================================
# STEP 5: Evaluate answers with an LLM judge
# =====================================================================
# A second model call reads the question, the policies and the answer,
# and gives scores. This catches problems rules miss, such as an answer
# that adds a promise the policy never made.
#
#   faithful (1-5): does the answer say ONLY what the policies support?
#   helpful  (1-5): does it actually answer the customer's question?
#
# The RAGAS library (used in Ex4_rag_reranking_ragas_langsmith.py)
# automates this kind of judging with ready-made metrics.

print("\n========== STEP 5: LLM judge ==========\n")

JUDGE_PROMPT = """You are checking an answer written by a billing support assistant.

Policies the assistant was given:
{context}

Customer question:
{question}

Assistant answer:
{answer}

Score the answer:
- faithful: 5 if every statement is supported by the policies, 1 if it invents facts.
  If the policies do not cover the question and the assistant said so, that is faithful (5).
- helpful: 5 if it fully answers the question (or correctly says the information is not available), 1 if not.

Reply with JSON only, in this exact form:
{{"faithful": <1-5>, "helpful": <1-5>, "reason": "<one short sentence>"}}"""

for row in results:
    reply = ask_llm(JUDGE_PROMPT.format(context=row["context"], question=row["question"], answer=row["answer"]))
    try:
        scores = json.loads(reply[reply.find("{"): reply.rfind("}") + 1])     # keep only the JSON part
    except ValueError:
        scores = {"faithful": 0, "helpful": 0, "reason": "judge reply was not valid JSON"}
    row["faithful"] = scores.get("faithful", 0)
    row["helpful"] = scores.get("helpful", 0)
    row["judge_reason"] = scores.get("reason", "")
    print(f"faithful {row['faithful']}  helpful {row['helpful']}  - {row['question'][:55]}")
    print("   judge says:", row["judge_reason"])


# =====================================================================
# STEP 6: The report card and a release decision
# =====================================================================
# Turn the checks into a few numbers, save every detail to a CSV file
# (open it in Excel), and decide: is this good enough to release?

print("\n========== STEP 6: report card ==========\n")

total = len(results)
rule_rate = sum(r["rule_pass"] for r in results) / total
avg_faithful = sum(r["faithful"] for r in results) / total
avg_helpful = sum(r["helpful"] for r in results) / total
retrieval_rows = [r for r in results if r["found_doc"] != "n/a"]
retrieval_rate = sum(r["found_doc"] for r in retrieval_rows) / len(retrieval_rows)

print(f"Retrieval hit rate : {retrieval_rate:.0%}")
print(f"Rule checks passed : {rule_rate:.0%}")
print(f"Average faithful   : {avg_faithful:.1f} / 5")
print(f"Average helpful    : {avg_helpful:.1f} / 5")

with open("evaluation_report.csv", "w", newline="", encoding="utf-8") as f:
    columns = ["question", "right_doc", "found_doc", "rule_pass", "faithful", "helpful", "judge_reason", "answer"]
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
print("\nEvery question, answer and score saved to evaluation_report.csv")

# A "quality gate": agree the minimum scores BEFORE testing, then let the
# numbers decide. Run this file again after every prompt or data change.
GATE = {"rule_rate": 0.80, "avg_faithful": 4.0}
ready = rule_rate >= GATE["rule_rate"] and avg_faithful >= GATE["avg_faithful"]
print("\nRelease decision:", "READY" if ready else "NOT READY",
      f"(needs rule checks >= {GATE['rule_rate']:.0%} and faithful >= {GATE['avg_faithful']})")
