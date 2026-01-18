import json
import urllib.request
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "jobs"
TOP_K = 6

# You confirmed this model exists locally
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# -----------------------------
# STEP 8: Filtering + Ranking
# -----------------------------
TARGET_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "mlops", "mle", "ai engineer",
    "llm", "rag", "retrieval", "embedding", "inference", "serving"
]

NEGATIVE_TITLE_KEYWORDS = [
    "onboarding", "customer support", "support", "sales", "recruiter",
    "account executive", "specialist", "call center", "collections"
]

BOOST_KEYWORDS = [
    "fastapi", "api", "rest", "inference", "serving", "deployment",
    "rag", "retrieval", "vector", "embedding", "chroma", "llm", "pytorch"
]

SYSTEM_RULES = """
You are a job-matching assistant for ML/AI roles.
Use ONLY the provided results. Do not invent jobs.
The results are already ranked best-first. Do not reorder them.

If results are not truly ML/AI, say so clearly and suggest a better query.

Return:
1) Top 3 matches with: Title — Company | Remote/Location | Level | Salary | URL
2) Why each match fits (2-4 bullets each)
3) Missing skills/keywords (if any)
4) One improved search query to run next
""".strip()


def safe_text(x) -> str:
    return (x or "").strip()


def looks_like_target_role(meta: dict) -> bool:
    title = safe_text(meta.get("title")).lower()
    if any(k in title for k in NEGATIVE_TITLE_KEYWORDS):
        return False
    return any(k in title for k in TARGET_TITLE_KEYWORDS)


def keyword_score(meta: dict, doc: str, query: str) -> int:
    title = safe_text(meta.get("title")).lower()
    skills = safe_text(meta.get("skills")).lower()
    text = safe_text(doc).lower()
    q = safe_text(query).lower()

    score = 0

    # Title signals
    for k in TARGET_TITLE_KEYWORDS:
        if k in title:
            score += 5

    # Boost keywords anywhere (title/skills/text)
    blob = " ".join([title, skills, text])
    for k in BOOST_KEYWORDS:
        if k in blob:
            score += 2

    # Reward matching query tokens
    for token in q.split():
        if len(token) >= 4 and token in blob:
            score += 1

    # Remote preference
    if meta.get("is_remote") is True:
        score += 2

    return score


def format_salary(m: dict) -> str:
    smin = m.get("salary_min")
    smax = m.get("salary_max")
    cur = m.get("salary_currency")
    per = m.get("salary_period")

    if smin is None and smax is None:
        return "Not listed"

    # make it nice if one side missing
    if smin is None and smax is not None:
        return f"Up to {smax} {cur} ({per})"
    if smin is not None and smax is None:
        return f"From {smin} {cur} ({per})"

    return f"{smin}–{smax} {cur} ({per})"


def ollama_generate(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


# -----------------------------
# INIT MODELS + DB
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(DB_DIR))

# safer than get_collection only:
try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = client.create_collection(name=COLLECTION_NAME)


def build_context(top_items, query: str) -> str:
    blocks = []
    for rank, (final, sim, kscore, meta, doc) in enumerate(top_items, start=1):
        url = meta.get("official_url") or meta.get("platform_url") or ""
        title = meta.get("title") or ""
        company = meta.get("company_name") or ""
        remote = meta.get("is_remote")
        location = meta.get("location") or ""
        level = meta.get("experience_level") or ""
        job_type = meta.get("job_type") or ""
        salary = format_salary(meta)

        snippet = (doc or "").replace("\n", " ").strip()
        if len(snippet) > 650:
            snippet = snippet[:650] + "..."

        blocks.append(
            f"RESULT {rank}\n"
            f"Title: {title}\n"
            f"Company: {company}\n"
            f"Remote: {remote} | Location: {location}\n"
            f"Level: {level} | Type: {job_type}\n"
            f"Salary: {salary}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}\n"
            f"Scores: final={final:.2f}, sim={sim:.3f}, keyword={kscore}\n"
        )
    return "\n".join(blocks)


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    q = input("\nAsk (or type exit): ").strip()
    if q.lower() == "exit":
        break
    if not q:
        continue

    q_emb = model.encode([q]).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # Step 8: filter + re-rank
    candidates = []
    for i in range(len(docs)):
        meta = metas[i]
        doc = docs[i]
        dist = dists[i]  # lower = better

        if not looks_like_target_role(meta):
            continue

        kscore = keyword_score(meta, doc, q)
        sim = 1.0 / (1.0 + float(dist))
        final = (sim * 10.0) + kscore

        candidates.append((final, sim, kscore, meta, doc))

    # If filter removed everything, fall back to unfiltered (still ranked)
    if not candidates:
        for i in range(len(docs)):
            meta = metas[i]
            doc = docs[i]
            dist = dists[i]

            kscore = keyword_score(meta, doc, q)
            sim = 1.0 / (1.0 + float(dist))
            final = (sim * 10.0) + kscore
            candidates.append((final, sim, kscore, meta, doc))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:TOP_K]

    context = build_context(top, q)

    prompt = f"""
{SYSTEM_RULES}

USER QUERY:
{q}

RESULTS (ranked best-first):
{context}
""".strip()

    print("\n" + "=" * 90)
    answer = ollama_generate(prompt)
    print(answer)
