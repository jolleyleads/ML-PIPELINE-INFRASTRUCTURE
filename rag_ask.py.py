import json
import urllib.request
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path("chroma_db")
COLLECTION_NAME = "jobs"
TOP_K = 6
OLLAMA_MODEL = "llama3.2:3b"  # the model you successfully pulled

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

def ollama_generate(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()

def salary_line(m: dict) -> str:
    smin, smax = m.get("salary_min"), m.get("salary_max")
    cur, per = m.get("salary_currency"), m.get("salary_period")
    if smin is None and smax is None:
        return "Not listed"
    return f"{smin}–{smax} {cur} ({per})"

def compact_result(i: int, meta: dict, text: str) -> str:
    url = meta.get("official_url") or meta.get("platform_url") or ""
    skills = meta.get("skills") or ""
    if isinstance(skills, str) and len(skills) > 180:
        skills = skills[:180] + "..."
    snippet = (text or "").replace("\n", " ").strip()
    if len(snippet) > 650:
        snippet = snippet[:650] + "..."
    return (
        f"RESULT {i}\n"
        f"Title: {meta.get('title')}\n"
        f"Company: {meta.get('company_name')}\n"
        f"Remote: {meta.get('is_remote')} | Location: {meta.get('location')}\n"
        f"Level: {meta.get('experience_level')} | Type: {meta.get('job_type')}\n"
        f"Salary: {salary_line(meta)}\n"
        f"Skills: {skills}\n"
        f"URL: {url}\n"
        f"Snippet: {snippet}\n"
    )

SYSTEM_RULES = """
You are a job-matching assistant for ML/AI roles.
Use ONLY the provided results. Do not invent jobs.
If results are not ML/AI, say so clearly and suggest a better query.
Return:
1) Top 3 matches with: Title — Company | Remote/Location | Level | Salary | URL
2) Why each match fits (2-4 bullets)
3) Missing skills/keywords (if any)
4) One improved search query to run next
""".strip()

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

    context = "\n".join(compact_result(i + 1, metas[i], docs[i]) for i in range(len(docs)))

    prompt = f"""
{SYSTEM_RULES}

USER QUERY: {q}

RESULTS:
{context}
""".strip()

    answer = ollama_generate(prompt)
    print("\n" + "=" * 90)
    print(answer)
