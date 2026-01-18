import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "jobs"
DEFAULT_TOP_K = 6
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# -----------------------------
# App + shared resources
# -----------------------------
app = FastAPI(title="MLE Job RAG API")

# Load embedder once (fast + consistent)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma client (persistent)
client = chromadb.PersistentClient(path=str(DB_DIR))

# -----------------------------
# Pydantic models
# -----------------------------
class AskRequest(BaseModel):
    query: str  # The search query
    top_k: int = DEFAULT_TOP_K  # Default to 6 results
    model: str = DEFAULT_OLLAMA_MODEL  # Default model for Ollama

class AskResponse(BaseModel):
    answer: str
    matches: List[Dict[str, Any]]

# -----------------------------
# Helper functions
# -----------------------------
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def ollama_generate(prompt: str, model_name: str) -> str:
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()

def format_match(meta: Dict[str, Any], doc: str, dist: float) -> Dict[str, Any]:
    return {
        "title": meta.get("title"),
        "company_name": meta.get("company_name"),
        "location": meta.get("location"),
        "is_remote": meta.get("is_remote"),
        "experience_level": meta.get("experience_level"),
        "job_type": meta.get("job_type"),
        "salary_min": meta.get("salary_min"),
        "salary_max": meta.get("salary_max"),
        "salary_currency": meta.get("salary_currency"),
        "salary_period": meta.get("salary_period"),
        "official_url": meta.get("official_url") or meta.get("platform_url"),
        "skills": meta.get("skills"),
        "distance": dist,
        "snippet": (doc or "")[:700],
    }

# -----------------------------
# System Rules
# -----------------------------
SYSTEM_RULES = (
    "You are a job-matching assistant for ML/AI roles.\n"
    "Use ONLY the provided results. Do not invent jobs.\n"
    "If results are not ML/AI, say so clearly and suggest a better query.\n"
    "Return:\n"
    "1) Top 3 matches with: Title — Company | Remote/Location | Level | Salary | URL\n"
    "2) Why each match fits (2-4 bullets)\n"
    "3) Missing skills/keywords (if any)\n"
    "4) One improved search query to run next\n"
)

def render_result(i: int, m: dict) -> str:
    return (
        f"RESULT {i}\n"
        f"Title: {safe_str(m.get('title'))}\n"
        f"Company: {safe_str(m.get('company_name'))}\n"
        f"Remote: {safe_str(m.get('is_remote'))} | Location: {safe_str(m.get('location'))}\n"
        f"Level: {safe_str(m.get('experience_level'))} | Type: {safe_str(m.get('job_type'))}\n"
        f"Salary: {safe_str(m.get('salary_min'))}-{safe_str(m.get('salary_max'))} {safe_str(m.get('salary_currency'))} ({safe_str(m.get('salary_period'))})\n"
        f"URL: {safe_str(m.get('official_url'))}\n"
        f"Snippet: {safe_str(m.get('snippet'))}\n"
    )

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "db_dir": str(DB_DIR), "collection": COLLECTION_NAME}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1) Get collection (fail clean if not indexed yet)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Chroma collection '{COLLECTION_NAME}' not found. Run indexing first. ({e})"
        )

    # 2) Embed query
    q_emb = embedder.encode([req.query]).tolist()

    # 3) Retrieve top-k
    res = collection.query(
        query_embeddings=q_emb,
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    matches = [format_match(metas[i], docs[i], dists[i]) for i in range(len(docs))]

    # 4) Build grounded context
    context = "\n".join(render_result(i + 1, matches[i]) for i in range(len(matches)))

    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"USER QUERY: {req.query}\n\n"
        f"RESULTS:\n{context}\n"
    )

    # 5) Generate answer
    answer = ollama_generate(prompt, req.model)

    # 6) Return clean JSON (this makes Swagger show a real schema)
    return AskResponse(answer=answer, matches=matches)
