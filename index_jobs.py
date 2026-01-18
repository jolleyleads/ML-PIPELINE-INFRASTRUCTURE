import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("job_chunks.jsonl")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "jobs"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def flatten_metadata(meta: dict) -> dict:
    """
    Chroma metadata must be ONLY: str, int, float, bool.
    No lists, no dicts, no None.
    """
    meta = dict(meta or {})

    # skills list -> string
    skills = meta.get("skills")
    if isinstance(skills, list):
        meta["skills"] = "; ".join([str(x) for x in skills])

    # flags dict -> separate booleans
    flags = meta.get("flags")
    if isinstance(flags, dict):
        meta["us_only"] = bool(flags.get("us_only")) if flags.get("us_only") is not None else False
        meta["requires_clearance"] = bool(flags.get("requires_clearance")) if flags.get("requires_clearance") is not None else False
        meta["visa_sponsorship_unavailable"] = bool(flags.get("visa_sponsorship_unavailable")) if flags.get("visa_sponsorship_unavailable") is not None else False
        meta.pop("flags", None)

    # Remove None values entirely (Chroma can choke on None)
    meta = {k: v for k, v in meta.items() if v is not None}

    # Convert leftover lists/dicts to strings (safety)
    for k, v in list(meta.items()):
        if isinstance(v, (list, dict)):
            meta[k] = str(v)

    # Final safety: keep only primitives
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned

ids = []
documents = []
metadatas = []

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        ids.append(row["chunk_id"])
        documents.append(row["text"])
        metadatas.append(flatten_metadata(row["metadata"]))

embeddings = model.encode(documents, show_progress_bar=True).tolist()

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"Indexed {len(ids)} chunks into Chroma at {DB_DIR} (collection: {COLLECTION_NAME})")


