from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path("chroma_db")
COLLECTION_NAME = "jobs"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

while True:
    question = input("\nAsk a job question (or type exit): ").strip()
    if question.lower() == "exit":
        break

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]

        print("\n" + "=" * 80)
        print(f"{meta.get('title')} — {meta.get('company_name')}")
        print(f"Location: {meta.get('location')} | Remote: {meta.get('is_remote')}")
        print(f"Level: {meta.get('experience_level')} | Type: {meta.get('job_type')}")
        print(
            f"Salary: {meta.get('salary_min')}–{meta.get('salary_max')} "
            f"{meta.get('salary_currency')} ({meta.get('salary_period')})"
        )
        print(f"URL: {meta.get('official_url') or meta.get('platform_url')}")
        print(f"Distance: {dist}")
        print("-" * 80)
        print(doc[:800])
