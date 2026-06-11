import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_embeddings(query, embeddings_store, top_k=3):
    query_embedding = get_embedding(query)

    results = []

    for item in embeddings_store:
        score = cosine_similarity(query_embedding, item["embedding"])

        results.append({
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "score": score
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


with open("embeddings_store.json", "r", encoding="utf-8") as f:
    embeddings_store = json.load(f)

query = input("Enter your question: ")

results = search_embeddings(query, embeddings_store, top_k=3)

print(f"\nQuery: {query}\n")

for result in results:
    print("=" * 80)
    print(f"Source: {result['source']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Similarity Score: {result['score']:.4f}")
    print("-" * 80)
    print(result["text"][:1000])
    print()