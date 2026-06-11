import os
import json
import numpy as np
from pathlib import Path
from PyPDF2 import PdfReader
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()



# -----------------------------
# Step 1: Setup
# -----------------------------

load_dotenv()
client = OpenAI()

PDF_FILE = "ai_hleg_ethics_guidelines_for_trustworthy_ai-en_87F84A41-A6E8-F38C-BFF661481B40077B_60419.pdf"
TRANSCRIPT_FILE = "podcast_transcript.txt"


# -----------------------------
# Load PDF
# -----------------------------

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Load transcript
# -----------------------------

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Simple token-based chunking
# -----------------------------

def chunk_text_by_tokens(text, chunk_size=500, overlap=50):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)

        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Step 2: Generate embeddings
# -----------------------------

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response.data[0].embedding


def embed_chunks(chunks, source_name):
    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding {source_name} chunk {i + 1}/{len(chunks)}")

        embedding = get_embedding(chunk)

        embedded_chunks.append({
            "source": source_name,
            "chunk_id": i,
            "text": chunk,
            "embedding": embedding
        })

    return embedded_chunks


# -----------------------------
# Main script
# -----------------------------
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def vector_search(query, embeddings_store, top_k=3):
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


def rag_query(query, embeddings_store, top_k=3):
    retrieved_chunks = vector_search(query, embeddings_store, top_k=top_k)

    context = "\n\n---\n\n".join(
        [
            f"Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']}\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG assistant. Answer the user's question "
                    "using only the provided context. If the answer is not in the context, "
                    "say that the documents do not provide enough information."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }
if __name__ == "__main__":

    if os.path.exists("embeddings_store.json"):
        print("Loading existing embeddings...")
        with open("embeddings_store.json", "r", encoding="utf-8") as f:
            embeddings_store = json.load(f)
    else:
        pdf_text = load_pdf_text(PDF_FILE)
        podcast_text = load_text_file(TRANSCRIPT_FILE)

        print("PDF characters:", len(pdf_text))
        print("Podcast characters:", len(podcast_text))

        pdf_chunks = chunk_text_by_tokens(pdf_text, chunk_size=500, overlap=50)
        podcast_chunks = chunk_text_by_tokens(podcast_text, chunk_size=500, overlap=50)

        print("PDF chunks:", len(pdf_chunks))
        print("Podcast chunks:", len(podcast_chunks))

        pdf_embeddings = embed_chunks(pdf_chunks, "PDF")
        podcast_embeddings = embed_chunks(podcast_chunks, "Podcast")

        embeddings_store = pdf_embeddings + podcast_embeddings

        with open("embeddings_store.json", "w", encoding="utf-8") as f:
            json.dump(embeddings_store, f)

        print("Saved embeddings to embeddings_store.json")
        print("Total embedded chunks:", len(embeddings_store))

    query = "What are the seven requirements for trustworthy AI?"

    result = rag_query(query, embeddings_store, top_k=3)

    print("\nQUESTION:")
    print(result["query"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED CHUNKS:")
    for chunk in result["retrieved_chunks"]:
        print("-" * 80)
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Score: {chunk['score']:.4f}")
        print(chunk["text"][:500])