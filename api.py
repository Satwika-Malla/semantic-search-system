from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from embeddings import documents
from vector_db import index
from cache import cache
from clustering import membership
import numpy as np

app = FastAPI()

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# cache statistics
hit_count = 0
miss_count = 0


# -----------------------------
# Home endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Semantic Search API is running"}


# -----------------------------
# Query endpoint
# -----------------------------
@app.post("/query")
def query_api(query: dict):

    global hit_count, miss_count

    query_text = query["query"]

    # check cache
    if query_text in cache:

        hit_count += 1

        cached_data = cache[query_text]

        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": query_text,
            "similarity_score": cached_data["similarity_score"],
            "result": cached_data["result"],
            "dominant_cluster": cached_data["dominant_cluster"]
        }

    # -------------------------
    # cache miss
    # -------------------------

    miss_count += 1

    # convert query to embedding
    query_embedding = model.encode([query_text])

    # search FAISS
    D, I = index.search(query_embedding, 5)

    # best document
    doc_index = I[0][0]

    # compute dominant cluster
    dominant_cluster = int(np.argmax(membership[:, doc_index]))

    # extract result text
    result_text = documents[doc_index][:200]

    # compute similarity
    similarity = round(1 / (1 + float(D[0][0])), 2)

    # store in cache
    cache[query_text] = {
        "result": result_text,
        "similarity_score": similarity,
        "dominant_cluster": dominant_cluster
    }
    matched_query = documents[doc_index][:100]
    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": matched_query,
        "similarity_score": similarity,
        "result": result_text,
        "dominant_cluster": dominant_cluster
    }


# -----------------------------
# Cache statistics endpoint
# -----------------------------
@app.get("/cache/stats")
def cache_stats():

    total_requests = hit_count + miss_count

    hit_rate = hit_count / total_requests if total_requests > 0 else 0

    return {
        "total_entries": len(cache),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": round(hit_rate, 3)
    }


# -----------------------------
# Clear cache endpoint
# -----------------------------
@app.delete("/cache")
def clear_cache():

    global hit_count, miss_count

    cache.clear()

    hit_count = 0
    miss_count = 0

    return {
        "message": "Cache cleared"
    }
