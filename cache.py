from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# simple dictionary cache
cache = {}

def get_cached_result(query):

    query_embedding = model.encode([query])

    for cached_query, result in cache.items():

        cached_embedding = model.encode([cached_query])

        similarity = cosine_similarity(query_embedding, cached_embedding)[0][0]

        if similarity > 0.9:
            return result

    return None


def store_cache(query, result):
    cache[query] = result