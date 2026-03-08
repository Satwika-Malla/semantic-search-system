import skfuzzy as fuzz
from embeddings import embeddings

# transpose embeddings for clustering
data = embeddings.T

# choose number of clusters
clusters = 10

# apply fuzzy c-means clustering
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data,
    c=clusters,
    m=2,
    error=0.005,
    maxiter=1000,
    seed=42
)

membership = u

print("Fuzzy clustering completed")
print("Number of clusters:", clusters)