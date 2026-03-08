from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

# load the dataset
data = fetch_20newsgroups(subset="all")

documents = data.data

print("Total documents:", len(documents))

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# convert documents to embeddings
embeddings = model.encode(documents)

print("Embedding shape:", embeddings.shape)
