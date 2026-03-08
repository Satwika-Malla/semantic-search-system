import faiss
import numpy as np
from embeddings import embeddings

# find the size of each embedding vector
dimension = embeddings.shape[1]

# create FAISS index
index = faiss.IndexFlatL2(dimension)

# add embeddings into the index
index.add(np.array(embeddings))

print("Vector database created successfully")
print("Total vectors stored:", index.ntotal)