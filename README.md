# Semantic Search System

This project implements a semantic search system using:

- Sentence Transformers for embeddings
- FAISS for vector search
- Fuzzy C-Means clustering
- Semantic caching
- FastAPI for serving queries
- Docker for containerization

## API Endpoints

POST /query  
Returns semantic search results with cache support.

GET /cache/stats  
Returns cache statistics.

DELETE /cache  
Clears cache.

## Run locally

uvicorn api:app --reload

## Run with Docker

docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
