# Shopnexus Recommender Service

Simple Flask service for product search and recommendations powered by Milvus and MGTE embeddings.

## Description

This service provides:
- Semantic product search using MGTE hybrid embeddings (dense + sparse) stored in Milvus.
- Personalized recommendations by fusing content embeddings with collaborative filtering (CF) vectors.
- Real-time account vector updates from analytics events.

The fusion is a simple, stable concatenation of content and CF vectors with optional L2 normalization, enabling cosine/IP similarity search.

## Architecture

- Flask API (`src/app.py`, `src/routes.py`)
- Service layer (`src/service.py`):
  - `EmbeddingService` (MGTE) for dense+sparse content vectors
  - `CFModel` (TensorFlow) for user/item CF embeddings
  - `EmbeddingFusion` (NumPy/Torch) to concatenate content+CF into a fused vector
  - `MilvusClient` to manage collections, hybrid search, and upserts

Flows:
- Search: encode query → hybrid search (sparse+content) on `products` → top ids
- Recommend: fetch `accounts.fused_vector` → search `products.fused_vector`; if missing, fall back to content search
- Events: process events → update `accounts.fused_vector`

## Prerequisites

- uv (for Python package management)
- Python 3.10+
- Docker (for Milvus)

## Setup

1) Install deps (uv):

```bash
uv sync
```

2) Start Milvus locally:

```bash
docker-compose -f docker-compose.yml -p shopnexus-recommender-system up -d
```

3) Run the API:

```bash
uv run src/app.py
```

Server runs at http://localhost:8000

## API

- POST /search
  - Body: {"query": "text", "limit": 10, "offset": 0, "weights": {"dense": 1.0, "sparse": 1.0}}
  - Returns top products by hybrid (dense+sparse) similarity

- GET /recommend?account_id=ID&limit=10
  - Returns products similar to the account fused vector; falls back to content if missing

- POST /events
  - Body: {"events": [{"account_id": 1, "event_type": "purchase", "ref_type": "Product", "ref_id": 10, "date_created": "2025-01-21T14:32:10Z"}]}
  - Updates account vectors from interactions

- POST /products
  - Body: {"products": [...], "metadata_only": false}
  - Upserts product metadata and embeddings

- POST /train
  - Resume/trigger CF training (experimental)

- GET /health
  - Basic health info

## Notes

- Milvus is expected at localhost:19530 (see docker-compose.yml)
- CF model uses TensorFlow; fusion uses NumPy/Torch
- Event weights and decay config live in `src/config.py`
