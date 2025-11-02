# Shopnexus Recommender Service

A hybrid recommendation system that combines **semantic search** (content embeddings) and **collaborative filtering** (behavioral embeddings) using Milvus vector database with MGTEEmbeddingFunction.

## Features

- **Semantic Search**: Hybrid search with dense and sparse vectors for content-based product search
- **Hybrid Recommendations**: Combines content embeddings (MGTE) and collaborative filtering embeddings for personalized recommendations
- **Collaborative Filtering**: Matrix Factorization model trained on user-item interactions
- **Customer Analytics**: Process events and create user preference vectors
- **Fused Embeddings**: Combines content and CF embeddings into 768-dimensional vectors for recommendations

## Architecture

The system uses **three Milvus collections**:

1. **`content_products`**: Products with content embeddings (dense + sparse) for semantic search
2. **`hybrid_products`**: Products with fused embeddings (768d) combining content + CF embeddings for recommendations
3. **`hybrid_customers`**: Users with fused embeddings (768d) combining CF + recent content embeddings

### Embedding Fusion

- **Item embeddings**: `fused = Linear(cat([MGTE_dense, CF_embedding]), 768)`
- **User embeddings**: `fused = Linear(cat([recent_MGTE_avg, CF_embedding]), 768)`

## Installation

1. Install dependencies with uv:

```bash
uv sync
```

2. Start Milvus using Docker Compose:

```bash
docker-compose up -d
```

This will start Milvus on `localhost:19530`

3. Start the service:

```bash
cd src
python app.py
```

This will start a Flask server on `http://localhost:8000`

## Quick Start

### 1. Train the CF Model (Required for Recommendations)

The model needs to be trained before recommendations will work. Training data is automatically collected from analytics events.

**Step 1**: Send some analytics events to collect training data:

```bash
curl -X POST http://localhost:8000/analytics/process \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "id": 1,
        "account_id": 101,
        "event_type": "purchase",
        "ref_type": "Product",
        "ref_id": 5001,
        "date_created": "2025-01-21T14:32:10.123Z"
      },
      {
        "id": 2,
        "account_id": 101,
        "event_type": "add_to_cart",
        "ref_type": "Product",
        "ref_id": 5002,
        "date_created": "2025-01-21T14:33:10.123Z"
      }
    ]
  }'
```

**Step 2**: Train the CF model:

```bash
curl -X POST http://localhost:8000/training/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 50,
    "batch_size": 256,
    "embedding_dim": 64
  }'
```

After training completes, the system will:
- Generate CF embeddings for users and items
- Create fused embeddings and store them in `hybrid_products` and `hybrid_customers`
- Enable personalized recommendations

### 2. Use Semantic Search (Works Immediately)

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "gaming laptop",
    "limit": 10,
    "weights": {"dense": 1.0, "sparse": 0.7}
  }'
```

### 3. Get Recommendations (After Training)

```bash
curl http://localhost:8000/user/101/recommendations?limit=5
```

## API Endpoints

### Product Semantic Search

Performs hybrid search in `content_products` collection using text query.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "laptop computer",
    "limit": 10,
    "offset": 0,
    "weights": {"dense": 1.0, "sparse": 0.7}
  }'
```

**Response**:
```json
[
  {"id": 5001, "score": 0.95},
  {"id": 5002, "score": 0.89}
]
```

### Get User Recommendations

Returns personalized product recommendations based on user's fused embedding.

```bash
curl http://localhost:8000/user/101/recommendations?limit=5
```

**Response**:
```json
[
  {"id": 5003, "score": 0.92},
  {"id": 5004, "score": 0.87}
]
```

**Note**: Returns empty list if user has no vector (model not trained or user not in training data).

### Train CF Model

Trains the collaborative filtering model on ingested training data and generates fused embeddings.

```bash
curl -X POST http://localhost:8000/training/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 50,
    "batch_size": 256,
    "embedding_dim": 64
  }'
```

**Parameters**:
- `epochs` (optional): Number of training epochs (default: 50)
- `batch_size` (optional): Batch size for training (default: 256)
- `embedding_dim` (optional): Dimension of CF embeddings (default: 64)

**Response**:
```json
{
  "message": "CF model training completed",
  "processed_at": "2025-01-21T14:35:10.123Z",
  "training_history": {
    "loss": [...],
    "accuracy": [...]
  },
  "performance": {
    "training_time_seconds": 45.23,
    "total_time_seconds": 45.25
  }
}
```

### Process Analytics Events

Processes user events and automatically ingests training data for CF model. Also updates user embeddings in real-time.

```bash
curl -X POST http://localhost:8000/analytics/process \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "id": 1,
        "account_id": 101,
        "event_type": "purchase",
        "ref_type": "Product",
        "ref_id": 5001,
        "date_created": "2025-01-21T14:32:10.123Z"
      }
    ]
  }'
```

**Event Types**: `purchase`, `add_to_cart`, `view`, `add_to_favorites`, `write_review`, `rating_high`, etc.

### Health Check

Returns service health status and collection entity counts.

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-21T14:35:10.123Z",
  "collections": {
    "content_products": 1250,
    "hybrid_products": 1200,
    "hybrid_customers": 500
  }
}
```

## Event Types and Weights

The system uses different weights for different event types when calculating user preferences:

**High Intent Events**:
- `purchase`: 1.0
- `repeat_purchase`: 1.2
- `add_to_cart`: 0.5
- `add_to_favorites`: 0.6

**Social & Validation**:
- `write_review`: 0.5
- `rating_high` (4-5 stars): 0.4
- `rating_medium` (3 stars): 0.1
- `ask_question`: 0.25

**Discovery Events**:
- `click_from_recommendation`: 0.15
- `click_from_search`: 0.2
- `click_from_category`: 0.12
- `view_similar_products`: 0.15

**Negative Signals**:
- `remove_from_cart`: -0.25
- `return_product`: -0.4
- `rating_low` (1-2 stars): -0.4
- `report_product`: -1.0

See `src/config.py` for complete list of event weights.

## Architecture Details

### Collections

1. **content_products** (`content_products`)
   - Fields: `id`, `name`, `dense_vector`, `sparse_vector`
   - Purpose: Semantic search using hybrid search (dense + sparse)
   - Index: AUTOINDEX on dense_vector, SPARSE_INVERTED_INDEX on sparse_vector

2. **hybrid_products** (`hybrid_products`)
   - Fields: `id`, `metadata`, `dense_vector` (768d)
   - Purpose: Product recommendations using fused embeddings
   - Index: AUTOINDEX on dense_vector

3. **hybrid_customers** (`hybrid_customers`)
   - Fields: `id`, `dense_vector` (768d)
   - Purpose: User query vectors for recommendations
   - Index: AUTOINDEX on dense_vector

### Training Workflow

1. **Data Collection**: Events are automatically ingested as training data via `/analytics/process`
2. **Training**: Call `/training/train` to train CF model
3. **Embedding Generation**: System generates:
   - CF embeddings for users and items
   - Fused embeddings by combining MGTE content embeddings with CF embeddings
4. **Storage**: Fused embeddings stored in `hybrid_products` and `hybrid_customers`

### Recommendation Flow

1. User queries `/user/<id>/recommendations`
2. System retrieves user's fused vector from `hybrid_customers`
3. Performs dense vector search in `hybrid_products` collection
4. Returns top-K most similar products

### Search Flow

1. User queries `/search` with text
2. System encodes text using MGTEEmbeddingFunction (dense + sparse)
3. Performs hybrid search in `content_products` collection
4. Returns semantically similar products

## Configuration

Key parameters in `Service` class:

- `milvus_host` and `milvus_port`: Milvus connection (default: localhost:19530)
- `event_weights`: Event type weights (see `src/config.py`)
- `update_weight`: Vector blending weight for real-time updates (0.5)
- `fused_dim`: Fused embedding dimension (768)
- `MAX_LENGTH_EMBED`: Max text length for embeddings (4048)
- `DESCRIPTION_LENGTH`: Max description length (10240)

## Dependencies

- Flask >= 3.1.2
- TensorFlow >= 2.20.0 (for CF model)
- PyTorch >= 2.8.0 (for fusion layer)
- pymilvus >= 2.6.2
- pymilvus-model >= 0.3.2 (for MGTEEmbeddingFunction)
- numpy >= 2.2.6

## Notes

- **Semantic search works immediately** (doesn't require training)
- **Recommendations require training** first via `/training/train`
- Training data is automatically collected from analytics events
- The CF model uses Matrix Factorization with configurable embedding dimensions
- Fused embeddings combine content (MGTE) and behavioral (CF) signals for hybrid recommendations
