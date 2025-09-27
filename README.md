# Shopnexus Recommender Service

A service that combines product search and customer analytics using Milvus vector database with MGTEEmbeddingFunction.

## Features

- **Product Search**: Hybrid search with dense and sparse vectors
- **Customer Analytics**: Process events and create user preference vectors
- **Product Recommendations**: Content-based recommendations
- **Product Management**: Batch update products with metadata and embeddings

## Installation

1. Install dependencies with uv:

```bash
uv sync
```

2. Make sure Milvus is running (localhost:19530)

## Usage

### Running the Service

To start the REST API server:

```bash
python service.py
```

This will start a Flask server on `http://localhost:8000`

### Testing

Test the service using the provided endpoints with curl commands or any HTTP client.

## API Endpoints

### Product Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "laptop computer",
    "limit": 10,
    "weights": {"dense": 1.0, "sparse": 0.7}
  }'
```

### Process Analytics Events

```bash
curl -X POST http://localhost:8000/analytics/process \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "id": 1,
        "account_id": 101,
        "event_type": "view",
        "ref_type": "Product",
        "ref_id": 5001,
        "date_created": "2025-01-21T14:32:10.123Z"
      }
    ]
  }'
```

### Get Recommendations

```bash
curl http://localhost:8000/user/101/recommendations?limit=5
```

### Update Products

```bash
curl -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {
        "id": 5001,
        "name": "Gaming Laptop",
        "description": "High-performance gaming laptop with RTX graphics",
        "brand": "TechBrand",
        "category": "Electronics",
        "is_active": true,
        "rating": {"score": 4.5, "total": 120},
        "sold": 45,
        "skus": [{"id": "sku1", "price": 1200000}]
      }
    ],
    "metadata_only": false
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Event Types and Weights

- `view`: 0.1
- `add_to_cart`: 0.3
- `purchase`: 0.6
- `rating`: 0.2

## Configuration

Key parameters in `MilvusService`:

- `milvus_host` and `milvus_port`: Milvus connection (default: localhost:19530)
- `event_weights`: Event type weights (view: 0.1, add_to_cart: 0.3, purchase: 0.6, rating: 0.2)
- `update_weight`: Vector blending weight (0.5)
- `MAX_LENGTH_EMBED`: Max text length for embeddings (4048)
- `DESCRIPTION_LENGTH`: Max description length (10240)
