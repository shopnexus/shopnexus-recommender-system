# Unified Milvus Service

A clean, focused service that combines product search and customer analytics functionality using Milvus vector database.

## Features

- **Product Search**: Hybrid search with dense and sparse vectors
- **Customer Analytics**: Process events and create user preference vectors
- **User Similarity**: Find users with similar preferences
- **Product Recommendations**: Multiple recommendation algorithms
- **REST API**: Clean HTTP endpoints for all functionality

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure Milvus is running (localhost:19530)

## Usage

### Running the Service

To start the REST API server:

```bash
python unified_service.py
```

This will start a Flask server on `http://localhost:8000`

### Testing

Run the test script to verify functionality:

```bash
python test_unified_service.py
```

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
# Content-based recommendations
curl http://localhost:8000/user/101/recommendations?limit=5

# Collaborative filtering
curl http://localhost:8000/user/101/recommendations/collaborative?limit=5

# Hybrid recommendations
curl http://localhost:8000/user/101/recommendations/hybrid?limit=5&content_weight=0.8&collaborative_weight=0.2
```

### Find Similar Users

```bash
curl http://localhost:8000/user/101/similar?limit=5
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Event Types and Weights

The system processes the following event types with their respective weights:

- `view`: 0.2 (lowest weight)
- `add_to_cart`: 0.3
- `purchase`: 0.4 (highest weight)
- `rating`: 0.1

## Data Format

### Input Events

```json
{
  "id": 1,
  "account_id": 101,
  "event_type": "view",
  "ref_type": "Product",
  "ref_id": 5001,
  "date_created": "2025-01-21T14:32:10.123Z"
}
```

```json
{
  "id": 2,
  "account_id": 101,
  "event_type": "add_to_cart",
  "ref_type": "Product",
  "ref_id": 5001,
  "metadata": {
    "quantity": 2,
    "price": 349000
  },
  "date_created": "2025-01-21T14:33:02.321Z"
}
```

## Milvus Collections

**Products Collection:**

- `id` (INT64, Primary Key): Product identifier
- `text` (VARCHAR): Product description
- `sparse_vector` (SPARSE_FLOAT_VECTOR): Sparse embedding
- `dense_vector` (FLOAT_VECTOR): Dense embedding

**Customer Collection:**

- `account_id` (INT64, Primary Key): User account identifier
- `dense_vector` (FLOAT_VECTOR): User preference vector
- `last_updated` (VARCHAR): Timestamp of last update
- `event_count` (INT64): Number of events processed

## Recommendation Algorithms

1. **Content-Based**: Uses user preference vector to find semantically similar products
2. **Collaborative Filtering**: Finds users with similar preferences and recommends products they liked
3. **Hybrid**: Combines both approaches with configurable weights

## Configuration

You can modify the following parameters in `UnifiedMilvusService`:

- `milvus_host` and `milvus_port`: Milvus connection details
- `event_weights`: Weights for different event types
- `update_weight`: Weight for blending new vectors with existing ones (0.3)

## Vector Calculation Logic

1. **Event Processing**: Groups events by `account_id`
2. **Weight Application**: Applies event type weights and metadata multipliers
3. **Product Vector Fetching**: Retrieves actual product embeddings from products collection
4. **Vector Creation**: Creates weighted combination of product vectors
5. **Vector Blending**: Updates existing vectors using: `(1 - update_weight) * existing + update_weight * new`

