import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.model.hybrid import MGTEEmbeddingFunction

logger = logging.getLogger(__name__)

content_products_schema = {
  "name": "content_products",
  "description": "Products with content embeddings for semantic search",
  "schema": [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="is_active", dtype=DataType.BOOL),
    FieldSchema(name="rating", dtype=DataType.FLOAT),
    FieldSchema(name="skus", dtype=DataType.JSON),
    FieldSchema(name="specifications", dtype=DataType.JSON),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
  ],
  "indexes": [
    {
      "field_name": "sparse_vector",
      "index_type": "SPARSE_INVERTED_INDEX",
      "metric_type": "IP"
    },
    {
      "field_name": "dense_vector",
      "index_type": "AUTOINDEX",
      "metric_type": "COSINE"
    }
  ]
}

hybrid_products_schema = {
  "name": "hybrid_products",
  "description": "Products with fused embeddings for recommendations",
  "schema": [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=832),
  ],
  "indexes": [
    {
      "field_name": "dense_vector",
      "index_type": "AUTOINDEX",
      "metric_type": "COSINE"
    }
  ]
}

hybrid_customers_schema = {
  "name": "hybrid_customers",
  "description": "Customers with fused embeddings for recommendations",
  "schema": [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=832),
  ],
  "indexes": [
    {
      "field_name": "dense_vector",
      "index_type": "AUTOINDEX",
      "metric_type": "COSINE"
    }
  ]
}


class MilvusClient:
    """Milvus client for managing connections, collections, and operations"""

    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """Initialize Milvus connection and setup collections"""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        # Initialize embedding function
        self.ef = MGTEEmbeddingFunction(use_fp16=False, device="cpu")
        self.dense_dim = self.ef.dim["dense"]
        self.sparse_dim = self.ef.dim["sparse"]

        # Setup connection and collections
        connections.connect(host=self.milvus_host, port=self.milvus_port)

        self.content_products_collection = self._setup_collection(content_products_schema)
        self.hybrid_products_collection = self._setup_collection(hybrid_products_schema)
        self.hybrid_customers_collection = self._setup_collection(hybrid_customers_schema)

    def _setup_collection(self, schema: dict):
        """Setup collection for user vectors (fused embeddings)"""
        
        collection = Collection(name=schema["name"], schema=CollectionSchema(schema["schema"], description=schema["description"]))
        for index in schema["indexes"]:
            collection.create_index(index["field_name"], {
                "index_type": index["index_type"],
                "metric_type": index["metric_type"]
            })
        collection.load()
        return collection

    def semantic_search(self, dense_vec, sparse_vec, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10):
        """Hybrid search in content_products collection"""
        dense_req = AnnSearchRequest([dense_vec], "dense_vector", {"metric_type": "COSINE"}, limit=limit)
        sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP"}, limit=limit)
        rerank = WeightedRanker(sparse_weight, dense_weight)

        return self.content_products_collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["id"],
            **{
                "offset": offset
              }
        )

    def dense_search_content_products(self, dense_vec, offset=0, limit=10):
        """Dense vector search in content_products collection"""

        return self.content_products_collection.search(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            output_fields=["id"],
            **{
                "offset": offset
            }
        )

    def dense_search_hybrid_products(self, dense_vec, offset=0, limit=10):
        """Dense vector search in hybrid_products collection"""

        return self.hybrid_products_collection.search(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            output_fields=["id"],
            **{
                "offset": offset
            }
        )

    def get_content_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fetch dense vectors for given product IDs from content_products collection"""
        return {result.get('id'): np.array(result.get('dense_vector')) for result in self.query_by_ids(
            self.content_products_collection, product_ids, ["id", "dense_vector"]
        )}

    def get_hybrid_user_vectors(self, account_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get existing user vectors from hybrid_customers collection"""
        return {result.get('id'): np.array(result.get('dense_vector')) for result in self.query_by_ids(
            self.hybrid_customers_collection, account_ids, ["id", "dense_vector"]
        )}

    def upsert_content_products(self, entities: List[List], partial_update: bool = True):
        """Upsert products to content_products collection"""
        self.content_products_collection.upsert(entities, None, None, **{"partial_update": partial_update})

    def upsert_hybrid_products(self, entities: List[List], partial_update: bool = True):
        """Upsert products to hybrid_products collection"""
        self.hybrid_products_collection.upsert(entities, None, None, **{"partial_update": partial_update})

    def upsert_hybrid_customers(self, entities: List[List], partial_update: bool = True):
        """Upsert users to hybrid_customers collection"""
        self.hybrid_customers_collection.upsert(entities, None, None, **{"partial_update": partial_update})

    @staticmethod
    def query_by_ids(collection: Collection, ids: List[int], output_fields: List[str]) -> List[Dict]:
        """Query collection by list of IDs"""
        if not ids:
            return []
        id_list = ','.join(map(str, ids))
        expr = f"id in [{id_list}]"
        return collection.query(expr=expr, output_fields=output_fields, limit=len(ids))