import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.model.hybrid import MGTEEmbeddingFunction

from config import DESCRIPTION_LENGTH

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus client for managing connections, collections, and operations"""

    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """Initialize Milvus connection and setup collections"""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        # Initialize embedding function
        self.ef = MGTEEmbeddingFunction(use_fp16=False, device="cpu")
        self.dense_dim = self.ef.dim["dense"]
        # Fused embedding dimension (from fusion layer)
        self.fused_dim = 768

        # Setup connection and collections
        self._setup_connection()
        self._setup_content_products_collection()
        self._setup_hybrid_products_collection()
        self._setup_hybrid_customers_collection()

    def _setup_connection(self):
        """Setup Milvus connection"""
        try:
            connections.connect(host=self.milvus_host, port=self.milvus_port)
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _setup_content_products_collection(self):
        """Setup content_products collection for semantic search (hybrid search)"""
        collection_name = "content_products"

        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
            ]
            schema = CollectionSchema(fields, description="Products with content embeddings for semantic search")
            self.content_products_collection = Collection(name=collection_name, schema=schema)

            # Create indexes
            self.content_products_collection.create_index("sparse_vector", {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP"
            })
            self.content_products_collection.create_index("dense_vector", {
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE"
            })
            logger.info(f"Created collection: {collection_name}")
        else:
            self.content_products_collection = Collection(collection_name)
            logger.info(f"Connected to existing collection: {collection_name}")

        self.content_products_collection.load()

    def _setup_hybrid_products_collection(self):
        """Setup hybrid_products collection for recommendations (fused embeddings)"""
        collection_name = "hybrid_products"

        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.fused_dim),
            ]
            schema = CollectionSchema(fields, description="Products with fused embeddings for recommendations")
            self.hybrid_products_collection = Collection(name=collection_name, schema=schema)

            # Create index
            self.hybrid_products_collection.create_index("dense_vector", {
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE"
            })
            logger.info(f"Created collection: {collection_name}")
        else:
            self.hybrid_products_collection = Collection(collection_name)
            logger.info(f"Connected to existing collection: {collection_name}")

        self.hybrid_products_collection.load()

    def _setup_hybrid_customers_collection(self):
        """Setup hybrid_customers collection for user vectors (fused embeddings)"""
        collection_name = "hybrid_customers"

        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.fused_dim),
            ]
            schema = CollectionSchema(fields, description="Customers with fused embeddings for recommendations")
            self.hybrid_customers_collection = Collection(name=collection_name, schema=schema)

            # Create index
            self.hybrid_customers_collection.create_index("dense_vector", {
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE"
            })
            logger.info(f"Created collection: {collection_name}")
        else:
            self.hybrid_customers_collection = Collection(collection_name)
            logger.info(f"Connected to existing collection: {collection_name}")

        self.hybrid_customers_collection.load()


    def hybrid_search(self, dense_vec, sparse_vec, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10):
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
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        return self.content_products_collection.search(
            data=[dense_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=limit,
            output_fields=["id"],
            **{
                "offset": offset
            }
        )

    def dense_search_hybrid_products(self, dense_vec, offset=0, limit=10):
        """Dense vector search in hybrid_products collection"""
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        return self.hybrid_products_collection.search(
            data=[dense_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=limit,
            output_fields=["id", "metadata"],
            **{
                "offset": offset
            }
        )

    def get_content_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fetch dense vectors for given product IDs from content_products collection"""
        return {result.get('id'): np.array(result.get('dense_vector')) for result in self.query_by_ids(
            self.content_products_collection, product_ids, ["id", "dense_vector"]
        )}

    def get_content_product_sparse_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fetch sparse vectors for given product IDs from content_products collection"""
        return {result.get('id'): np.array(result.get('sparse_vector')) for result in self.query_by_ids(
            self.content_products_collection, product_ids, ["id", "sparse_vector"]
        )}

    def get_hybrid_user_vectors(self, account_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get existing user vectors from hybrid_customers collection"""
        return {result.get('id'): np.array(result.get('dense_vector')) for result in self.query_by_ids(
            self.hybrid_customers_collection, account_ids, ["id", "dense_vector"]
        )}

    def get_hybrid_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get fused product vectors from hybrid_products collection"""
        return {result.get('id'): np.array(result.get('dense_vector')) for result in self.query_by_ids(
            self.hybrid_products_collection, product_ids, ["id", "dense_vector"]
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