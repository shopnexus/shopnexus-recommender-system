"""Service layer for recommendation system (recommendation-only)"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from milvus import MilvusClient
from embeddings import EmbeddingService
from utils import avg_vec_by_event, avg_vec_by_weight, get_event_weight
from cf_model import CFModel
from fusion import EmbeddingFusion

logger = logging.getLogger(__name__)


class Service:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """Initialize service with Milvus connection and embedding services"""
        self.embedding_service = EmbeddingService()  # Embedding dim
        self.model = CFModel()  # CF dim
        # self.model.load_model()
        self.fusion = EmbeddingFusion(
            content_dim=self.embedding_service.dense_dim,
            cf_dim=self.model.embedding_dim,
        )  # Fused dim
        self.client = MilvusClient(
            cf_dim=self.fusion.cf_dim,
            fused_dim=self.fusion.fused_dim,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )

    def semantic_search(
        self, query: str, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10
    ):
        """Semantic search using content_products collection"""
        # Encode query
        query_embeddings = self.embedding_service.embed_text(query)

        # Perform sparse and content search
        results = self.client.semantic_search(
            content_vec=query_embeddings["dense"],
            sparse_vec=query_embeddings["sparse"],
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            offset=offset,
            limit=limit,
        )

        return [{"id": hit["id"], "score": float(hit.score)} for hit in results[0]]

    def recommend(self, account_id: int, limit: int = 10):
        """Recommend products for a user using accounts collection"""
        # Get account fused vector from accounts collection
        account_fused_vector = self.client.get_vector(
            self.client.accounts_collection, account_id, "fused_vector"
        )

        if account_fused_vector is None:
            logger.warning(
                f"No account vector found for account_id: {account_id}. Returning content-based fallback."
            )
            return self.semantic_search(query="", limit=limit)

        # Search similar products by fused vector
        results = self.client.search(
            collection=self.client.products_collection,
            anns_field="fused_vector",
            vector=account_fused_vector,
            limit=limit,
        )

        return [{"id": hit["id"], "score": float(hit.score)} for hit in results]

    def _cold_start_recommendations(self, account_id: int, limit: int):
        """Cold-start recommendations using content embeddings"""
        # For cold-start, you might return popular items; for now, return empty list
        logger.info(
            f"Cold-start recommendation requested for account {account_id}. No user vector found."
        )
        return []

    def update_products(self, products: List[Dict], metadata_only: bool):
        """Update products in Milvus collection"""

        """
        type ProductDetail struct {
          ID             int64                  `json:"id"`
          Code           string                 `json:"code"`
          VendorID       int64                  `json:"vendor_id"`
          Name           string                 `json:"name"`
          Description    string                 `json:"description"`
          Brand          db.CatalogBrand        `json:"brand"`
          IsActive       bool                   `json:"is_active"`
          Category       db.CatalogCategory     `json:"category"`
          Rating         ProductRating          `json:"rating"`
          Resources      []sharedmodel.Resource `json:"resources"`
          Promotions     []ProductCardPromo     `json:"promotions"`
          Skus           []ProductDetailSku     `json:"skus"`
          Specifications map[string]string      `json:"specifications"`
        }
        """

        upsert_products = []
        non_exist_ids = self.client.get_non_existing_ids(
            self.client.products_collection, [product.get("id") for product in products]
        )
        embeddings = {}  # MAP [product.id] -> embedding
        item_cf_vectors = {}  # MAP [product.id] -> cf vector

        if not metadata_only:
            embeddings = self.embedding_service.embed_texts(
                [
                    f"{product.get('name')} {product.get('description')}"
                    for product in products
                ]
            )
            embeddings = {
                products[i].get("id"): embedding
                for i, embedding in enumerate(embeddings)
            }
            item_cf_vectors = self.client.get_vectors(
                self.client.products_collection,
                "cf_vector",
                set([product.get("id") for product in products]),
                # Find similar content products and average the corresponding cf vectors
                not_found_callback=lambda id: avg_vec_by_weight(
                    [
                        {
                            "score": result.score,
                            "vector": np.asarray(result.entity.get("cf_vector")),
                        }
                        for result in self.client.search(
                            self.client.products_collection,
                            "content_vector",
                            embeddings[id]["dense"],
                            output_fields=["id", "cf_vector"],
                            limit=5,
                        )
                    ],
                    np.zeros(self.fusion.cf_dim, dtype=np.float32),
                ),
            )

        for i, product in enumerate(products):
            update = {
                "id": product.get("id"),
                "name": product.get("name"),
                "description": product.get("description"),
                "brand": product.get("brand").get("name"),
                "category": product.get("category").get("name"),
                "is_active": product.get("is_active"),
                "rating": product.get("rating").get("score"),
                "skus": product.get("skus") or [],
                "specifications": product.get("specifications") or {},
            }

            # If metadata only, skip if product is not exist
            if metadata_only and product.get("id") in non_exist_ids:
                continue

            # Update the embedding vectors
            if not metadata_only:
                embedding = embeddings[product.get("id")]
                update["sparse_vector"] = embedding.get("sparse")
                update["content_vector"] = embedding.get("dense")
                update["cf_vector"] = item_cf_vectors[product.get("id")]
                update["fused_vector"] = self.fusion.fuse_embeddings(
                    update["content_vector"], update["cf_vector"]
                )

            upsert_products.append(update)

        self.client.upsert(
            self.client.products_collection,
            upsert_products,
            partial_update=True,
        )

    def process_events(self, events: List[Dict]):
        """Process analytics events"""

        # Group account events by account_id
        account_events = defaultdict(list)
        for event in events:
            account_events[event.get("account_id")].append(event)

        # Fetch vectors
        user_ids = list(set([event.get("account_id") for event in events]))
        item_ids = set([event.get("ref_id") for event in events])

        item_content_vectors = self.client.get_vectors(
            self.client.products_collection,
            anns_field="content_vector",
            ids=item_ids,
            not_found_callback=lambda id: np.zeros(
                self.fusion.content_dim, dtype=np.float32
            ),
        )

        item_cf_vectors = self.client.get_vectors(
            self.client.products_collection,
            anns_field="cf_vector",
            ids=item_ids,
            not_found_callback=lambda id: np.zeros(
                self.fusion.cf_dim, dtype=np.float32
            ),
        )

        # Update rows
        update_rows = []

        for account_id, events in account_events.items():
            item_content_vec = avg_vec_by_event(
                item_content_vectors,
                events,
                np.zeros(self.fusion.content_dim, dtype=np.float32),
            )
            item_cf_vec = avg_vec_by_event(
                item_cf_vectors,
                events,
                np.zeros(self.fusion.cf_dim, dtype=np.float32),
            )

            fused_vec = self.fusion.fuse_embeddings(item_content_vec, item_cf_vec)
            update_rows.append(
                {
                    "id": int(account_id),
                    "cf_vector": item_cf_vec,
                    "fused_vector": fused_vec,
                }
            )

        self.client.upsert(self.client.accounts_collection, update_rows)

    def resume_training(
        self,
        learning_rate: Optional[float] = None,
        l2_reg: Optional[float] = None,
        dropout_rate: Optional[float] = None,
    ):
        """Resume training the model, NOTE: NOT READY YET"""

        # Fetch things
        events = self.client.get_rows(
            self.client.interactions_collection,
            output_fields=["account_id", "ref_id", "event_type", "date_created"],
        )
        user_ids = list(set([event.get("account_id") for event in events]))
        item_ids = set([event.get("ref_id") for event in events])
        scores = np.array(
            [
                get_event_weight(event.get("event_type"), event.get("date_created"))
                for event in events
            ]
        )

        # Rebuild model
        # TODO: read the note.md
        self.client.accounts_collection.flush()
        self.client.products_collection.flush()
        self.model.n_users = self.client.accounts_collection.num_entities
        self.model.n_products = self.client.products_collection.num_entities
        self.model.learning_rate = (
            learning_rate if learning_rate is not None else self.model.learning_rate
        )
        self.model.l2_reg = l2_reg if l2_reg is not None else self.model.l2_reg
        self.model.dropout_rate = (
            dropout_rate if dropout_rate is not None else self.model.dropout_rate
        )
        self.model.build_model()

        # Train model
        self.model.train(
            user_ids=user_ids,
            product_ids=item_ids,
            scores=scores,
            shuffle=False,
            validation_split=0.2,
            epochs=10,
        )
