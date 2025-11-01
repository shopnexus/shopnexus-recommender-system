import time
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

from config import MAX_LENGTH_EMBED, DESCRIPTION_LENGTH, event_weights
from utils import DataUtils, MilvusOperations

logger = logging.getLogger(__name__)


class Service:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """Initialize service with Milvus connection"""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        self.update_weight = 0.5

        # Setup Milvus and collections
        self._setup_milvus()

    def _setup_milvus(self):
        """Setup Milvus connection and create collections"""
        try:
            connections.connect(host=self.milvus_host, port=self.milvus_port)
            logger.info("Connected to Milvus")

            # Initialize embedding function
            self.ef = MGTEEmbeddingFunction(use_fp16=False, device="cpu")
            self.dense_dim = self.ef.dim["dense"]

            # Setup collections
            self._setup_products_collection()
            self._setup_customer_collection()

        except Exception as e:
            logger.error(f"Failed to setup Milvus: {e}")
            raise

    def _setup_products_collection(self):
        """Setup products collection for search"""
        collection_name = "products"

        if not utility.has_collection(collection_name):
            fields = [
                # Product ID - primary key matching TProductDetail.id
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                
                # Product code matching TProductDetail.code
                FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=512),
                
                # Product name matching TProductDetail.name
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
                
                # Product description matching TProductDetail.description
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=DESCRIPTION_LENGTH),
                
                # Brand object matching TProductDetail.brand (Brand: {id, code, name, description})
                FieldSchema(name="brand", dtype=DataType.JSON),
                
                # Active status matching TProductDetail.is_active
                FieldSchema(name="is_active", dtype=DataType.BOOL),
                
                # Category object matching TProductDetail.category (Category: {id, name, description, parent_id})
                FieldSchema(name="category", dtype=DataType.JSON),
                
                # Rating object matching TProductDetail.rating ({score, total, breakdown})
                FieldSchema(name="rating", dtype=DataType.JSON),
                
                # SKUs array matching TProductDetail.skus
                FieldSchema(name="skus", dtype=DataType.JSON),
                
                # Specifications object matching TProductDetail.specifications (Record<string, string>)
                FieldSchema(name="specifications", dtype=DataType.JSON),
                
                # Vector fields for hybrid search
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
            ]
            schema = CollectionSchema(fields)
            self.products_collection = Collection(name=collection_name, schema=schema)

            # Create indexes
            self.products_collection.create_index("sparse_vector", {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP"
            })
            self.products_collection.create_index("dense_vector", {
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE"
            })
            logger.info(f"Created collection: {collection_name}")
        else:
            self.products_collection = Collection(collection_name)
            logger.info(f"Connected to existing collection: {collection_name}")

        self.products_collection.load()
        # self.products_collection.drop()

    def _setup_customer_collection(self):
        """Setup customer collection for user vectors"""
        collection_name = "customer"

        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="account_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
                FieldSchema(name="last_updated", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="event_count", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="Customer preference vectors")
            self.customer_collection = Collection(name=collection_name, schema=schema)

            self.customer_collection.create_index("dense_vector", {
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE"
            })
            logger.info(f"Created collection: {collection_name}")
        else:
            self.customer_collection = Collection(collection_name)

            logger.info(f"Connected to existing collection: {collection_name}")

        self.customer_collection.load()
        # self.customer_collection.drop()
        # exit(1)

    def hybrid_search(self, dense_vec, sparse_vec, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10):
        """Hybrid search in products collection"""
        dense_req = AnnSearchRequest([dense_vec], "dense_vector", {"metric_type": "COSINE"}, limit=limit)
        sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP"}, limit=limit)
        rerank = WeightedRanker(sparse_weight, dense_weight)

        return self.products_collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["id"],
            **{
                "offset": offset
            })[0]

    def dense_search(self, dense_vec, offset=0, limit=10):
        """Dense vector search in products collection"""
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        return self.products_collection.search(
            data=[dense_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=limit,
            output_fields=["id"],
            **{
                "offset": offset
            }
        )[0]

    def calculate_user_vector(self, events: List[Dict]) -> Optional[np.ndarray]:
        """Calculate user preference vector from events by fetching product embeddings"""
        if not events:
            return None

        start_time = time.time()

        # Collect product IDs and their preference scores
        preference_start = time.time()
        product_preferences = {}

        for event in events:
            ref_id = event.get('ref_id')
            event_type = event.get('event_type')

            if not ref_id or not event_type:
                continue

            weight = event_weights.get(event_type, 0.1)
            metadata_weight = 1.0

            if 'metadata' in event:
                metadata = event['metadata']
                if 'quantity' in metadata:
                    metadata_weight *= min(metadata['quantity'] / 5.0, 2.0)
                if 'price' in metadata:
                    metadata_weight *= min(metadata['price'] / 1000000, 1.5)

            if ref_id not in product_preferences:
                product_preferences[ref_id] = 0.0
            product_preferences[ref_id] += weight * metadata_weight

        preference_time = time.time() - preference_start

        if not product_preferences:
            return None

        # Fetch dense vectors for all products from products collection
        fetch_start = time.time()
        product_ids = list(product_preferences.keys())
        product_vectors = self._get_product_vectors(product_ids)
        fetch_time = time.time() - fetch_start

        if not product_vectors:
            logger.warning(f"No product vectors found for products: {product_ids}")
            return None

        # Create weighted user preference vector
        vector_start = time.time()
        preference_vector = np.zeros(self.dense_dim)
        total_weight = 0.0

        for product_id, preference_score in product_preferences.items():
            if product_id in product_vectors:
                product_vector = product_vectors[product_id]
                preference_vector += preference_score * product_vector
                total_weight += preference_score

        # Normalize the vector
        if total_weight > 0:
            preference_vector = preference_vector / total_weight

        vector_time = time.time() - vector_start
        total_time = time.time() - start_time

        logger.debug(f"Preference vector calculation: {len(product_preferences)} products, "
                     f"{len(product_vectors)} vectors fetched in {total_time:.3f}s "
                     f"(preference: {preference_time:.3f}s, fetch: {fetch_time:.3f}s, vector: {vector_time:.3f}s)")

        return preference_vector

    def _get_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fetch dense vectors for given product IDs from products collection"""
        try:
            results = MilvusOperations.query_by_ids(
                self.products_collection, product_ids, ["id", "dense_vector"]
            )
            
            product_vectors = {}
            for entity in results:
                product_id = entity.get('id')
                dense_vector = np.array(entity.get('dense_vector'))
                product_vectors[product_id] = dense_vector

            logger.info(f"Fetched {len(product_vectors)} product vectors out of {len(product_ids)} requested")
            return product_vectors

        except Exception as e:
            logger.error(f"Error fetching product vectors: {e}")
            return {}

    def get_user_vector(self, account_id: int) -> Optional[np.ndarray]:
        """Get existing user vector from Milvus"""
        try:
            results = MilvusOperations.query_by_account_id(
                self.customer_collection, account_id, ["dense_vector"]
            )
            return np.array(results[0]['dense_vector']) if results else None

        except Exception as e:
            logger.error(f"Error retrieving user vector for account {account_id}: {e}")
            return None

    def update_user_vector(self, account_id: int, new_vector: np.ndarray, event_count: int):
        """Update user vector with blending"""
        try:
            existing_vector = self.get_user_vector(account_id)
            blended_vector = (
                (1 - self.update_weight) * existing_vector + self.update_weight * new_vector
                if existing_vector is not None else new_vector
            )

            entities = [
                [account_id], [blended_vector.tolist()], 
                [datetime.now().isoformat()], [event_count]
            ]

            MilvusOperations.upsert_and_flush(
                self.customer_collection, entities, partial_update=True
            )
            logger.info(f"Updated user vector for account {account_id}")

        except Exception as e:
            logger.error(f"Error updating user vector for account {account_id}: {e}")

    def process_events_batch(self, events: List[Dict]):
        """Process events and update user vectors"""
        start_time = time.time()

        # Group events by account_id
        group_start = time.time()
        account_events = {}
        for event in events:
            account_id = event.get('account_id')
            if account_id:
                if account_id not in account_events:
                    account_events[account_id] = []
                account_events[account_id].append(event)
        group_time = time.time() - group_start

        # Process each account's events
        process_start = time.time()
        processed_accounts = 0
        total_events_processed = 0

        for account_id, user_events in account_events.items():
            try:
                # Calculate new user vector
                vector_start = time.time()
                new_vector = self.calculate_user_vector(user_events)
                vector_time = time.time() - vector_start

                if new_vector is not None:
                    # Update user vector
                    update_start = time.time()
                    self.update_user_vector(account_id, new_vector, len(user_events))
                    update_time = time.time() - update_start

                    processed_accounts += 1
                    total_events_processed += len(user_events)

                    logger.info(f"Account {account_id}: {len(user_events)} events processed "
                                f"(vector: {vector_time:.3f}s, update: {update_time:.3f}s)")
                else:
                    logger.warning(f"Account {account_id}: Failed to calculate user vector")

            except Exception as e:
                logger.error(f"Error processing events for account {account_id}: {e}")

        process_time = time.time() - process_start

        # Flush all updates at once for better performance
        flush_start = time.time()
        self.customer_collection.flush()
        flush_time = time.time() - flush_start

        total_time = time.time() - start_time

        logger.info(f"Batch processing completed: {processed_accounts} accounts, "
                    f"{total_events_processed} events in {process_time:.3f}s "
                    f"(grouping: {group_time:.3f}s, processing: {process_time:.3f}s, flush: {flush_time:.3f}s, total: {total_time:.3f}s)")

    @staticmethod
    def _prepare_product_data(products: List[Dict]) -> Tuple[List[str], List[int]]:
        """Prepare product data for processing"""
        product_data = []
        product_ids = []

        for product in products:
            if not product.get('id') or not product.get('name'):
                logger.warning(f"Skipping product with missing required fields: {product}")
                continue

            product_id, code, name, description, brand, is_active, category, rating, skus, specifications = DataUtils.extract_product_fields(product)
            text_content = DataUtils.create_product_text_content(name, description, brand, category)

            product_data.append(text_content)
            product_ids.append(product_id)

        return product_data, product_ids

    def _build_product_entities(self, products: List[Dict], product_ids: List[int],
                               embeddings: Dict = None) -> List[List]:
        """Build entities array for Milvus upsert matching TProductDetail schema"""
        # Initialize lists for all fields matching the schema order:
        # id, code, name, description, brand, is_active, category, rating, skus, specifications, sparse_vector, dense_vector
        ids_list, codes, names, descriptions = [], [], [], []
        brands, is_actives, categories = [], [], []
        ratings, skus_list, specifications_list = [], [], []
        
        for product in products:
            product_id = product.get('id')
            if product_id not in product_ids:
                continue
                
            extracted_id, code, name, description, brand, is_active, category, rating, skus, specifications = DataUtils.extract_product_fields(product)
            
            # Use the extracted_id to ensure consistency
            product_id = extracted_id
            
            ids_list.append(product_id)
            codes.append(code)
            names.append(name)
            descriptions.append(description)
            brands.append(brand)
            is_actives.append(is_active)
            categories.append(category)
            ratings.append(rating)
            skus_list.append(skus)
            specifications_list.append(specifications)
        
        if embeddings:
            sparse_vectors = embeddings["sparse"]
            dense_vectors = embeddings["dense"]
        else:
            # Create empty vectors for new products
            sparse_vectors = [{} for _ in product_ids]
            dense_vectors = [[0.0] * self.dense_dim for _ in product_ids]
        
        return [
            ids_list, codes, names, descriptions, brands, is_actives, categories,
            ratings, skus_list, specifications_list, sparse_vectors, dense_vectors
        ]
    
    def update_products_batch(self, products: List[Dict], metadata_only: bool = False) -> int:
        """Process products and update Milvus collection"""
        start_time = time.time()
        
        if not products:
            return 0

        prepare_start = time.time()
        product_data, product_ids = self._prepare_product_data(products)
        prepare_time = time.time() - prepare_start
        
        if not product_data:
            logger.warning("No valid products to process")
            return 0

        if metadata_only:
            return self._update_products_metadata_only(products, product_ids, start_time, prepare_time)
        else:
            return self._update_products_with_embeddings(product_data, product_ids, products, start_time, prepare_time)

    def _update_products_metadata_only(self, products: List[Dict], product_ids: List[int], 
                                      start_time: float, prepare_time: float) -> int:
        """Handle metadata-only product updates"""
        logger.info(f"Metadata-only update mode: {len(product_ids)} products")
        
        check_start = time.time()
        existing_products, existing_vectors = self._get_existing_product_vectors(product_ids)
        check_time = time.time() - check_start
        
        new_product_ids = [pid for pid in product_ids if pid not in existing_products]
        
        upsert_start = time.time()
        entities = self._build_metadata_only_entities(products, product_ids, existing_vectors, new_product_ids)
        
        try:
            MilvusOperations.upsert_and_flush(self.products_collection, entities, partial_update=True)
            logger.info(f"Updated {len(existing_products)} existing products and created {len(new_product_ids)} new products")
        except Exception as e:
            logger.error(f"Error in metadata-only upsert: {e}")
        
        upsert_time = time.time() - upsert_start
        total_time = time.time() - start_time
        
        logger.info(f"Metadata-only update completed: {len(product_ids)} products "
                   f"(prepare: {prepare_time:.3f}s, check: {check_time:.3f}s, upsert: {upsert_time:.3f}s, total: {total_time:.3f}s)")
        
        return len(product_ids)
    
    def _update_products_with_embeddings(self, product_data: List[str], product_ids: List[int], 
                                        products: List[Dict], start_time: float, prepare_time: float) -> int:
        """Handle full product updates with embeddings"""
        embedding_start = time.time()
        try:
            embeddings = self.ef(product_data)
            embedding_time = time.time() - embedding_start
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return 0

        upsert_start = time.time()
        entities = self._build_product_entities(products, product_ids, embeddings)
        
        try:
            MilvusOperations.upsert_and_flush(self.products_collection, entities, partial_update=True)
            upsert_time = time.time() - upsert_start
            total_time = time.time() - start_time
            
            logger.info(f"Products batch update completed: {len(product_ids)} products "
                       f"(prepare: {prepare_time:.3f}s, embedding: {embedding_time:.3f}s, "
                       f"upsert: {upsert_time:.3f}s, total: {total_time:.3f}s)")
            
            return len(product_ids)
            
        except Exception as e:
            logger.error(f"Error upserting products to Milvus: {e}")
            return 0
    
    def _get_existing_product_vectors(self, product_ids: List[int]) -> Tuple[set, Dict]:
        """Get existing product vectors from Milvus"""
        existing_products = set()
        existing_vectors = {}
        
        try:
            results = MilvusOperations.query_by_ids(
                self.products_collection, product_ids, ["id", "sparse_vector", "dense_vector"]
            )
            for result in results:
                product_id = result['id']
                existing_products.add(product_id)
                existing_vectors[product_id] = {
                    'sparse': result['sparse_vector'],
                    'dense': result['dense_vector']
                }
            logger.info(f"Found {len(existing_products)} existing products out of {len(product_ids)}")
        except Exception as e:
            logger.warning(f"Error checking existing products: {e}")
        
        return existing_products, existing_vectors
    
    def _build_metadata_only_entities(self, products: List[Dict], product_ids: List[int], 
                                     existing_vectors: Dict, new_product_ids: List[int]) -> List[List]:
        """Build entities for metadata-only updates"""
        sparse_vectors, dense_vectors = [], []
        
        # Build metadata entities first
        entities = self._build_product_entities(products, product_ids)
        
        # Override vectors based on existing/new status
        for i, product_id in enumerate(product_ids):
            if product_id in new_product_ids:
                sparse_vectors.append({})
                dense_vectors.append([0.0] * self.dense_dim)
            else:
                if product_id in existing_vectors:
                    sparse_vectors.append(existing_vectors[product_id]['sparse'])
                    dense_vectors.append(existing_vectors[product_id]['dense'])
                else:
                    sparse_vectors.append({})
                    dense_vectors.append([0.0] * self.dense_dim)
        
        # Replace the vector fields in entities
        entities[-2] = sparse_vectors  # sparse_vector
        entities[-1] = dense_vectors   # dense_vector
        
        return entities
    
    def generate_embeddings(self, texts: List[str]) -> Dict[str, List]:
        """Generate embeddings for given texts"""
        try:
            return self.ef(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {"sparse": [], "dense": []}
