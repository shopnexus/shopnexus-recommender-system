import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
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

# NEW: NCF imports
import torch
import pickle
from ncf_model import NCF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH_EMBED = 4048
DESCRIPTION_LENGTH = 10 * 1024

# Helper functions
class DataUtils:
    """Utility class for data processing"""
    
    @staticmethod
    def truncate_text(text: str, max_length: int = MAX_LENGTH_EMBED) -> str:
        """Truncate text to max length"""
        if not text:
            return ""
        b = text.encode("utf-8")[:max_length]
        return b.decode("utf-8", errors="ignore")
    
    @staticmethod
    def extract_product_fields(product: Dict) -> tuple[str, str, str, str, bool, float, int, int, list[Any]]:
        """Extract and normalize product fields"""
        name = product.get('name', '')
        description = product.get('description', '')
        brand = product.get('brand', '')
        category = product.get('category', '')
        is_active = bool(product.get('is_active', False))
        
        rating = product.get('rating', {})
        rating_score = float(rating.get('score', 0.0))
        rating_total = int(rating.get('total', 0))
        
        sold = int(product.get('sold', 0))
        skus = product.get('skus', [])
        
        return name, description, brand, category, is_active, rating_score, rating_total, sold, skus
    
    @staticmethod
    def create_product_text_content(name: str, description: str, brand: str, category: str) -> str:
        """Create combined text content for embedding"""
        text_content = f"{name}. {description}. Brand: {brand}. Category: {category}"
        return DataUtils.truncate_text(text_content, MAX_LENGTH_EMBED)

class ResponseUtils:
    """Utility class for API responses"""
    
    @staticmethod
    def create_performance_response(message: str, count: int, processing_time: float, 
                                  total_time: float, processed_at: str = None) -> Dict:
        """Create standardized performance response"""
        return {
            "message": message,
            "processed_at": processed_at or datetime.now().isoformat(),
            "performance": {
                "items_count": count,
                "processing_time_seconds": round(processing_time, 3),
                "total_time_seconds": round(total_time, 3),
                "items_per_second": round(count / processing_time, 2) if processing_time > 0 else 0
            }
        }

class MilvusOperations:
    """Helper class for Milvus operations"""
    
    @staticmethod
    def query_by_ids(collection: Collection, ids: List[int], output_fields: List[str]) -> List[Dict]:
        """Query collection by list of IDs"""
        if not ids:
            return []
        id_list = ','.join(map(str, ids))
        expr = f"id in [{id_list}]"
        return collection.query(expr=expr, output_fields=output_fields, limit=len(ids))
    
    @staticmethod
    def query_by_account_id(collection: Collection, account_id: int, output_fields: List[str]) -> List[Dict]:
        """Query collection by account ID"""
        return collection.query(expr=f"account_id == {account_id}", output_fields=output_fields, limit=1)
    
    @staticmethod
    def upsert_and_flush(collection: Collection, entities: List[List], **kwargs):
        """Upsert entities and flush collection"""
        collection.upsert(entities, None, None, **kwargs)
        collection.flush()

class Service:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """Initialize service with Milvus connection"""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        # Existing weights
        self.event_weights = {
            "view": 0.1, "add_to_cart": 0.3, "purchase": 0.6, "rating": 0.2
        }
        self.update_weight = 0.5

        # Setup Milvus and collections
        self._setup_milvus()
        
        # NEW: Setup NCF model
        self.ncf_model = None
        self.ncf_mappings = None
        self._setup_ncf()

    def _setup_milvus(self):
        """Setup Milvus connection and create collections"""
        try:
            connections.connect(host=self.milvus_host, port=self.milvus_port)

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
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=DESCRIPTION_LENGTH),
                FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="is_active", dtype=DataType.BOOL),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="rating_score", dtype=DataType.DOUBLE),
                FieldSchema(name="rating_total", dtype=DataType.INT64),
                FieldSchema(name="sold", dtype=DataType.INT64),
                FieldSchema(name="skus", dtype=DataType.JSON),
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
            # logger.info(f"Connected to existing collection: {collection_name}")

        self.products_collection.load()

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

            weight = self.event_weights.get(event_type, 0.1)
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

            name, description, brand, category, _, _, _, _, _ = DataUtils.extract_product_fields(product)
            text_content = DataUtils.create_product_text_content(name, description, brand, category)

            product_data.append(text_content)
            product_ids.append(product['id'])

        return product_data, product_ids

    def _build_product_entities(self, products: List[Dict], product_ids: List[int],
                               embeddings: Dict = None) -> List[List]:
        """Build entities array for Milvus upsert"""
        names, descriptions, brands, is_actives, categories = [], [], [], [], []
        rating_scores, rating_totals, solds, skus_list = [], [], [], []
        
        for product in products:
            if product.get('id') not in product_ids:
                continue
                
            name, description, brand, category, is_active, rating_score, rating_total, sold, skus = DataUtils.extract_product_fields(product)
            
            names.append(name)
            descriptions.append(description)
            brands.append(brand)
            is_actives.append(is_active)
            categories.append(category)
            rating_scores.append(rating_score)
            rating_totals.append(rating_total)
            solds.append(sold)
            skus_list.append(skus)
        
        if embeddings:
            sparse_vectors = embeddings["sparse"]
            dense_vectors = embeddings["dense"]
        else:
            # Create empty vectors for new products
            sparse_vectors = [{} for _ in product_ids]
            dense_vectors = [[0.0] * self.dense_dim for _ in product_ids]
        
        return [
            product_ids, names, descriptions, brands, is_actives, categories,
            rating_scores, rating_totals, solds, skus_list, sparse_vectors, dense_vectors
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

    def _setup_ncf(self):
        """
        Load trained NCF model
        """
        try:
            logger.info("Loading NCF model...")
            
            # Load model checkpoint
            checkpoint = torch.load(
                './models/ncf_model_final.pt',
                map_location='cpu'
            )
            
            # Initialize model
            self.ncf_model = NCF(
                num_users=checkpoint['num_users'],
                num_products=checkpoint['num_products'],
                embed_dim=checkpoint['embed_dim'],
                mlp_layers=checkpoint['mlp_layers']
            )
            
            # Load weights
            self.ncf_model.load_state_dict(checkpoint['model_state_dict'])
            self.ncf_model.eval()
            
            # Load mappings
            with open('./models/ncf_mappings.pkl', 'rb') as f:
                self.ncf_mappings = pickle.load(f)
            
            logger.info(f"✅ NCF model loaded: {checkpoint['num_users']} users, "
                       f"{checkpoint['num_products']} products")
            
        except FileNotFoundError:
            # logger.warning("NCF model not found. Train model first using train_ncf_model()")
            # logger.warning("NCF recommendations will not be available") 
            return
        except Exception as e:
            logger.error(f"Error loading NCF model: {e}")
    
    def get_ncf_recommendations(
        self,
        account_id: int,
        limit: int = 20,
        exclude_interacted: bool = True
    ) -> List[Dict]:
        """
        Get product recommendations using trained NCF model
        
        Args:
            account_id: User ID
            limit: Number of recommendations
            exclude_interacted: Exclude products user already interacted with
        
        Returns:
            List of recommended products with scores
        """
        if self.ncf_model is None:
            logger.warning("NCF model not loaded")
            return []
        
        try:
            start_time = time.time()
            
            # Check if user is in training data
            if account_id not in self.ncf_mappings['user_id_to_idx']:
                logger.warning(f"User {account_id} not in NCF training data (cold start)")
                return self._handle_ncf_cold_start(account_id, limit)
            
            user_idx = self.ncf_mappings['user_id_to_idx'][account_id]
            
            # Get all product indices
            product_indices = list(range(len(self.ncf_mappings['idx_to_product_id'])))
            
            # Predict scores for all products
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx] * len(product_indices))
                product_tensor = torch.LongTensor(product_indices)
                
                scores = self.ncf_model(user_tensor, product_tensor).numpy()
            
            # Get product IDs
            product_ids = [
                self.ncf_mappings['idx_to_product_id'][idx]
                for idx in product_indices
            ]
            
            # Exclude already interacted products
            if exclude_interacted:
                interacted_products = self._get_user_interacted_products(account_id)
                valid_indices = [
                    i for i, pid in enumerate(product_ids)
                    if pid not in interacted_products
                ]
                product_ids = [product_ids[i] for i in valid_indices]
                scores = scores[valid_indices]
              
            # Sort by score
            sorted_indices = np.argsort(scores)[::-1][:limit]
            
            # Fetch product details
            recommended_product_ids = [product_ids[i] for i in sorted_indices]
            recommended_scores = [float(scores[i]) for i in sorted_indices]
            
            products = MilvusOperations.query_by_ids(
                self.products_collection,
                recommended_product_ids,
                ["id", "name", "brand", "rating_score", "sold", "description"]
            )
            
            # Build response
            results = []
            product_map = {p['id']: p for p in products}
            
            for product_id, score in zip(recommended_product_ids, recommended_scores):
                if product_id in product_map:
                    product = product_map[product_id]
                    results.append({
                        'id': product_id,
                        'name': product['name'],
                        'brand': product['brand'],
                        'rating': product['rating_score'],
                        'sold': product['sold'],
                        'ncf_score': score,
                        'source': 'ncf_collaborative_filtering'
                    })
            
            logger.info(f"NCF recommendations for user {account_id}: "
                       f"{len(results)} products in {time.time() - start_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in NCF recommendations: {e}")
            return []
    
    # def _get_user_interacted_products(self, account_id: int) -> set:
    #     """Get set of products user has interacted with"""
    #     # Query from analytics_events or from cache
    #     # This is a simplified version - implement based on your DB schema
    #     try:
    #         # Example: query recent interactions
    #         # In production, this should be cached
    #         query = f"""
    #             SELECT DISTINCT ref_id 
    #             FROM analytics_events 
    #             WHERE account_id = {account_id} 
    #             AND timestamp >= NOW() - INTERVAL '90 days'
    #         """
    #         # Execute and return set of product IDs
    #         # This is placeholder - implement based on your DB
    #         return set()
    #     except:
    #         return set()
    def _get_user_interacted_products(self, account_id: int) -> set:
      """Get products user interacted with from NCF mappings"""
      try:
          # If NCF model exists, check training data
          if self.ncf_model is None or self.ncf_mappings is None:
              return set()
          
          # Simple: Return empty set → NCF will score all products
          # The training already knows which products user liked
          return set()
          
      except Exception as e:
          logger.error(f"Error: {e}")
          return set()

    
    def _handle_ncf_cold_start(self, account_id: int, limit: int) -> List[Dict]:
        """
        Handle cold start for users not in training data
        
        Strategy: Return popular products (high rating + high sales)
        """
        try:
            # Query popular products
            query_expr = "is_active == True"
            
            results = self.products_collection.query(
                expr=query_expr,
                output_fields=["id", "name", "brand", "rating_score", "rating_total", "sold"],
                limit=limit * 2
            )
            
            # Score products by popularity
            scored_products = []
            for product in results:
                # Popularity score = weighted sum of rating and sales
                rating_score = product['rating_score'] * min(product['rating_total'] / 100, 1.0)
                sales_score = min(product['sold'] / 1000, 1.0)
                popularity = 0.6 * rating_score + 0.4 * sales_score
                
                scored_products.append({
                    'id': product['id'],
                    'name': product['name'],
                    'brand': product['brand'],
                    'rating': product['rating_score'],
                    'sold': product['sold'],
                    'ncf_score': float(popularity),
                    'source': 'ncf_cold_start_popular'
                })
            
            # Sort and return top-K
            scored_products.sort(key=lambda x: x['ncf_score'], reverse=True)
            
            logger.info(f"Cold start recommendations for new user {account_id}: "
                       f"{len(scored_products[:limit])} popular products")
            
            return scored_products[:limit]
            
        except Exception as e:
            logger.error(f"Error in cold start handling: {e}")
            return []
    
    def get_hybrid_recommendations(
        self,
        account_id: int,
        limit: int = 20,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Hybrid recommendations combining multiple methods
        
        Methods:
        1. Content-based (existing user vector search)
        2. NCF collaborative filtering
        3. Popular items (fallback)
        
        Args:
            account_id: User ID
            limit: Number of recommendations
            weights: Dict with keys 'content', 'collaborative', 'popular'
                    Default: {'content': 0.5, 'collaborative': 0.4, 'popular': 0.1}
        """
        if weights is None:
            weights = {
                'content': 0.5,
                'collaborative': 0.4,
                'popular': 0.1
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Get recommendations from each source
        product_scores = {}
        
        # 1. Content-based (existing method)
        try:
            user_vector = self.get_user_vector(account_id)
            if user_vector is not None:
                content_results = self.dense_search(user_vector.tolist(), limit=limit * 2)
                
                for hit in content_results:
                    product_id = hit['id']
                    product_scores[product_id] = product_scores.get(product_id, 0) + \
                                                 weights['content'] * float(hit.score)
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
        
        # 2. NCF collaborative filtering
        try:
            ncf_results = self.get_ncf_recommendations(account_id, limit=limit * 2)
            
            for item in ncf_results:
                product_id = item['id']
                product_scores[product_id] = product_scores.get(product_id, 0) + \
                                            weights['collaborative'] * item['ncf_score']
        except Exception as e:
            logger.error(f"Error in NCF recommendations: {e}")
        
        # 3. Popular items (small contribution)
        try:
            popular_results = self._handle_ncf_cold_start(account_id, limit=limit)
            
            for item in popular_results:
                product_id = item['id']
                product_scores[product_id] = product_scores.get(product_id, 0) + \
                                            weights['popular'] * item['ncf_score']
        except Exception as e:
            logger.error(f"Error in popular recommendations: {e}")
        
        # Sort by combined score
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        # Fetch product details
        product_ids = [pid for pid, _ in sorted_products]
        products = MilvusOperations.query_by_ids(
            self.products_collection,
            product_ids,
            ["id", "name", "brand", "rating_score", "sold", "description"]
        )
        
        # Build response
        product_map = {p['id']: p for p in products}
        results = []
        
        for product_id, score in sorted_products:
            if product_id in product_map:
                product = product_map[product_id]
                results.append({
                    'id': product_id,
                    'name': product['name'],
                    'brand': product['brand'],
                    'rating': product['rating_score'],
                    'sold': product['sold'],
                    'hybrid_score': float(score),
                    'source': 'hybrid_recommendation'
                })
        
        logger.info(f"Hybrid recommendations for user {account_id}: {len(results)} products")
        
        return results

service = Service()