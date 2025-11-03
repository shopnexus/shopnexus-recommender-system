"""Service layer for recommendation system"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import event_weights
from utils import DataUtils
from milvus import MilvusClient
from embeddings import EmbeddingService
from cf_model import CFModel
from fusion import EmbeddingFusion

logger = logging.getLogger(__name__)

# CF embedding dimension configuration
CF_DIM = 64

class Service:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """Initialize service with Milvus connection and embedding services"""
        self.update_weight = 0.5

        # Initialize embedding service
        self.embedding_service = EmbeddingService()

        # Initialize Milvus client (dynamic fused dim = dense_dim + cf_dim)
        self.client = MilvusClient(milvus_host=milvus_host, milvus_port=milvus_port)
        self.dense_dim = self.client.dense_dim
        self.sparse_dim = self.client.sparse_dim
        self.fused_dim = self.client.fused_dim

        # Initialize fusion layer (no shrink; output dim = dense_dim + cf_dim)
        self.fusion = EmbeddingFusion(
            content_dim=self.dense_dim,
            cf_dim=CF_DIM
        )

        # Training data storage (in-memory for MVP)
        self.training_data: List[Dict] = []
        
        # CF model (will be initialized when training)
        self.cf_model: Optional[CFModel] = None
        self.user_cf_embeddings: Optional[np.ndarray] = None
        self.item_cf_embeddings: Optional[np.ndarray] = None

    def semantic_search(self, query: str, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10):
        """Semantic search using content_products collection"""
        # Encode query
        query_embeddings = self.embedding_service.encode_text(query)
        
        # Perform hybrid search in content_products
        results = self.client.semantic_search(
            query_embeddings["dense"],
            query_embeddings["sparse"],
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            offset=offset,
            limit=limit
        )
        
        return [{"id": hit["id"], "score": float(hit.score)} for hit in results[0]]

    def recommend(self, account_id: int, limit: int = 10):
        """Recommend products for a user using hybrid_products collection"""
        # Get user fused vector from hybrid_customers
        user_vectors = self.client.get_hybrid_user_vectors([account_id])
        user_vector = user_vectors.get(account_id)
        
        if user_vector is None or len(user_vector) == 0:
            logger.warning(f"No user vector found for account_id: {account_id}. Returning content-based fallback.")
            # Cold-start fallback: use content-based search if no user vector exists
            # This works even if CF model hasn't been trained yet
            return self._cold_start_recommendations(account_id, limit)
        
        # Search in hybrid_products collection
        results = self.client.dense_search_hybrid_products(
            user_vector,
            offset=0,
            limit=limit
        )
        
        return [{"id": hit["id"], "score": float(hit.score)} for hit in results[0]]
    
    def _cold_start_recommendations(self, account_id: int, limit: int):
        """Cold-start recommendations using content embeddings"""
        # For cold-start, return popular items from content_products
        # In a real system, you might want to use actual popular items from analytics
        # For now, we'll return empty list and log a message
        logger.info(f"Cold-start recommendation requested for account {account_id}. Train CF model for personalized recommendations.")
        return []

    def ingest_training_data(self, interactions: List[Dict]):
        """Ingest user-item interactions for CF training
        
        Args:
            interactions: List of interaction dictionaries
                Format: [{"user_id": int, "item_id": int, "rating": float, "timestamp": str}]
        """
        # Validate interactions
        valid_interactions = []
        for interaction in interactions:
            if not all(k in interaction for k in ["user_id", "item_id", "rating"]):
                logger.warning(f"Invalid interaction: {interaction}, skipping")
                continue
            valid_interactions.append(interaction)
        
        # Add timestamp if not present
        current_time = datetime.now().isoformat()
        for interaction in valid_interactions:
            if "timestamp" not in interaction:
                interaction["timestamp"] = current_time
        
        self.training_data.extend(valid_interactions)
        logger.info(f"Ingested {len(valid_interactions)} interactions. Total: {len(self.training_data)}")

    def train_cf_model(self, epochs: int = 50, batch_size: int = 256):
        """Train CF model from ingested training data"""
        if not self.training_data:
            logger.error("No training data available. Please ingest training data first.")
            return {"error": "No training data available"}
        
        logger.info(f"Starting CF model training on {len(self.training_data)} interactions")
        
        # Extract unique user and item IDs
        user_ids = set(interaction["user_id"] for interaction in self.training_data)
        item_ids = set(interaction["item_id"] for interaction in self.training_data)
        
        num_users = max(user_ids) + 1  # Assume IDs are 0-indexed or need to remap
        num_items = max(item_ids) + 1
        
        logger.info(f"Unique users: {len(user_ids)}, Unique items: {len(item_ids)}")
        logger.info(f"Model dimensions: users={num_users}, items={num_items}, embedding_dim={CF_DIM}")
        
        # Create user and item ID mappings (to ensure contiguous IDs starting from 0)
        user_id_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        item_id_map = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
        
        # Map interactions to model IDs
        mapped_interactions = [
            (
                user_id_map[interaction["user_id"]],
                item_id_map[interaction["item_id"]],
                float(interaction["rating"])
            )
            for interaction in self.training_data
        ]
        
        # Prepare warm-start matrices if available
        initial_user = None
        initial_item = None
        if getattr(self, 'user_cf_embeddings', None) is not None and getattr(self, 'user_id_map', None) is not None:
            initial_user = np.zeros((len(user_ids), CF_DIM), dtype=np.float32)
            for uid, new_idx in user_id_map.items():
                old_idx = self.user_id_map.get(uid)
                if old_idx is not None and old_idx < self.user_cf_embeddings.shape[0]:
                    initial_user[new_idx] = self.user_cf_embeddings[old_idx]
        if getattr(self, 'item_cf_embeddings', None) is not None and getattr(self, 'item_id_map', None) is not None:
            initial_item = np.zeros((len(item_ids), CF_DIM), dtype=np.float32)
            for iid, new_idx in item_id_map.items():
                old_idx = self.item_id_map.get(iid)
                if old_idx is not None and old_idx < self.item_cf_embeddings.shape[0]:
                    initial_item[new_idx] = self.item_cf_embeddings[old_idx]

        # Initialize and train CF model (with L2, seed, early stopping)
        self.cf_model = CFModel(
            num_users=len(user_ids),
            num_items=len(item_ids),
            embedding_dim=CF_DIM,
            l2_lambda=1e-5,
            seed=42,
        )
        
        history = self.cf_model.train(
            interactions=mapped_interactions,
            epochs=epochs,
            batch_size=batch_size,
            patience=3,
            initial_user_embeddings=initial_user,
            initial_item_embeddings=initial_item,
        )
        
        # Get trained embeddings
        self.user_cf_embeddings, self.item_cf_embeddings = self.cf_model.get_embeddings()
        
        # Store mappings for later use
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        
        logger.info("CF model training completed")
        
        # Update hybrid collections with fused embeddings
        self._update_hybrid_embeddings(user_id_map, item_id_map)
        
        return history

    def _update_hybrid_embeddings(self, user_id_map: Dict[int, int], item_id_map: Dict[int, int]):
        """Update hybrid_products and hybrid_customers with fused embeddings"""
        if self.user_cf_embeddings is None or self.item_cf_embeddings is None:
            logger.error("CF embeddings not available. Train model first.")
            return
        
        # Get all product IDs from content_products to update hybrid_products
        # For MVP, we'll need to query all products or process in batches
        # Here we'll update products that are in the training data
        logger.info("Updating hybrid_products with fused embeddings")
        
        # Get item IDs from training data
        training_item_ids = list(set(interaction["item_id"] for interaction in self.training_data))
        
        # Get content embeddings for these items
        content_vectors = self.client.get_content_product_vectors(training_item_ids)
        
        # Update hybrid_products
        hybrid_product_ids = []
        hybrid_product_vectors = []
        
        for item_id in training_item_ids:
            if item_id not in item_id_map:
                continue
            
            model_item_idx = item_id_map[item_id]
            item_cf_embedding = self.item_cf_embeddings[model_item_idx]
            
            if item_id not in content_vectors:
                logger.warning(f"No content vector for item {item_id}, skipping")
                continue
            
            content_vector = content_vectors[item_id]
            
            # Fuse embeddings
            fused_vector = self.fusion.fuse_embeddings(content_vector, item_cf_embedding)
            
            hybrid_product_ids.append(item_id)
            hybrid_product_vectors.append(fused_vector.tolist())
        
        if hybrid_product_ids:
            entities = [
                hybrid_product_ids,
                hybrid_product_vectors
            ]
            self.client.upsert_hybrid_products(entities)
            logger.info(f"Updated {len(hybrid_product_ids)} products in hybrid_products")
        
        # Update hybrid_customers
        logger.info("Updating hybrid_customers with fused embeddings")
        
        training_user_ids = list(set(interaction["user_id"] for interaction in self.training_data))
        
        hybrid_user_ids = []
        hybrid_user_vectors = []
        
        for user_id in training_user_ids:
            if user_id not in user_id_map:
                continue
            
            model_user_idx = user_id_map[user_id]
            user_cf_embedding = self.user_cf_embeddings[model_user_idx]
            
            # Get recent product embeddings for this user
            user_items = [
                interaction["item_id"]
                for interaction in self.training_data
                if interaction["user_id"] == user_id
            ]
            
            if not user_items:
                continue
            
            # Get content vectors for user's items
            user_content_vectors = self.client.get_content_product_vectors(user_items[:10])  # Last 10 items
            
            if not user_content_vectors:
                # If no content vectors, use zero vector
                recent_content_avg = np.zeros(self.dense_dim)
            else:
                # Average recent content embeddings
                recent_content_avg = np.mean(list(user_content_vectors.values()), axis=0)
            
            # Fuse embeddings
            fused_vector = self.fusion.fuse_embeddings(recent_content_avg, user_cf_embedding)
            
            hybrid_user_ids.append(user_id)
            hybrid_user_vectors.append(fused_vector.tolist())
        
        if hybrid_user_ids:
            entities = [
                hybrid_user_ids,
                hybrid_user_vectors
            ]
            self.client.upsert_hybrid_customers(entities)
            logger.info(f"Updated {len(hybrid_user_ids)} users in hybrid_customers")

    def update_user_embedding_from_events(self, account_id: int, events: List[Dict]):
        """Update user embedding from events (for real-time updates)
        
        Args:
            account_id: User account ID
            events: List of event dictionaries [{'ref_id': int, 'event_type': str, 'date_created': str}]
        """
        # Calculate recent content embedding average from events
        product_ids = [event.get("ref_id") for event in events if event.get("ref_id")]
        
        if not product_ids:
            logger.warning(f"No product IDs found in events for account {account_id}")
            return
        
        # Get content vectors
        content_vectors = self.client.get_content_product_vectors(product_ids)
        
        if not content_vectors:
            logger.warning(f"No content vectors found for products: {product_ids}")
            return
        
        # Average content embeddings
        recent_content_avg = np.mean(list(content_vectors.values()), axis=0)
        
        # Get user CF embedding if available
        if self.user_cf_embeddings is None:
            logger.warning("CF embeddings not available. User embedding update skipped.")
            return
        
        if account_id not in getattr(self, 'user_id_map', {}):
            logger.warning(f"User {account_id} not in CF model. Skipping.")
            return
        
        model_user_idx = self.user_id_map[account_id]
        user_cf_embedding = self.user_cf_embeddings[model_user_idx]
        
        # Fuse embeddings
        fused_vector = self.fusion.fuse_embeddings(recent_content_avg, user_cf_embedding)
        
        # Update in Milvus
        entities = [
            [account_id],
            [fused_vector.tolist()]
        ]
        self.client.upsert_hybrid_customers(entities)
        
        logger.info(f"Updated user embedding for account {account_id}")

    def process_events_batch(self, events: List[Dict]):
        """Process events and update user vectors (legacy method for backward compatibility)
        
        Args:
            events: List of event dictionaries
        """
        # Group events by account_id
        account_events = {}
        for event in events:
            account_id = event.get('account_id')
            if account_id:
                if account_id not in account_events:
                    account_events[account_id] = []
                account_events[account_id].append(event)
        
        # Update user embeddings from events
        for account_id, user_events in account_events.items():
            try:
                self.update_user_embedding_from_events(account_id, user_events)
            except Exception as e:
                logger.error(f"Error updating user embedding for account {account_id}: {e}")
                continue

        # ingest_training_data
        self.ingest_training_data([{
            "user_id": account_id,
            "item_id": event.get("ref_id"),
            "rating": event_weights.get(event.get("event_type"), 0),
            "timestamp": event.get("date_created")
        } for account_id, user_events in account_events.items() for event in user_events])
        
        logger.info(f"Batch processing completed: {len(account_events)} accounts, {len(events)} events")
        
        # Flush collections
        self.client.hybrid_customers_collection.flush()
        self.client.content_products_collection.flush()