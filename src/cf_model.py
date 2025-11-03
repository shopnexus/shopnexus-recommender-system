"""Collaborative Filtering model using Matrix Factorization"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

logger = logging.getLogger(__name__)


class CFModel:
    """Collaborative Filtering model using Matrix Factorization (warm start, L2, early stopping)"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, l2_lambda: float = 1e-5, seed: int = 42):
        """Initialize CF model"""
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.l2_lambda = l2_lambda
        self.seed = seed
        self.model = None
        self.user_embeddings = None
        self.item_embeddings = None
    
    def build_model(self):
        """Build TensorFlow model with L2 regularization and fixed seeds"""
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # Input layers
        user_input = layers.Input(shape=(), name='user_id', dtype='int32')
        item_input = layers.Input(shape=(), name='item_id', dtype='int32')
        
        # Embedding layers
        user_embedding = layers.Embedding(
            self.num_users,
            self.embedding_dim,
            name='user_embedding',
            embeddings_regularizer=regularizers.l2(self.l2_lambda)
        )(user_input)
        
        item_embedding = layers.Embedding(
            self.num_items,
            self.embedding_dim,
            name='item_embedding',
            embeddings_regularizer=regularizers.l2(self.l2_lambda)
        )(item_input)
        
        # Flatten embeddings
        user_vec = layers.Flatten()(user_embedding)
        item_vec = layers.Flatten()(item_embedding)
        
        # Dot product + sigmoid for rating prediction
        dot_product = layers.Dot(axes=1)([user_vec, item_vec])
        output = layers.Activation('sigmoid')(dot_product)
        
        self.model = keras.Model([user_input, item_input], output)
        self.model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        return self.model
    
    def train(
        self,
        interactions: List[Tuple[int, int, float]],
        epochs: int = 50,
        batch_size: int = 256,
        validation_split: float = 0.1,
        verbose: int = 1,
        patience: int = 3,
        initial_user_embeddings: Optional[np.ndarray] = None,
        initial_item_embeddings: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train the CF model (supports warm start and early stopping)"""
        if not self.model:
            self.build_model()
        
        if not interactions:
            logger.warning("No interactions provided for training")
            return {}
        
        # Convert to numpy arrays
        user_ids = np.array([u for u, _, _ in interactions], dtype=np.int32)
        item_ids = np.array([i for _, i, _ in interactions], dtype=np.int32)
        ratings = np.array([r for _, _, r in interactions], dtype=np.float32)
        
        # Normalize ratings to [0, 1] range if needed
        if ratings.max() > 1.0 or ratings.min() < 0.0:
            logger.info(f"Normalizing ratings from [{ratings.min()}, {ratings.max()}] to [0, 1]")
            ratings_min = ratings.min()
            ratings_max = ratings.max()
            if ratings_max > ratings_min:
                ratings = (ratings - ratings_min) / (ratings_max - ratings_min)
            else:
                ratings = np.ones_like(ratings) * 0.5
        
        logger.info(f"Training CF model on {len(interactions)} interactions")
        logger.info(f"Users: {self.num_users}, Items: {self.num_items}, Embedding dim: {self.embedding_dim}")

        # Warm start
        if initial_user_embeddings is not None and initial_user_embeddings.shape == (self.num_users, self.embedding_dim):
            self.model.get_layer('user_embedding').set_weights([initial_user_embeddings])
            logger.info("Warm-started user embeddings")
        if initial_item_embeddings is not None and initial_item_embeddings.shape == (self.num_items, self.embedding_dim):
            self.model.get_layer('item_embedding').set_weights([initial_item_embeddings])
            logger.info("Warm-started item embeddings")
        
        # Train model
        early = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(
            [user_ids, item_ids],
            ratings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early],
            verbose=verbose,
            shuffle=True,
        )
        
        # Extract embeddings from trained model
        user_embedding_layer = self.model.get_layer('user_embedding')
        item_embedding_layer = self.model.get_layer('item_embedding')
        
        self.user_embeddings = user_embedding_layer.get_weights()[0]
        self.item_embeddings = item_embedding_layer.get_weights()[0]
        
        logger.info(f"Model trained. User embeddings shape: {self.user_embeddings.shape}")
        logger.info(f"Item embeddings shape: {self.item_embeddings.shape}")
        
        return history.history
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get trained embeddings"""
        if self.user_embeddings is None or self.item_embeddings is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.user_embeddings, self.item_embeddings
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """Predict ratings for user-item pairs"""
        if not self.model:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        user_ids_arr = np.array(user_ids, dtype=np.int32)
        item_ids_arr = np.array(item_ids, dtype=np.int32)
        
        predictions = self.model.predict([user_ids_arr, item_ids_arr], verbose=0)
        return predictions.flatten()

