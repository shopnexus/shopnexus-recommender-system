import tensorflow as tf
import numpy as np
import os
import json
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CFModel:
    """
    Collaborative Filtering model using Matrix Factorization for generating embeddings.
    Suitable for integration with Milvus vector database.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-6,
        dropout_rate: float = 0.2,
        n_users: int = 0,
        n_products: int = 0,
    ):
        """
        Initialize the Matrix Factorization model.

        Args:
            embedding_dim: Dimension of embeddings (should match your mgte embeddings)
            learning_rate: Learning rate for optimizer
            l2_reg: L2 regularization factor
            dropout_rate: Dropout rate for regularization
            n_users: Total number of unique users
            n_products: Total number of unique products
        """
        self.embedding_dim = embedding_dim
        self.n_users = n_users
        self.n_products = n_products
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = None
        self.user_embedding_model = None
        self.product_embedding_model = None

    def build_model(self) -> tf.keras.Model:
        """
        Build the matrix factorization model with embedding layers.

        Returns:
            Compiled Keras model
        """
        if self.n_users == 0 or self.n_products == 0:
            raise ValueError("n_users and n_products must be set")

        # Input layers
        user_input = tf.keras.layers.Input(shape=(1,), name="user_input")
        product_input = tf.keras.layers.Input(shape=(1,), name="product_input")

        # Embedding layers with L2 regularization
        user_embedding = tf.keras.layers.Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="user_embedding",
        )(user_input)

        product_embedding = tf.keras.layers.Embedding(
            input_dim=self.n_products,
            output_dim=self.embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="product_embedding",
        )(product_input)

        # Flatten embeddings
        user_vec = tf.keras.layers.Flatten(name="user_vec")(user_embedding)
        product_vec = tf.keras.layers.Flatten(name="product_vec")(product_embedding)

        # Apply dropout for regularization
        user_vec = tf.keras.layers.Dropout(self.dropout_rate)(user_vec)
        product_vec = tf.keras.layers.Dropout(self.dropout_rate)(product_vec)

        # Cosine similarity output directly in [-1, 1] (no biases, no extra activation)
        output = tf.keras.layers.Dot(axes=1, normalize=True, name="prediction")(
            [user_vec, product_vec]
        )

        # Create model
        self.model = tf.keras.Model(
            inputs=[user_input, product_input],
            outputs=output,
            name="matrix_factorization",
        )

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",  # Mean Squared Error for regression
            metrics=["mae", "mse"],
        )

        # Create separate models for extracting embeddings
        self._build_embedding_models()

        return self.model

    def _build_embedding_models(self):
        """Build separate models for extracting user and product embeddings."""
        # User embedding extractor
        user_input = tf.keras.layers.Input(shape=(1,))
        user_emb_layer = self.model.get_layer("user_embedding")
        user_embedding = user_emb_layer(user_input)
        user_embedding = tf.keras.layers.Flatten()(user_embedding)
        # L2 normalize for cosine similarity
        user_embedding = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1)
        )(user_embedding)

        self.user_embedding_model = tf.keras.Model(
            inputs=user_input, outputs=user_embedding, name="user_embedding_extractor"
        )

        # Product embedding extractor
        product_input = tf.keras.layers.Input(shape=(1,))
        product_emb_layer = self.model.get_layer("product_embedding")
        product_embedding = product_emb_layer(product_input)
        product_embedding = tf.keras.layers.Flatten()(product_embedding)
        # L2 normalize for cosine similarity
        product_embedding = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1)
        )(product_embedding)

        self.product_embedding_model = tf.keras.Model(
            inputs=product_input,
            outputs=product_embedding,
            name="product_embedding_extractor",
        )

    def train(
        self,
        user_ids: np.ndarray,
        product_ids: np.ndarray,
        scores: np.ndarray,
        validation_split: Optional[float] = None,
        validation_data: Optional[
            Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
        shuffle: Optional[bool] = None,
        epochs: int = 50,
        batch_size: int = 256,
        callbacks: Optional[list] = None,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Train the matrix factorization model.

        Args:
            user_ids: Array of user IDs
            product_ids: Array of product IDs
            scores: Array of scores (float -1 to 1)
            validation_split: Fraction of data to use for validation
            validation_data: Tuple of (Tuple[np.ndarray, np.ndarray], np.ndarray) for validation
            shuffle: Whether to shuffle the data
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: Optional list of Keras callbacks
            verbose: Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
                ),
            ]

        # Train the model
        history = self.model.fit(
            x=[user_ids, product_ids],
            y=scores,
            validation_split=validation_split,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle,
        )

        # save model and parameters
        self.save_model()

        return history

    def predict(
        self, user_ids: np.ndarray, product_ids: np.ndarray, batch_size: int = 256
    ) -> np.ndarray:
        """
        Predict scores for user-product pairs.

        Args:
            user_ids: Array of user IDs
            product_ids: Array of product IDs
            batch_size: Prediction batch size

        Returns:
            Array of predicted scores (values between -1 and 1)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        predictions = self.model.predict(
            [user_ids, product_ids], batch_size=batch_size, verbose=0
        )

        return predictions.flatten()

    def get_user_embeddings(self, user_ids: np.ndarray) -> np.ndarray:
        """
        Extract normalized user embeddings for Milvus indexing.

        Args:
            user_ids: Array of user IDs

        Returns:
            Normalized user embeddings (shape: [n_users, embedding_dim])
        """
        if self.user_embedding_model is None:
            raise ValueError("Model not built. Call build_model() first.")

        embeddings = self.user_embedding_model.predict(user_ids, verbose=0)
        return embeddings

    def get_product_embeddings(self, product_ids: np.ndarray) -> np.ndarray:
        """
        Extract normalized product embeddings for Milvus indexing.

        Args:
            product_ids: Array of product IDs

        Returns:
            Normalized product embeddings (shape: [n_products, embedding_dim])
        """
        if self.product_embedding_model is None:
            raise ValueError("Model not built. Call build_model() first.")

        embeddings = self.product_embedding_model.predict(product_ids, verbose=0)
        return embeddings

    def save_model(self):
        """
        Save the model and its parameters to disk.
        """
        self.model.save("model/cf_model.keras")
        with open("model/params.json", "w") as f:
            json.dump(
                {
                    "n_users": self.n_users,
                    "n_products": self.n_products,
                    "embedding_dim": self.embedding_dim,
                    "learning_rate": self.learning_rate,
                    "l2_reg": self.l2_reg,
                    "dropout_rate": self.dropout_rate,
                },
                f,
            )
        print(
            "Model and parameters saved to model/cf_model.keras and model/params.json"
        )

    def load_model(self):
        """
        Load a saved model and its parameters from disk, then rebuild the model.

        Args:
            filepath: Path to the saved model file
        """
        # Load parameters first
        params_filepath = "model/params.json"
        if not os.path.exists(params_filepath):
            logger.error(
                f"Parameters file not found: {params_filepath}. Make sure the model was saved with train() method."
            )
            return

        with open(params_filepath, "r") as f:
            params = json.load(f)

        # Restore parameters
        self.n_users = params["n_users"]
        self.n_products = params["n_products"]
        self.embedding_dim = params["embedding_dim"]
        self.learning_rate = params["learning_rate"]
        self.l2_reg = params["l2_reg"]
        self.dropout_rate = params["dropout_rate"]

        # Load the model
        self.model = tf.keras.models.load_model("model/cf_model.keras")
        print("Model loaded from model/cf_model.keras and model/params.json")

        # Rebuild embedding models
        self._build_embedding_models()
