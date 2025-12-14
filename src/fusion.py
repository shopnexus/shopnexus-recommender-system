"""Fusion module for combining content and CF embeddings with improved stability"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingFusion:
    """Fusion layer for combining content and CF embeddings with stability improvements."""

    def __init__(
        self,
        content_dim: int,
        cf_dim: int,
        normalize_output: bool = True,
    ):
        """
        Initialize fusion with per-branch LayerNorm and concatenation.

        Args:
            content_dim: Dimension of content embeddings
            cf_dim: Dimension of CF embeddings
        """
        self.content_dim = content_dim
        self.cf_dim = cf_dim
        self.fused_dim = content_dim + cf_dim
        self.normalize_output = normalize_output

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < eps:
            return vec
        return vec / norm

    def _validate_input(
        self, vec: np.ndarray, expected_dim: int, name: str
    ) -> np.ndarray:
        """Validate and preprocess input vector."""
        vec = np.asarray(vec).flatten()

        if len(vec) != expected_dim:
            raise ValueError(
                f"{name} vector dimension mismatch: expected {expected_dim}, got {len(vec)}"
            )

        # Check for invalid values
        if np.any(np.isnan(vec)):
            logger.warning(
                f"NaN values detected in {name} vector, replacing with zeros"
            )
            vec = np.nan_to_num(vec, nan=0.0)

        if np.any(np.isinf(vec)):
            logger.warning(f"Inf values detected in {name} vector, clipping")
            vec = np.nan_to_num(vec, posinf=1.0, neginf=-1.0)

        return vec

    def fuse_embeddings(
        self,
        content_vec: np.ndarray,
        cf_vec: np.ndarray,
    ) -> np.ndarray:
        """Fuse content and CF embeddings with improved stability."""
        # Validate inputs
        content = self._validate_input(content_vec, self.content_dim, "Content")
        cf = self._validate_input(cf_vec, self.cf_dim, "CF")
        
        fused = np.concatenate([content, cf])
        
        # Optional L2 normalization
        if self.normalize_output:
            fused = self._l2_normalize(fused)
        
        return fused

    def batch_fuse_embeddings(
        self,
        content_vecs: np.ndarray,
        cf_vecs: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """Efficiently fuse batches of embeddings."""
        content_vecs = np.asarray(content_vecs)
        cf_vecs = np.asarray(cf_vecs)
        
        if len(content_vecs) != len(cf_vecs):
            raise ValueError(f"Batch size mismatch: {len(content_vecs)} vs {len(cf_vecs)}")
        
        # Validate dimensions
        if content_vecs.shape[1] != self.content_dim:
            raise ValueError(
                f"Content vector dimension mismatch: expected {self.content_dim}, "
                f"got {content_vecs.shape[1]}"
            )
        if cf_vecs.shape[1] != self.cf_dim:
            raise ValueError(
                f"CF vector dimension mismatch: expected {self.cf_dim}, "
                f"got {cf_vecs.shape[1]}"
            )
        
        # Check for invalid values (vectorized)
        if np.any(np.isnan(content_vecs)) or np.any(np.isnan(cf_vecs)):
            if verbose:
                logger.warning("NaN values detected, replacing with zeros")
            content_vecs = np.nan_to_num(content_vecs, nan=0.0)
            cf_vecs = np.nan_to_num(cf_vecs, nan=0.0)
        
        if np.any(np.isinf(content_vecs)) or np.any(np.isinf(cf_vecs)):
            if verbose:
                logger.warning("Inf values detected, clipping")
            content_vecs = np.nan_to_num(content_vecs, posinf=1.0, neginf=-1.0)
            cf_vecs = np.nan_to_num(cf_vecs, posinf=1.0, neginf=-1.0)
        
        # Vectorized concatenation
        fused = np.concatenate([content_vecs, cf_vecs], axis=1)
        
        # Vectorized normalization
        if self.normalize_output:
            norms = np.linalg.norm(fused, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            fused = fused / norms
        
        return fused.astype(np.float32)
