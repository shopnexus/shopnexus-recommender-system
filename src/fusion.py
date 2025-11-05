"""Fusion module for combining content and CF embeddings with improved stability"""

import numpy as np
import torch


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
            print(
                f"Warning: NaN values detected in {name} vector, replacing with zeros"
            )
            vec = np.nan_to_num(vec, nan=0.0)

        if np.any(np.isinf(vec)):
            print(f"Warning: Inf values detected in {name} vector, clipping")
            vec = np.nan_to_num(vec, posinf=1.0, neginf=-1.0)

        return vec

    def fuse_embeddings(
        self,
        content_vec: np.ndarray,
        cf_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse content and CF embeddings with improved stability.

        Args:
            content_vec: Content embedding vector
            cf_vec: CF embedding vector

        Returns:
            Fused and L2-normalized embedding
        """
        # Validate inputs
        content = self._validate_input(content_vec, self.content_dim, "Content")
        cf = self._validate_input(cf_vec, self.cf_dim, "CF")

        with torch.no_grad():
            # Convert to torch tensors
            content_t = torch.from_numpy(content).float().unsqueeze(0)
            cf_t = torch.from_numpy(cf).float().unsqueeze(0)

            # Concatenate
            fused = torch.cat([content_t, cf_t], dim=1)

            # Convert back to numpy
            out = fused.squeeze(0).numpy()

        # Optional L2 normalization for cosine/IP similarity backends
        if self.normalize_output:
            out = self._l2_normalize(out)

        return out

    def batch_fuse_embeddings(
        self, content_vecs: np.ndarray, cf_vecs: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """
        Efficiently fuse batches of embeddings with progress tracking.

        Args:
            content_vecs: Batch of content embeddings (N x content_dim)
            cf_vecs: Batch of CF embeddings (N x cf_dim)
            verbose: If True, show progress

        Returns:
            Batch of fused embeddings (N x output_dim)
        """
        content_vecs = np.asarray(content_vecs)
        cf_vecs = np.asarray(cf_vecs)

        if len(content_vecs) != len(cf_vecs):
            raise ValueError(
                f"Batch size mismatch: {len(content_vecs)} vs {len(cf_vecs)}"
            )

        n_samples = len(content_vecs)

        # Pre-allocate output array
        fused_embeddings = np.zeros((n_samples, self.fused_dim), dtype=np.float32)

        # Process in batches for memory efficiency
        batch_size = 1000
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)

            if verbose and i % 10000 == 0:
                print(f"Processing: {i}/{n_samples}")

            # Process batch with vectorization where possible
            with torch.no_grad():
                content_batch = content_vecs[i:batch_end]
                cf_batch = cf_vecs[i:batch_end]

                # Validate batch
                for j in range(len(content_batch)):
                    idx = i + j
                    content = self._validate_input(
                        content_batch[j], self.content_dim, f"Content[{idx}]"
                    )
                    cf = self._validate_input(cf_batch[j], self.cf_dim, f"CF[{idx}]")

                    # Convert to tensors
                    content_t = torch.from_numpy(content).float().unsqueeze(0)
                    cf_t = torch.from_numpy(cf).float().unsqueeze(0)

                    # Concatenate
                    fused = torch.cat([content_t, cf_t], dim=1)
                    out = fused.squeeze(0).numpy()
                    if self.normalize_output:
                        out = self._l2_normalize(out)

                    fused_embeddings[idx] = out

        return fused_embeddings
