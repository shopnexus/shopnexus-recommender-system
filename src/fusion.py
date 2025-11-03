"""Fusion module for combining content and CF embeddings"""
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingFusion:
    """Fusion layer for combining content and CF embeddings with safety-first normalization."""
    
    def __init__(self, content_dim: int, cf_dim: int):
        """Initialize fusion with per-branch LayerNorm and no shrinking. Output dim will be content_dim + cf_dim."""
        self.content_dim = content_dim
        self.cf_dim = cf_dim
        
        # Per-branch normalization to prevent dominance from scale differences
        self.content_norm = nn.LayerNorm(content_dim)
        self.cf_norm = nn.LayerNorm(cf_dim)

        self.content_norm.eval()
        self.cf_norm.eval()

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < eps:
            return vec
        return vec / norm
    
    def fuse_embeddings(self, content_vec: np.ndarray, cf_vec: np.ndarray) -> np.ndarray:
        """General fusion for a single pair of embeddings with normalization and MLP."""
        content = np.asarray(content_vec).flatten()
        cf = np.asarray(cf_vec).flatten()

        if len(content) != self.content_dim:
            raise ValueError(
                f"Content vector dimension mismatch: expected {self.content_dim}, got {len(content)}"
            )
        if len(cf) != self.cf_dim:
            raise ValueError(
                f"CF vector dimension mismatch: expected {self.cf_dim}, got {len(cf)}"
            )

        with torch.no_grad():
            content_t = torch.from_numpy(content).float().unsqueeze(0)
            cf_t = torch.from_numpy(cf).float().unsqueeze(0)
            # Per-branch LayerNorm, then concat directly (no shrink)
            content_n = self.content_norm(content_t)
            cf_n = self.cf_norm(cf_t)
            fused = torch.cat([content_n, cf_n], dim=1)
            out = fused.squeeze(0).numpy()

        # L2 normalize output for stable COSINE similarity
        return self._l2_normalize(out)
    
    def get_state_dict(self):
        """Get PyTorch state dict for saving"""
        return self.fusion_layer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load PyTorch state dict"""
        self.fusion_layer.load_state_dict(state_dict)
        self.fusion_layer.eval()

