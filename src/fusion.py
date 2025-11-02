"""Fusion module for combining content and CF embeddings"""
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingFusion:
    """Fusion layer for combining content and CF embeddings"""
    
    def __init__(self, content_dim: int, cf_dim: int, output_dim: int = 768):
        """Initialize fusion layer
        
        Args:
            content_dim: Dimension of content embedding (MGTE dense vector)
            cf_dim: Dimension of CF embedding
            output_dim: Output dimension (default 768)
        """
        self.content_dim = content_dim
        self.cf_dim = cf_dim
        self.output_dim = output_dim
        
        # Create PyTorch linear layer
        input_dim = content_dim + cf_dim
        self.fusion_layer = nn.Linear(input_dim, output_dim, bias=True)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)
        
        self.fusion_layer.eval()  # Set to evaluation mode by default
    
    def fuse_item_embedding(
        self,
        mgte_dense: np.ndarray,
        cf: np.ndarray
    ) -> np.ndarray:
        """Fuse item embedding: combine MGTE dense and CF embeddings
        
        Args:
            mgte_dense: MGTE dense vector (content embedding)
            cf: CF embedding vector
            
        Returns:
            Fused embedding vector of shape (output_dim,)
        """
        # Ensure arrays are 1D and correct dimensions
        mgte_dense = np.asarray(mgte_dense).flatten()
        cf = np.asarray(cf).flatten()
        
        if len(mgte_dense) != self.content_dim:
            raise ValueError(
                f"MGTE dense vector dimension mismatch: "
                f"expected {self.content_dim}, got {len(mgte_dense)}"
            )
        if len(cf) != self.cf_dim:
            raise ValueError(
                f"CF vector dimension mismatch: "
                f"expected {self.cf_dim}, got {len(cf)}"
            )
        
        # Concatenate embeddings
        concat = np.concatenate([mgte_dense, cf])
        
        # Convert to torch tensor
        concat_tensor = torch.from_numpy(concat).float().unsqueeze(0)
        
        # Pass through linear layer
        with torch.no_grad():
            fused_tensor = self.fusion_layer(concat_tensor)
        
        # Convert back to numpy
        fused = fused_tensor.squeeze(0).numpy()
        
        return fused
    
    def fuse_user_embedding(
        self,
        recent_mgte_dense_avg: np.ndarray,
        cf: np.ndarray
    ) -> np.ndarray:
        """Fuse user embedding: combine recent MGTE dense average and CF embeddings
        
        Args:
            recent_mgte_dense_avg: Average of recent product MGTE dense vectors
            cf: CF user embedding vector
            
        Returns:
            Fused user embedding vector of shape (output_dim,)
        """
        # Ensure arrays are 1D and correct dimensions
        recent_mgte_dense_avg = np.asarray(recent_mgte_dense_avg).flatten()
        cf = np.asarray(cf).flatten()
        
        if len(recent_mgte_dense_avg) != self.content_dim:
            raise ValueError(
                f"Recent MGTE dense vector dimension mismatch: "
                f"expected {self.content_dim}, got {len(recent_mgte_dense_avg)}"
            )
        if len(cf) != self.cf_dim:
            raise ValueError(
                f"CF vector dimension mismatch: "
                f"expected {self.cf_dim}, got {len(cf)}"
            )
        
        # Concatenate embeddings
        concat = np.concatenate([recent_mgte_dense_avg, cf])
        
        # Convert to torch tensor
        concat_tensor = torch.from_numpy(concat).float().unsqueeze(0)
        
        # Pass through linear layer
        with torch.no_grad():
            fused_tensor = self.fusion_layer(concat_tensor)
        
        # Convert back to numpy
        fused = fused_tensor.squeeze(0).numpy()
        
        return fused
    
    def fuse_item_embeddings_batch(
        self,
        mgte_dense_batch: np.ndarray,
        cf_batch: np.ndarray
    ) -> np.ndarray:
        """Fuse multiple item embeddings in batch
        
        Args:
            mgte_dense_batch: Array of shape (batch_size, content_dim)
            cf_batch: Array of shape (batch_size, cf_dim)
            
        Returns:
            Array of shape (batch_size, output_dim)
        """
        mgte_dense_batch = np.asarray(mgte_dense_batch)
        cf_batch = np.asarray(cf_batch)
        
        if mgte_dense_batch.shape[0] != cf_batch.shape[0]:
            raise ValueError("Batch sizes must match")
        
        if mgte_dense_batch.shape[1] != self.content_dim:
            raise ValueError(
                f"MGTE dense batch dimension mismatch: "
                f"expected {self.content_dim}, got {mgte_dense_batch.shape[1]}"
            )
        if cf_batch.shape[1] != self.cf_dim:
            raise ValueError(
                f"CF batch dimension mismatch: "
                f"expected {self.cf_dim}, got {cf_batch.shape[1]}"
            )
        
        # Concatenate embeddings
        concat = np.concatenate([mgte_dense_batch, cf_batch], axis=1)
        
        # Convert to torch tensor
        concat_tensor = torch.from_numpy(concat).float()
        
        # Pass through linear layer
        with torch.no_grad():
            fused_tensor = self.fusion_layer(concat_tensor)
        
        # Convert back to numpy
        fused = fused_tensor.numpy()
        
        return fused
    
    def get_state_dict(self):
        """Get PyTorch state dict for saving"""
        return self.fusion_layer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load PyTorch state dict"""
        self.fusion_layer.load_state_dict(state_dict)
        self.fusion_layer.eval()

