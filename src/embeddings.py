"""Embedding functions using MGTEEmbeddingFunction for content encoding"""

import logging
from typing import Dict, List
from tqdm.auto import tqdm
from tqdm.auto import tqdm
import numpy as np
from pymilvus.model.hybrid import MGTEEmbeddingFunction

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using MGTEEmbeddingFunction"""

    def __init__(self, use_fp16: bool = False, device: str = "cpu"):
        """Initialize embedding function"""
        self.ef = MGTEEmbeddingFunction(use_fp16=use_fp16, device=device)
        self.dense_dim = self.ef.dim["dense"]

    def embed_text(self, text: str) -> Dict[str, np.ndarray]:
        """Embed text query into dense and sparse vectors"""
        try:
            result = self.ef.encode_documents([text])
            return {"dense": result["dense"][0], "sparse": result["sparse"][[0]]}
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[Dict[str, np.ndarray]]:
        return [self.embed_text(t) for t in tqdm(texts, desc="Embedding", unit="text")]
