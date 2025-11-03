"""Embedding functions using MGTEEmbeddingFunction for content encoding"""
import logging
from typing import Dict, List
import numpy as np
from pymilvus.model.hybrid import MGTEEmbeddingFunction

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using MGTEEmbeddingFunction"""
    
    def __init__(self, use_fp16: bool = False, device: str = "cpu"):
        """Initialize embedding function"""
        self.ef = MGTEEmbeddingFunction(use_fp16=use_fp16, device=device)
        self.dense_dim = self.ef.dim["dense"]
    
    def encode_text(self, text: str) -> Dict[str, np.ndarray]:
        """Encode text query into dense and sparse vectors"""
        try:
            result = self.ef.encode_documents([text])
            return {
                "dense": result["dense"][0],
                "sparse": result["sparse"][[0]]
            }
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def encode_product(self, product_data: dict) -> Dict[str, np.ndarray]:
        """Encode product information into embeddings"""
        # Combine product text fields for encoding
        name = product_data.get("name", "")
        description = product_data.get("description", "")
        
        # Create text representation of product
        product_text = f"{name} {description}".strip()
        
        if not product_text:
            logger.warning(f"Empty product text for product {product_data.get('id')}")
            product_text = "product"
        
        return self.encode_text(product_text)
    
    def encode_products(self, products: List[dict]) -> Dict[str, np.ndarray]:
        """Encode multiple products"""
        # Combine text fields for each product
        texts = []
        encoded_products = []
        for product in products:
            name = product.get("name", "")
            description = product.get("description", "")
            product_text = f"{name} {description}".strip()
            if not product_text:
                product_text = "product"
            texts.append(product_text)
            encoded_products.append(self.encode_text(product_text))

        return encoded_products
