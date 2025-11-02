import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pymilvus import Collection
from config import DESCRIPTION_LENGTH, event_weights

logger = logging.getLogger(__name__)


class DataUtils:
    """Utility class for data processing"""
    
    @staticmethod
    def truncate_text(text: str, max_length: int = DESCRIPTION_LENGTH) -> str:
        """Truncate text to max length"""
        if not text:
            return ""
        b = text.encode("utf-8")[:max_length]
        return b.decode("utf-8", errors="ignore")
    
    @staticmethod
    def extract_product_fields(product: Dict) -> tuple[int, str, str, str, Dict, bool, Dict, Dict, list[Any], Dict]:
        """Extract and normalize product fields matching TProductDetail schema"""
        product_id = int(product.get('id', 0))
        code = str(product.get('code', ''))
        name = str(product.get('name', ''))
        description = str(product.get('description', ''))
        
        # Brand as JSON object matching TProductDetail.brand
        brand = product.get('brand', {})
        if isinstance(brand, str):
            # Handle legacy string format
            brand = {'name': brand}
        
        is_active = bool(product.get('is_active', False))
        
        # Category as JSON object matching TProductDetail.category
        category = product.get('category', {})
        if isinstance(category, str):
            # Handle legacy string format
            category = {'name': category}
        
        # Rating as JSON object matching TProductDetail.rating
        rating = product.get('rating', {})
        if not isinstance(rating, dict):
            rating = {}
        
        # SKUs array matching TProductDetail.skus
        skus = product.get('skus', [])
        if not isinstance(skus, list):
            skus = []
        
        # Specifications as JSON object matching TProductDetail.specifications
        specifications = product.get('specifications', {})
        if not isinstance(specifications, dict):
            specifications = {}
        
        return product_id, code, name, description, brand, is_active, category, rating, skus, specifications


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
