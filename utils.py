import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pymilvus import Collection
from config import MAX_LENGTH_EMBED

logger = logging.getLogger(__name__)


class DataUtils:
    """Utility class for data processing"""
    
    @staticmethod
    def truncate_text(text: str, max_length: int = MAX_LENGTH_EMBED) -> str:
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
    
    @staticmethod
    def create_product_text_content(name: str, description: str, brand: Dict, category: Dict) -> str:
        """Create combined text content for embedding from TProductDetail schema"""
        # Extract brand name from brand object
        brand_name = brand.get('name', '') if isinstance(brand, dict) else str(brand)
        
        # Extract category name from category object
        category_name = category.get('name', '') if isinstance(category, dict) else str(category)
        
        text_content = f"{name}. {description}. Brand: {brand_name}. Category: {category_name}"
        return DataUtils.truncate_text(text_content, MAX_LENGTH_EMBED)


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


class MilvusOperations:
    """Helper class for Milvus operations"""
    
    @staticmethod
    def query_by_ids(collection: Collection, ids: List[int], output_fields: List[str]) -> List[Dict]:
        """Query collection by list of IDs"""
        if not ids:
            return []
        id_list = ','.join(map(str, ids))
        expr = f"id in [{id_list}]"
        return collection.query(expr=expr, output_fields=output_fields, limit=len(ids))
    
    @staticmethod
    def query_by_account_id(collection: Collection, account_id: int, output_fields: List[str]) -> List[Dict]:
        """Query collection by account ID"""
        return collection.query(expr=f"account_id == {account_id}", output_fields=output_fields, limit=1)
    
    @staticmethod
    def upsert_and_flush(collection: Collection, entities: List[List], **kwargs):
        """Upsert entities and flush collection"""
        collection.upsert(entities, None, None, **kwargs)
        collection.flush()

