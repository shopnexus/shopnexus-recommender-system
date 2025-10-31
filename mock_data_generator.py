"""
mock_data_generator.py (UPDATED)

Generate realistic synthetic interaction data with clear patterns
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class MockDataGenerator:
    """
    Generate realistic synthetic interaction data (IMPROVED VERSION)
    
    Changes:
    - Strong user preferences by category/brand
    - More interactions per user (min 15)
    - Clear patterns for model to learn
    """
    
    def __init__(self, service=None):
        """
        Args:
            service: Service instance (to get product data from Milvus)
        """
        self.service = service
        self.products = None
        self.product_metadata = {}
        
        # Event type weights
        self.event_types = ['view', 'add_to_cart', 'purchase', 'rating']
        self.event_weights = [0.60, 0.20, 0.15, 0.05]
        
        # Product categories and brands
        self.categories = [
            'Electronics', 'Fashion', 'Home', 'Sports', 'Books',
            'Beauty', 'Toys', 'Food', 'Health', 'Automotive'
        ]
        
        self.brands = [
            'Apple', 'Samsung', 'Sony', 'LG', 'Dell',
            'HP', 'Lenovo', 'Asus', 'Nike', 'Adidas',
            'Puma', 'Zara', 'H&M', 'Uniqlo', 'Ikea',
            'Panasonic', 'Canon', 'Nikon', 'Xiaomi', 'Huawei'
        ]
        
        # Load products from Milvus if service provided
        if service:
            self._load_products_from_milvus()
    
    def _load_products_from_milvus(self):
        """Load product data from Milvus"""
        try:
            logger.info("Loading products from Milvus...")
            
            results = self.service.products_collection.query(
                expr="id > 0",
                output_fields=["id", "name", "brand", "category", "rating_score", "sold"],
                limit=10000
            )
            
            self.products = results
            
            for product in results:
                self.product_metadata[product['id']] = {
                    'name': product.get('name', ''),
                    'brand': product.get('brand', ''),
                    'category': product.get('category', ''),
                    'rating': product.get('rating_score', 0),
                    'sold': product.get('sold', 0)
                }
            
            logger.info(f"✅ Loaded {len(self.products)} products from Milvus")
            
        except Exception as e:
            logger.error(f"Error loading products from Milvus: {e}")
            self.products = None
    
    def generate_interactions(
        self,
        num_users: int = 5000,
        num_products: int = 2000,
        num_interactions: int = None,
        min_interactions_per_user: int = 15,  # Increased from 5
        max_interactions_per_user: int = 100,
        days: int = 90
    ) -> List[Dict]:
        """
        Generate synthetic user-product interactions with clear patterns
        
        Args:
            num_users: Number of unique users
            num_products: Number of products to use
            num_interactions: Total interactions (computed if None)
            min_interactions_per_user: Minimum interactions per user
            max_interactions_per_user: Maximum interactions per user
            days: Number of days to spread interactions over
        
        Returns:
            List of interaction dicts
        """
        logger.info(f"Generating interactions with clear patterns...")
        
        # Get product IDs
        if self.products:
            available_products = [p['id'] for p in self.products[:num_products]]
        else:
            available_products = list(range(1, num_products + 1))
        
        # Assign categories and brands to products
        products_info = self._assign_product_metadata(available_products)
        
        # Generate user profiles with strong preferences
        logger.info("Creating user profiles with strong preferences...")
        user_profiles = self._generate_user_profiles_with_preferences(
            num_users,
            products_info
        )
        
        # Generate interactions based on preferences
        logger.info("Generating interactions based on user preferences...")
        interactions = self._generate_preference_based_interactions(
            user_profiles,
            products_info,
            min_interactions_per_user,
            max_interactions_per_user,
            days
        )
        
        # Sort by timestamp
        interactions.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"✅ Generated {len(interactions)} interactions")
        self._print_statistics(interactions, num_users)
        
        return interactions
    
    def _assign_product_metadata(self, product_ids):
        """Assign categories and brands to products"""
        products_info = {}
        
        for pid in product_ids:
            # Use existing metadata if available
            if pid in self.product_metadata:
                category = self.product_metadata[pid].get('category', random.choice(self.categories))
                brand = self.product_metadata[pid].get('brand', random.choice(self.brands))
            else:
                category = random.choice(self.categories)
                brand = random.choice(self.brands)
            
            products_info[pid] = {
                'category': category,
                'brand': brand
            }
        
        return products_info
    
    def _generate_user_profiles_with_preferences(self, num_users, products_info):
        """Generate users with strong category/brand preferences"""
        profiles = {}
        
        # Group products by category and brand
        products_by_category = defaultdict(list)
        products_by_brand = defaultdict(list)
        
        for pid, info in products_info.items():
            products_by_category[info['category']].append(pid)
            products_by_brand[info['brand']].append(pid)
        
        for user_id in range(1, num_users + 1):
            # Each user has 1-3 favorite categories (strong preference)
            num_fav_categories = random.randint(1, 3)
            favorite_categories = random.sample(
                list(products_by_category.keys()),
                min(num_fav_categories, len(products_by_category))
            )
            
            # Each user has 1-2 favorite brands
            num_fav_brands = random.randint(1, 2)
            favorite_brands = random.sample(
                list(products_by_brand.keys()),
                min(num_fav_brands, len(products_by_brand))
            )
            
            # Build preferred products list
            preferred_products = []
            for category in favorite_categories:
                preferred_products.extend(products_by_category[category][:20])
            for brand in favorite_brands:
                preferred_products.extend(products_by_brand[brand][:10])
            
            preferred_products = list(set(preferred_products))
            
            profiles[user_id] = {
                'favorite_categories': favorite_categories,
                'favorite_brands': favorite_brands,
                'preferred_products': preferred_products,
                'activity_level': np.random.exponential(scale=2.0)
            }
        
        return profiles
    
    def _generate_preference_based_interactions(
        self,
        user_profiles,
        products_info,
        min_interactions,
        max_interactions,
        days
    ):
        """Generate interactions that follow user preferences"""
        interactions = []
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        for user_id, profile in user_profiles.items():
            # Number of interactions for this user
            num_interactions = random.randint(min_interactions, max_interactions)
            
            for _ in range(num_interactions):
                # 80% match user preferences, 20% exploration
                if random.random() < 0.8 and profile['preferred_products']:
                    # Select from preferred products
                    product_id = random.choice(profile['preferred_products'])
                else:
                    # Random exploration
                    product_id = random.choice(list(products_info.keys()))
                
                # Event type (weighted)
                event_type = random.choices(
                    self.event_types,
                    weights=self.event_weights
                )[0]
                
                # Random timestamp
                timestamp = start_time + timedelta(
                    seconds=random.randint(0, int((end_time - start_time).total_seconds()))
                )
                
                # Metadata
                metadata = self._generate_metadata(event_type, product_id)
                
                interactions.append({
                    'account_id': user_id,
                    'ref_id': product_id,
                    'event_type': event_type,
                    'timestamp': timestamp,
                    'metadata': metadata
                })
        
        return interactions
    
    def _generate_metadata(self, event_type: str, product_id: int) -> Dict:
        """Generate metadata for interaction"""
        metadata = {}
        
        if event_type in ['add_to_cart', 'purchase']:
            metadata['quantity'] = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.70, 0.15, 0.08, 0.05, 0.02]
            )[0]
        
        if event_type == 'purchase':
            metadata['price'] = random.randint(100000, 50000000)
        
        if event_type == 'rating':
            rating = np.random.normal(4.0, 0.8)
            rating = max(1, min(5, rating))
            metadata['rating'] = round(rating, 1)
        
        return metadata
    
    def _print_statistics(self, interactions: List[Dict], num_users: int):
        """Print statistics about generated data"""
        event_counts = defaultdict(int)
        for interaction in interactions:
            event_counts[interaction['event_type']] += 1
        
        logger.info("\n" + "=" * 60)
        logger.info("GENERATED DATA STATISTICS")
        logger.info("=" * 60)
        
        logger.info(f"Total interactions: {len(interactions)}")
        logger.info(f"Unique users: {num_users}")
        logger.info(f"Unique products: {len(set(i['ref_id'] for i in interactions))}")
        
        logger.info("\nEvent type distribution:")
        for event_type, count in sorted(event_counts.items()):
            percentage = count / len(interactions) * 100
            logger.info(f"  {event_type}: {count} ({percentage:.1f}%)")
        
        # User activity distribution
        user_activity = defaultdict(int)
        for interaction in interactions:
            user_activity[interaction['account_id']] += 1
        
        activities = list(user_activity.values())
        logger.info(f"\nUser activity (interactions per user):")
        logger.info(f"  Mean: {np.mean(activities):.1f}")
        logger.info(f"  Median: {np.median(activities):.1f}")
        logger.info(f"  Min: {min(activities)}")
        logger.info(f"  Max: {max(activities)}")
        
        logger.info("=" * 60 + "\n")
    
    def save_to_file(self, interactions: List[Dict], filename: str = "mock_interactions.csv"):
        """Save interactions to CSV file"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            if not interactions:
                return
            
            fieldnames = ['account_id', 'ref_id', 'event_type', 'timestamp', 'metadata']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for interaction in interactions:
                row = interaction.copy()
                row['timestamp'] = interaction['timestamp'].isoformat()
                row['metadata'] = str(interaction['metadata'])
                writer.writerow(row)
        
        logger.info(f"✅ Saved {len(interactions)} interactions to {filename}")


class MockDBConnection:
    """Mock database connection for NCF training"""
    
    def __init__(self, interactions: List[Dict]):
        self.interactions = interactions
        self._cursor = None
    
    def cursor(self):
        self._cursor = MockCursor(self.interactions)
        return self._cursor
    
    def execute(self, query):
        return self._cursor
    
    def close(self):
        pass


class MockCursor:
    """Mock database cursor"""
    
    def __init__(self, interactions: List[Dict]):
        self.interactions = interactions
        self._results = None
    
    def execute(self, query):
        days = 90
        if 'INTERVAL' in query:
            import re
            match = re.search(r"INTERVAL '(\d+) days'", query)
            if match:
                days = int(match.group(1))
        
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered = [
            i for i in self.interactions
            if i['timestamp'] >= cutoff_date
        ]
        
        self._results = [
            (i['account_id'], i['ref_id'], i['event_type'], i['timestamp'])
            for i in filtered
        ]
        
        return self
    
    def fetchall(self):
        return self._results if self._results else []
    
    def close(self):
        pass


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_mock_data_for_ncf(
    service=None,
    num_users: int = 5000,
    num_products: int = 2000,
    min_interactions_per_user: int = 15,  # Increased
    max_interactions_per_user: int = 100,
    days: int = 90,
    save_csv: bool = False
) -> Tuple[List[Dict], MockDBConnection]:
    """
    Convenience function to generate mock data
    
    Returns:
        (interactions, mock_db_connection)
    """
    generator = MockDataGenerator(service)
    
    interactions = generator.generate_interactions(
        num_users=num_users,
        num_products=num_products,
        min_interactions_per_user=min_interactions_per_user,
        max_interactions_per_user=max_interactions_per_user,
        days=days
    )
    
    if save_csv:
        generator.save_to_file(interactions)
    
    mock_db = MockDBConnection(interactions)
    
    return interactions, mock_db


def load_interactions_from_csv(filename: str) -> Tuple[List[Dict], MockDBConnection]:
    """Load interactions from CSV file"""
    import csv
    from ast import literal_eval
    
    interactions = []
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            interactions.append({
                'account_id': int(row['account_id']),
                'ref_id': int(row['ref_id']),
                'event_type': row['event_type'],
                'timestamp': datetime.fromisoformat(row['timestamp']),
                'metadata': literal_eval(row['metadata'])
            })
    
    logger.info(f"✅ Loaded {len(interactions)} interactions from {filename}")
    
    mock_db = MockDBConnection(interactions)
    return interactions, mock_db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = MockDataGenerator()
    interactions = generator.generate_interactions(
        num_users=1000,
        num_products=500,
        min_interactions_per_user=15,
        max_interactions_per_user=50
    )
    
    print(f"\n✅ Generated {len(interactions)} interactions")