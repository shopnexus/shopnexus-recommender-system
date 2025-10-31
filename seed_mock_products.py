"""
seed_mock_products.py

Generate realistic mock products and insert into Milvus
"""

import logging
import random
import numpy as np
from typing import List, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockProductGenerator:
    """Generate realistic mock products for Milvus"""
    
    def __init__(self):
        # Product categories with subcategories
        self.categories = {
            'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Smartwatch', 'Headphones', 'Camera', 'TV', 'Speaker'],
            'Fashion': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Bag', 'Watch', 'Sunglasses', 'Jacket'],
            'Home': ['Sofa', 'Table', 'Chair', 'Lamp', 'Bed', 'Shelf', 'Mirror', 'Curtain'],
            'Sports': ['Running Shoes', 'Gym Equipment', 'Yoga Mat', 'Tennis Racket', 'Basketball', 'Bicycle', 'Swimsuit'],
            'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Magazine', 'Comic', 'Biography', 'Self-Help'],
            'Beauty': ['Lipstick', 'Foundation', 'Moisturizer', 'Shampoo', 'Perfume', 'Nail Polish', 'Face Mask'],
            'Toys': ['Action Figure', 'Doll', 'Puzzle', 'Board Game', 'LEGO', 'Remote Car', 'Plush Toy'],
            'Food': ['Snack', 'Beverage', 'Organic Food', 'Frozen Food', 'Spice', 'Coffee', 'Tea'],
            'Health': ['Vitamin', 'Supplement', 'First Aid', 'Thermometer', 'Blood Pressure Monitor'],
            'Automotive': ['Car Accessory', 'Motor Oil', 'Tire', 'Car Charger', 'Dash Cam', 'Air Freshener']
        }
        
        # Brands by category
        self.brands = {
            'Electronics': ['Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'HP', 'Lenovo', 'Asus', 'Xiaomi', 'Huawei'],
            'Fashion': ['Nike', 'Adidas', 'Puma', 'Zara', 'H&M', 'Uniqlo', 'Gucci', 'Louis Vuitton', 'Chanel'],
            'Home': ['Ikea', 'Muji', 'Ashley', 'Wayfair', 'West Elm', 'Crate & Barrel'],
            'Sports': ['Nike', 'Adidas', 'Under Armour', 'Reebok', 'Puma', 'New Balance', 'Asics'],
            'Books': ['Penguin', 'Harper Collins', 'Simon & Schuster', 'Macmillan', 'Random House'],
            'Beauty': ['L\'Oreal', 'Maybelline', 'MAC', 'Estee Lauder', 'Clinique', 'Shiseido', 'Dior'],
            'Toys': ['LEGO', 'Mattel', 'Hasbro', 'Fisher-Price', 'Bandai', 'Hot Wheels'],
            'Food': ['Nestle', 'Coca-Cola', 'PepsiCo', 'Unilever', 'Kraft', 'General Mills'],
            'Health': ['Johnson & Johnson', 'Pfizer', 'Abbott', 'Bayer', 'GNC', 'Nature Made'],
            'Automotive': ['Bosch', 'Michelin', 'Shell', 'Castrol', 'Garmin', '3M']
        }
        
        # Product adjectives
        self.adjectives = [
            'Premium', 'Professional', 'Ultra', 'Pro', 'Advanced', 'Smart', 'Portable',
            'Wireless', 'High-Performance', 'Luxury', 'Essential', 'Modern', 'Classic',
            'Innovative', 'Eco-Friendly', 'Durable', 'Compact', 'Elegant', 'Powerful'
        ]
        
        # Description templates
        self.description_templates = [
            "Experience the ultimate {adjective} {subcategory} from {brand}. Perfect for daily use.",
            "High-quality {subcategory} designed for maximum comfort and performance. {brand} quality guaranteed.",
            "Discover the new generation of {adjective} {subcategory}. {brand} brings innovation to your lifestyle.",
            "{brand}'s flagship {subcategory} combines style and functionality. {adjective} design for modern living.",
            "Upgrade your experience with this {adjective} {subcategory}. {brand} craftsmanship at its finest.",
        ]
    
    def generate_products(self, num_products: int = 2000) -> List[Dict]:
        """
        Generate mock products with realistic data
        
        Args:
            num_products: Number of products to generate
        
        Returns:
            List of product dictionaries
        """
        logger.info(f"Generating {num_products} mock products...")
        
        products = []
        product_id = 1
        
        # Calculate products per category
        products_per_category = max(1, num_products // len(self.categories))
        
        for category, subcategories in self.categories.items():
            brands = self.brands.get(category, ['Generic'])
            
            for _ in range(products_per_category):
                if product_id > num_products:
                    break
                
                # Random subcategory and brand
                subcategory = random.choice(subcategories)
                brand = random.choice(brands)
                adjective = random.choice(self.adjectives)
                
                # Generate name
                name = f"{brand} {adjective} {subcategory}"
                
                # Generate description
                description = random.choice(self.description_templates).format(
                    adjective=adjective.lower(),
                    subcategory=subcategory.lower(),
                    brand=brand
                )
                
                # Add more details to description
                description += f" {category} category. High-quality materials. "
                description += random.choice([
                    "Fast shipping available.",
                    "Best seller in its class.",
                    "Recommended by experts.",
                    "Limited time offer.",
                    "Customer favorite."
                ])
                
                # Generate realistic ratings and sales
                # Popular products: high rating + high sales
                # New products: lower sales
                popularity = random.random()
                
                if popularity > 0.8:  # Top 20% - very popular
                    rating_score = round(random.uniform(4.3, 5.0), 1)
                    rating_total = random.randint(500, 5000)
                    sold = random.randint(1000, 10000)
                elif popularity > 0.5:  # Mid tier - moderate popularity
                    rating_score = round(random.uniform(3.8, 4.5), 1)
                    rating_total = random.randint(50, 500)
                    sold = random.randint(100, 1000)
                else:  # Bottom tier - new or unpopular
                    rating_score = round(random.uniform(3.0, 4.2), 1)
                    rating_total = random.randint(10, 100)
                    sold = random.randint(10, 200)
                
                # Generate SKUs
                num_skus = random.randint(1, 5)
                skus = []
                for sku_idx in range(num_skus):
                    skus.append({
                        'sku_id': f"SKU-{product_id}-{sku_idx}",
                        'price': random.randint(100000, 50000000),
                        'stock': random.randint(0, 100),
                        'variant': random.choice(['Default', 'Color: Black', 'Color: White', 'Size: M', 'Size: L'])
                    })
                
                product = {
                    'id': product_id,
                    'name': name,
                    'description': description,
                    'brand': brand,
                    'category': category,
                    'is_active': random.random() > 0.1,  # 90% active
                    'rating': {
                        'score': rating_score,
                        'total': rating_total
                    },
                    'sold': sold,
                    'skus': skus
                }
                
                products.append(product)
                product_id += 1
        
        logger.info(f"✅ Generated {len(products)} products")
        self._print_statistics(products)
        
        return products
    
    def _print_statistics(self, products: List[Dict]):
        """Print statistics about generated products"""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATED PRODUCTS STATISTICS")
        logger.info("=" * 60)
        
        # Category distribution
        category_counts = {}
        brand_counts = {}
        
        for product in products:
            category = product['category']
            brand = product['brand']
            category_counts[category] = category_counts.get(category, 0) + 1
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        logger.info(f"\nTotal products: {len(products)}")
        logger.info(f"Unique categories: {len(category_counts)}")
        logger.info(f"Unique brands: {len(brand_counts)}")
        
        logger.info("\nTop 5 categories:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {category}: {count}")
        
        logger.info("\nTop 10 brands:")
        for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {brand}: {count}")
        
        # Rating distribution
        ratings = [p['rating']['score'] for p in products]
        logger.info(f"\nRating distribution:")
        logger.info(f"  Mean: {np.mean(ratings):.2f}")
        logger.info(f"  Median: {np.median(ratings):.2f}")
        logger.info(f"  Min: {min(ratings):.1f}")
        logger.info(f"  Max: {max(ratings):.1f}")
        
        # Sales distribution
        sales = [p['sold'] for p in products]
        logger.info(f"\nSales distribution:")
        logger.info(f"  Mean: {int(np.mean(sales))}")
        logger.info(f"  Median: {int(np.median(sales))}")
        logger.info(f"  Min: {min(sales)}")
        logger.info(f"  Max: {max(sales)}")
        
        logger.info("=" * 60 + "\n")


def seed_products_to_milvus(service, num_products: int = 2000):
    """
    Generate and insert mock products into Milvus
    
    Args:
        service: Service instance
        num_products: Number of products to generate
    
    Returns:
        List of generated products
    """
    logger.info("=" * 70)
    logger.info("SEEDING MOCK PRODUCTS TO MILVUS")
    logger.info("=" * 70)
    
    # Generate products
    generator = MockProductGenerator()
    products = generator.generate_products(num_products)
    
    # Insert into Milvus
    logger.info(f"\nInserting {len(products)} products into Milvus...")
    
    try:
        # Use service's update_products_batch with full embeddings
        count = service.update_products_batch(products, metadata_only=False)
        
        logger.info(f"✅ Successfully inserted {count} products into Milvus")
        
        # Verify insertion
        logger.info("\nVerifying insertion...")
        result = service.products_collection.query(
            expr="id > 0",
            output_fields=["id", "name", "brand", "category"],
            limit=5
        )
        
        logger.info(f"Sample products in Milvus:")
        for product in result:
            logger.info(f"  - {product['id']}: {product['name']} ({product['brand']}, {product['category']})")
        
        logger.info("\n" + "=" * 70)
        logger.info("SEEDING COMPLETED")
        logger.info("=" * 70)
        
        return products
        
    except Exception as e:
        logger.error(f"Failed to insert products: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # Standalone test
    from service import Service
    
    service = Service()
    products = seed_products_to_milvus(service, num_products=2000)
    
    logger.info(f"\n✅ Seeded {len(products)} products to Milvus")
    logger.info("You can now train NCF model with these products!")