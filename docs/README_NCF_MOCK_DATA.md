# NCF Recommendation System with Mock Data

Complete pipeline to train Neural Collaborative Filtering (NCF) model with realistic mock data stored in Milvus.

## 🎯 Overview

This system provides:
1. **Mock product generation** - Create realistic products with categories, brands, ratings
2. **Milvus integration** - Store products with vector embeddings for search
3. **Mock interaction generation** - Generate user interactions based on preferences
4. **NCF training** - Train collaborative filtering model
5. **Hybrid recommendations** - Combine content-based + collaborative filtering

## 📦 Files

### Core Files
- `service.py` - Main service class with Milvus operations and recommendation methods
- `ncf_model.py` - NCF model architecture and training code

### New Mock Data Pipeline
- `seed_mock_products.py` - Generate and insert mock products into Milvus
- `mock_data_generator_v2.py` - Generate user interactions using REAL products from Milvus
- `train_ncf_with_mock_data_v2.py` - Complete training pipeline

### Legacy Files (for reference)
- `mock_data_generator.py` - Old version (doesn't use Milvus)
- `train_ncf_with_mock_data.py` - Old version

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the complete pipeline that handles everything:

```bash
python train_ncf_with_mock_data_v2.py
```

This will:
1. ✅ Seed 2000 mock products to Milvus
2. ✅ Generate 5000 users with ~15-100 interactions each
3. ✅ Train NCF model
4. ✅ Plot training results
5. ✅ Test recommendations

### Option 2: Step-by-Step

#### Step 1: Seed Products to Milvus

```python
from service import Service
from seed_mock_products import seed_products_to_milvus

service = Service()
products = seed_products_to_milvus(service, num_products=2000)
```

#### Step 2: Generate Interactions

```python
from mock_data_generator_v2 import generate_mock_data_for_ncf

interactions, mock_db = generate_mock_data_for_ncf(
    service=service,
    num_users=5000,
    min_interactions_per_user=15,
    max_interactions_per_user=100,
    days=90,
    save_csv=True
)
```

#### Step 3: Train NCF Model

```python
from ncf_model import train_ncf_model

model, dataset, history = train_ncf_model(
    db_connection=mock_db,
    save_dir='./models'
)
```

## 📊 Mock Data Details

### Products (2000 total)

**Categories:**
- Electronics (Laptop, Smartphone, Camera, etc.)
- Fashion (T-Shirt, Jeans, Shoes, etc.)
- Home (Sofa, Table, Chair, etc.)
- Sports (Running Shoes, Gym Equipment, etc.)
- Books, Beauty, Toys, Food, Health, Automotive

**Brands:** Apple, Samsung, Sony, Nike, Adidas, Zara, Ikea, etc.

**Product Attributes:**
- Name, Description (with embeddings)
- Brand, Category
- Rating (1-5), Total Ratings
- Sold Count
- Multiple SKUs with price/stock/variant
- Active status

### Interactions (75K-500K total)

**Event Types:**
- `view` (60%) - Product views
- `add_to_cart` (20%) - Add to cart
- `purchase` (15%) - Purchases
- `rating` (5%) - Product ratings

**User Behavior:**
- Each user has 1-3 favorite categories
- Each user has 1-2 favorite brands
- 80% interactions match preferences (strong signal)
- 20% random exploration
- 15-100 interactions per user

**Time Period:** Last 90 days

## 🧪 Testing Recommendations

After training, test the recommendations:

```python
from service import Service

service = Service()

# NCF collaborative filtering
ncf_recs = service.get_ncf_recommendations(
    account_id=1,
    limit=10
)

# Hybrid (content + collaborative + popular)
hybrid_recs = service.get_hybrid_recommendations(
    account_id=1,
    limit=10,
    weights={
        'content': 0.5,
        'collaborative': 0.4,
        'popular': 0.1
    }
)

print("NCF Recommendations:")
for rec in ncf_recs:
    print(f"  {rec['name']} - Score: {rec['ncf_score']:.4f}")

print("\nHybrid Recommendations:")
for rec in hybrid_recs:
    print(f"  {rec['name']} - Score: {rec['hybrid_score']:.4f}")
```

## 📈 Expected Results

With the improved mock data:

- **Training AUC:** ~0.85-0.95
- **Validation AUC:** ~0.75-0.85
- **Training time:** 5-15 minutes (20 epochs)

The model should learn clear patterns:
- Users who like "Electronics + Apple" → Recommend Apple products
- Users who like "Fashion + Nike" → Recommend Nike shoes/apparel
- High-rated products get recommended more often

## 🔧 Configuration

### Product Generation

```python
# In seed_mock_products.py
products = seed_products_to_milvus(
    service,
    num_products=2000  # Adjust number of products
)
```

### Interaction Generation

```python
# In mock_data_generator_v2.py
interactions, mock_db = generate_mock_data_for_ncf(
    service=service,
    num_users=5000,  # Number of users
    min_interactions_per_user=15,  # Min interactions
    max_interactions_per_user=100,  # Max interactions
    days=90,  # Time period
    save_csv=True  # Save to CSV
)
```

### NCF Training

```python
# In ncf_model.py train_ncf_model()
model, dataset, history = train_ncf_model(
    db_connection=mock_db,
    embed_dim=64,  # Embedding dimension
    mlp_layers=[128, 64, 32],  # MLP layer sizes
    epochs=20,  # Training epochs
    batch_size=512,  # Batch size
    learning_rate=0.001,  # Initial learning rate
    save_dir='./models'
)
```

## 🐛 Troubleshooting

### No products in Milvus

```
Error: No products found in Milvus!
```

**Solution:** Run seed_mock_products.py first:
```bash
python seed_mock_products.py
```

### Products already exist

When running the pipeline again, you'll be asked:
```
Products already exist. Re-seed? (y/N):
```

- Type `y` to delete old products and create new ones
- Type `n` to use existing products

### Low AUC scores

If validation AUC < 0.65:

1. **Increase interactions per user:**
   ```python
   min_interactions_per_user=20  # instead of 15
   max_interactions_per_user=150  # instead of 100
   ```

2. **Increase number of users:**
   ```python
   num_users=10000  # instead of 5000
   ```

3. **Train longer:**
   ```python
   epochs=30  # instead of 20
   ```

### Out of memory

Reduce batch size:
```python
batch_size=256  # instead of 512
```

## 📁 Output Files

After training:

- `./models/ncf_model_final.pt` - Trained NCF model
- `./models/ncf_mappings.pkl` - User/product ID mappings
- `training_results.png` - Training curves (Loss, AUC, LR)
- `mock_interactions.csv` - Generated interactions (optional)

## 🎓 How It Works

### 1. Product Embeddings (Milvus)

Products are embedded using **MGTE (Multi-Grained Text Embeddings)**:
- **Dense vectors** (768-dim) for semantic search
- **Sparse vectors** for keyword matching
- Hybrid search combines both

### 2. User Preferences

User vectors are computed from interaction history:
```
user_vector = Σ (event_weight × product_vector)
```

Where event weights:
- view: 0.1
- add_to_cart: 0.3
- purchase: 0.6
- rating: 0.2

### 3. NCF Model

Neural Collaborative Filtering:
```
GMF: user_embedding ⊙ product_embedding
MLP: concat(user_embedding, product_embedding) → [128, 64, 32]
Output: σ(concat(GMF, MLP))
```

### 4. Hybrid Recommendations

Combines three methods:
```
final_score = 0.5 × content_score 
            + 0.4 × collaborative_score 
            + 0.1 × popularity_score
```

## 🚀 Production Usage

To use in production:

1. **Replace mock data with real data:**
   ```python
   # Instead of mock_db, use real database connection
   model, dataset, history = train_ncf_model(
       db_connection=psycopg2.connect(
           host="your_db_host",
           database="your_db_name",
           ...
       ),
       save_dir='./models'
   )
   ```

2. **Schedule periodic retraining:**
   - Daily: Process new events → Update user vectors
   - Weekly: Retrain NCF model with new data
   - Monthly: Add new products to Milvus

3. **Monitor performance:**
   - Track CTR (Click-Through Rate)
   - Track conversion rate
   - A/B test different recommendation strategies

## 📚 References

- [Neural Collaborative Filtering (He et al., 2017)](https://arxiv.org/abs/1708.05031)
- [Milvus Vector Database](https://milvus.io/)
- [MGTE Embeddings](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)

## 📝 License

MIT License - Feel free to use in your projects!
