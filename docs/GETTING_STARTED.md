# 🚀 Getting Started - NCF Recommendation System

Complete guide to train and use the NCF recommendation system with mock data in Milvus.

## 📋 Prerequisites

1. **Python 3.8+**
2. **Milvus** running on localhost:19530
3. **Required packages:**
```bash
pip install pymilvus torch numpy scikit-learn matplotlib
```

## 🎯 Three Ways to Get Started

### Option 1: Quick Demo (Fastest - 3-5 minutes)

Perfect for testing and understanding the system:

```bash
python quick_demo.py
```

This runs with:
- 500 products
- 1000 users
- 10 training epochs

### Option 2: Full Pipeline (Recommended - 10-15 minutes)

Complete production-ready pipeline:

```bash
python train_ncf_with_mock_data_v2.py
```

This runs with:
- 2000 products
- 5000 users
- 20 training epochs
- Better accuracy

### Option 3: Step-by-Step (For Learning)

Run each step manually:

#### Step 1: Seed Products

```python
from service import Service
from seed_mock_products import seed_products_to_milvus

service = Service()
products = seed_products_to_milvus(service, num_products=2000)
```

**Output:**
```
✅ Generated 2000 products
✅ Successfully inserted 2000 products into Milvus
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

**Output:**
```
✅ Generated 250000 interactions
User activity (interactions per user):
  Mean: 50.0
  Min: 15
  Max: 100
```

#### Step 3: Train NCF Model

```python
from ncf_model import train_ncf_model
import os

os.makedirs('./models', exist_ok=True)

model, dataset, history = train_ncf_model(
    db_connection=mock_db,
    save_dir='./models'
)
```

**Output:**
```
Epoch 20/20: Train Loss: 0.1234 | Val Loss: 0.1567 | Val AUC: 0.8567
✅ Training completed! Best AUC: 0.8567
```

#### Step 4: Test Recommendations

```python
from service import Service

# Reload service to load NCF model
service = Service()

# Get NCF recommendations
ncf_recs = service.get_ncf_recommendations(
    account_id=1,
    limit=10
)

# Print results
for i, rec in enumerate(ncf_recs, 1):
    print(f"{i}. {rec['name']}")
    print(f"   Score: {rec['ncf_score']:.4f} | Rating: {rec['rating']:.1f}")
```

**Output:**
```
1. Apple Premium Laptop
   Score: 0.9234 | Rating: 4.5
2. Samsung Ultra Smartphone
   Score: 0.8956 | Rating: 4.7
...
```

## 📊 Analyzing Results

After training, analyze the recommendations:

```bash
python analyze_recommendations.py
```

This will:
1. Test recommendations for sample users
2. Show diversity metrics
3. Compare recommendation methods
4. Generate visualization plots

**Output files:**
- `recommendation_analysis.png` - Visual analysis
- Console output with diversity metrics

## 🔍 Understanding the Output

### Model Files

After training, you'll have:

```
./models/
├── ncf_model_final.pt       # Trained NCF model (~50MB)
├── ncf_mappings.pkl          # User/Product ID mappings (~1MB)
```

### Training Plots

`training_results.png` contains 3 plots:

1. **Loss curves** - Should decrease over epochs
2. **AUC curve** - Should increase (>0.75 is good)
3. **Learning rate** - Shows LR schedule

### Recommendations

Each recommendation contains:

```python
{
    'id': 123,
    'name': 'Apple Premium Laptop',
    'brand': 'Apple',
    'rating': 4.5,
    'sold': 5000,
    'ncf_score': 0.9234,  # or 'hybrid_score' for hybrid
    'source': 'ncf_collaborative_filtering'
}
```

## 🎓 How to Use in Your Application

### Basic Usage

```python
from service import Service

# Initialize once
service = Service()

# Get recommendations for a user
recommendations = service.get_ncf_recommendations(
    account_id=user_id,
    limit=20,
    exclude_interacted=True  # Don't recommend already seen products
)

# Display to user
for rec in recommendations:
    print(f"{rec['name']} - {rec['brand']}")
    print(f"Rating: {rec['rating']}/5 | Sold: {rec['sold']}")
```

### Advanced: Hybrid Recommendations

Combine multiple recommendation strategies:

```python
recommendations = service.get_hybrid_recommendations(
    account_id=user_id,
    limit=20,
    weights={
        'content': 0.5,        # Content-based (user vector)
        'collaborative': 0.4,   # NCF collaborative filtering
        'popular': 0.1         # Popular products
    }
)
```

### Handling Cold Start

For new users without interaction history:

```python
recommendations = service.get_ncf_recommendations(
    account_id=new_user_id,
    limit=20
)

# NCF automatically falls back to popular products
# Check the 'source' field
if recommendations and recommendations[0]['source'] == 'ncf_cold_start_popular':
    print("New user - showing popular products")
```

## 🔧 Configuration

### Adjust Mock Data Scale

**Small dataset (testing):**
```python
# quick_demo.py settings
products: 500
users: 1000
interactions_per_user: 10-50
epochs: 10
time: ~3 minutes
```

**Medium dataset (development):**
```python
products: 2000
users: 5000
interactions_per_user: 15-100
epochs: 20
time: ~10 minutes
```

**Large dataset (production simulation):**
```python
products: 5000
users: 10000
interactions_per_user: 20-150
epochs: 30
time: ~30 minutes
```

### Tune Model Performance

If AUC is low (<0.65):

1. **More data:**
```python
num_users=10000
min_interactions_per_user=20
```

2. **Larger model:**
```python
embed_dim=128  # instead of 64
mlp_layers=[256, 128, 64]  # instead of [128, 64, 32]
```

3. **Train longer:**
```python
epochs=30  # instead of 20
```

If training is too slow:

1. **Smaller batch:**
```python
batch_size=256  # instead of 512
```

2. **Fewer epochs:**
```python
epochs=10  # instead of 20
```

## 📈 Expected Performance

### Mock Data Metrics

- **Users:** 5000
- **Products:** 2000
- **Interactions:** ~250,000
- **Interaction density:** 2.5%

### Model Performance

- **Training AUC:** 0.90-0.95
- **Validation AUC:** 0.75-0.85
- **Training time:** 10-15 minutes
- **Inference time:** <10ms per user

### Recommendation Quality

- **Diversity:** 100-200 unique products in top-10 recommendations across 100 users
- **Personalization:** 80%+ recommendations match user's preferred categories/brands
- **Quality:** Average rating of recommendations: 4.0-4.5

## 🐛 Troubleshooting

### Error: "No products found in Milvus"

**Solution:** Run seed first:
```bash
python seed_mock_products.py
```

### Error: "NCF model not found"

**Solution:** Train the model first:
```bash
python train_ncf_with_mock_data_v2.py
```

### Low AUC (<0.60)

**Possible causes:**
1. Not enough training data
2. Weak user preferences in mock data
3. Model underfitting

**Solutions:**
- Increase `num_users` and `min_interactions_per_user`
- Train longer (`epochs=30`)
- Use larger model (`embed_dim=128`)

### Out of Memory

**Solutions:**
- Reduce `batch_size` to 256 or 128
- Use smaller `embed_dim` (32 instead of 64)
- Reduce number of users/products in mock data

### Milvus connection error

**Solution:**
```bash
# Check if Milvus is running
docker ps | grep milvus

# Start Milvus if needed
docker-compose up -d
```

## 📚 Next Steps

1. **Explore the code:**
   - `service.py` - Main recommendation logic
   - `ncf_model.py` - Neural network architecture
   - `seed_mock_products.py` - Data generation

2. **Customize for your needs:**
   - Modify product categories in `seed_mock_products.py`
   - Adjust user preference patterns in `mock_data_generator_v2.py`
   - Tune model hyperparameters in `train_ncf_with_mock_data_v2.py`

3. **Integrate with real data:**
   - Replace `mock_db` with real database connection
   - Use real products from your catalog
   - Use real user interactions from analytics

4. **Deploy to production:**
   - Set up periodic retraining (weekly/monthly)
   - Add A/B testing
   - Monitor CTR and conversion rates

## 💡 Tips

1. **Start small:** Use `quick_demo.py` to verify everything works
2. **Monitor training:** Watch the AUC - should improve each epoch
3. **Test recommendations:** Use `analyze_recommendations.py` to check quality
4. **Iterate:** Adjust parameters based on your needs

## 🆘 Getting Help

Check these files for more information:
- `README_NCF_MOCK_DATA.md` - Detailed documentation
- `ncf_model.py` - Model architecture and training code
- `service.py` - Recommendation methods

## ✅ Checklist

Before running:
- [ ] Milvus is running
- [ ] Python packages installed
- [ ] `./models/` directory exists (or will be created)

After training:
- [ ] `training_results.png` shows improving AUC
- [ ] Recommendations look relevant for test users
- [ ] Model files saved in `./models/`

Ready to deploy:
- [ ] AUC > 0.75
- [ ] Recommendations are diverse
- [ ] Cold start handled properly
- [ ] Inference time < 100ms

## 🎉 You're All Set!

You now have a working NCF recommendation system. Start with `quick_demo.py` and scale up from there!

Questions? Check the main README or analyze the code in `service.py` and `ncf_model.py`.

Happy recommending! 🚀
