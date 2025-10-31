"""
ncf_model.py (UPDATED)

Neural Collaborative Filtering with fixes for overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from datetime import datetime
import logging
import pickle
import json

logger = logging.getLogger(__name__)


# ============================================================
# NCF MODEL DEFINITION (OPTIMIZED)
# ============================================================

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (OPTIMIZED VERSION)
    
    Changes:
    - Smaller embedding dimensions (64 → 32)
    - Higher dropout (0.2 → 0.3)
    - Better initialization
    """
    
    def __init__(self, num_users, num_products, embed_dim=32, mlp_layers=[128, 64, 32]):
        super().__init__()
        
        self.num_users = num_users
        self.num_products = num_products
        
        # === GMF (Generalized Matrix Factorization) Path ===
        self.user_embedding_gmf = nn.Embedding(num_users, embed_dim)
        self.product_embedding_gmf = nn.Embedding(num_products, embed_dim)
        
        # === MLP Path ===
        self.user_embedding_mlp = nn.Embedding(num_users, embed_dim)
        self.product_embedding_mlp = nn.Embedding(num_products, embed_dim)
        
        # MLP layers with LayerNorm (better than BatchNorm for small batches)
        mlp_modules = []
        input_size = embed_dim * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.LayerNorm(layer_size))  # Changed from BatchNorm
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.3))  # Increased from 0.2
            input_size = layer_size
        
        self.mlp = nn.Sequential(*mlp_modules)
        
        # === Fusion Layer ===
        self.predict_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, user_ids, product_ids):
        """
        Forward pass
        
        Args:
            user_ids: (batch_size,) tensor of user indices
            product_ids: (batch_size,) tensor of product indices
        
        Returns:
            scores: (batch_size,) tensor of predicted scores [0, 1]
        """
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        product_emb_gmf = self.product_embedding_gmf(product_ids)
        gmf_output = user_emb_gmf * product_emb_gmf
        
        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        product_emb_mlp = self.product_embedding_mlp(product_ids)
        mlp_input = torch.cat([user_emb_mlp, product_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        
        # Final prediction
        prediction = torch.sigmoid(self.predict_layer(concat))
        
        return prediction.squeeze(-1)
    
    def get_user_embedding(self, user_id):
        """Get user embedding for similarity computation"""
        with torch.no_grad():
            user_gmf = self.user_embedding_gmf(torch.LongTensor([user_id]))
            user_mlp = self.user_embedding_mlp(torch.LongTensor([user_id]))
            return torch.cat([user_gmf, user_mlp], dim=-1).numpy()
    
    def predict_batch(self, user_ids, product_ids):
        """Batch prediction for multiple user-product pairs"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor(user_ids)
            product_tensor = torch.LongTensor(product_ids)
            scores = self.forward(user_tensor, product_tensor)
            return scores.numpy()


# ============================================================
# DATASET FOR TRAINING (IMPROVED)
# ============================================================

class NCFDataset(Dataset):
    """
    Dataset for NCF training (IMPROVED VERSION)
    
    Changes:
    - Better negative sampling
    - Minimum interactions filter
    """
    
    def __init__(
        self,
        db_connection,
        days=90,
        negative_ratio=4,
        min_interactions_per_user=10,
        event_weight_map=None
    ):
        """
        Args:
            db_connection: Database connection
            days: Number of days to look back
            negative_ratio: Number of negative samples per positive sample
            min_interactions_per_user: Minimum interactions required per user
        """
        self.negative_ratio = negative_ratio
        self.min_interactions_per_user = min_interactions_per_user
        self.event_weight_map = event_weight_map or {
            'view': 0.2,
            'add_to_cart': 0.6,
            'purchase': 0.95
        }
        
        logger.info(f"Loading interactions from last {days} days...")
        interactions = self._load_interactions(db_connection, days)
        
        logger.info("Building user/product mappings...")
        self._build_mappings(interactions)
        
        logger.info("Preparing training samples...")
        self.samples = self._prepare_samples(interactions)
        
        logger.info(f"✅ Dataset ready: {len(self.user_id_to_idx)} users, "
                   f"{len(self.product_id_to_idx)} products, "
                   f"{len(self.samples)} samples")
    
    def _load_interactions(self, db, days):
        """Load interactions from database"""
        query = f"""
            SELECT 
                account_id,
                ref_id as product_id,
                event_type,
                timestamp
            FROM analytics_events
            WHERE timestamp >= NOW() - INTERVAL '{days} days'
                AND account_id IS NOT NULL
                AND ref_id IS NOT NULL
            ORDER BY timestamp DESC
        """
        
        cursor = db.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        interactions = []
        for row in rows:
            interactions.append({
                'user_id': row[0],
                'product_id': row[1],
                'event_type': row[2],
                'timestamp': row[3]
            })
        
        return interactions
    
    def _build_mappings(self, interactions):
        """Build user/product ID to index mappings"""
        # Count interactions per user
        user_interaction_counts = defaultdict(int)
        for i in interactions:
            user_interaction_counts[i['user_id']] += 1
        
        # Filter users with minimum interactions
        valid_users = {
            uid for uid, count in user_interaction_counts.items()
            if count >= self.min_interactions_per_user
        }
        
        logger.info(f"Filtered {len(valid_users)} users with >= {self.min_interactions_per_user} interactions")
        
        # Filter interactions
        interactions = [i for i in interactions if i['user_id'] in valid_users]
        
        # Get unique users and products
        unique_users = sorted(valid_users)
        unique_products = sorted(set(i['product_id'] for i in interactions))
        
        # Create mappings
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(unique_products)}
        
        # Reverse mappings
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_product_id = {idx: pid for pid, idx in self.product_id_to_idx.items()}
        
        # Build user -> products set for negative sampling
        self.user_interacted_products = defaultdict(set)
        for interaction in interactions:
            user_id = interaction['user_id']
            product_id = interaction['product_id']
            self.user_interacted_products[user_id].add(product_id)
        
        self.all_product_ids = set(unique_products)
        self.filtered_interactions = interactions
    
    def _prepare_samples(self, interactions):
        """Prepare positive and negative samples"""
        samples = []
        
        # Positive samples
        for interaction in self.filtered_interactions:
            user_id = interaction['user_id']
            product_id = interaction['product_id']
            event_type = interaction.get('event_type')
            
            user_idx = self.user_id_to_idx[user_id]
            product_idx = self.product_id_to_idx[product_id]
            label = self.event_weight_map.get(event_type, 1.0)
            label = float(np.clip(label, 0.0, 1.0))
            
            samples.append({
                'user_idx': user_idx,
                'product_idx': product_idx,
                'label': label,
                'binary_label': 1.0
            })
        
        # Negative samples
        logger.info("Generating negative samples...")
        for interaction in self.filtered_interactions:
            user_id = interaction['user_id']
            
            # Sample products user hasn't interacted with
            negative_candidates = list(
                self.all_product_ids - self.user_interacted_products[user_id]
            )
            
            if len(negative_candidates) >= self.negative_ratio:
                negative_products = np.random.choice(
                    negative_candidates,
                    size=self.negative_ratio,
                    replace=False
                )
                
                user_idx = self.user_id_to_idx[user_id]
                
                for neg_product in negative_products:
                    product_idx = self.product_id_to_idx[neg_product]
                    samples.append({
                        'user_idx': user_idx,
                        'product_idx': product_idx,
                        'label': 0.0,
                        'binary_label': 0.0
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'user_idx': sample['user_idx'],
            'product_idx': sample['product_idx'],
            'label': float(sample['label']),
            'binary_label': float(sample.get('binary_label', 1.0 if sample['label'] > 0 else 0.0))
        }


# ============================================================
# TRAINING PIPELINE (IMPROVED)
# ============================================================

class NCFTrainer:
    """Training pipeline for NCF model (IMPROVED VERSION)"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Lower learning rate with weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,  # Reduced from 0.001
            weight_decay=0.01  # Increased from 1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Monitor AUC (higher is better)
            factor=0.5,
            patience=3
            #True  # Removed for PyTorch compatibility
        )
        
        self.criterion = nn.BCELoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            user_ids = batch['user_idx'].to(self.device)
            product_ids = batch['product_idx'].to(self.device)
            labels = batch['label'].float().to(self.device)  # Weighted labels already scaled
            
            # Forward pass
            predictions = self.model(user_ids, product_ids)
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_binary_labels = []
        num_batches = 0
        
        for batch in val_loader:
            user_ids = batch['user_idx'].to(self.device)
            product_ids = batch['product_idx'].to(self.device)
            labels = batch['label'].float().to(self.device)  # Weighted labels
            binary_labels = batch.get('binary_label')
            if binary_labels is not None:
                binary_labels = binary_labels.float().to(self.device)
            else:
                binary_labels = (labels > 0).float()
            
            predictions = self.model(user_ids, product_ids)
            loss = self.criterion(predictions, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(predictions.cpu().numpy())
            all_binary_labels.extend(binary_labels.cpu().numpy())
        
        # Compute AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_binary_labels, all_predictions)
        except:
            auc = 0.5  # If AUC computation fails
        
        return total_loss / num_batches, auc
    
    def train(self, train_loader, val_loader, num_epochs=30, patience=10):
        """Full training loop with early stopping"""
        best_val_auc = 0.0
        patience_counter = 0
        
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_auc = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rates'].append(current_lr)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val AUC={val_auc:.4f}, "
                f"LR={current_lr:.6f}"
            )
            
            # Early stopping based on AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                self.save_checkpoint('ncf_model_best.pt')
                logger.info(f"✅ Saved best model (AUC={best_val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Stop if learning rate too small
            if current_lr < 1e-6:
                logger.info("Learning rate too small, stopping training")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation AUC: {best_val_auc:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {path}")


# ============================================================
# TRAINING SCRIPT
# ============================================================

def train_ncf_model(db_connection, save_dir='./models'):
    """
    Main training function
    
    Usage:
        import psycopg2
        db = psycopg2.connect(...)
        train_ncf_model(db)
    """
    
    # 1. Create dataset
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing dataset")
    logger.info("=" * 60)
    
    dataset = NCFDataset(
        db_connection,
        days=90,
        negative_ratio=4,
        min_interactions_per_user=10  # Filter users
    )
    
    # 2. Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 3. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 4. Initialize model
    logger.info("=" * 60)
    logger.info("STEP 2: Initializing model")
    logger.info("=" * 60)
    
    num_users = len(dataset.user_id_to_idx)
    num_products = len(dataset.product_id_to_idx)
    
    model = NCF(
        num_users=num_users,
        num_products=num_products,
        embed_dim=32,  # Reduced from 64
        mlp_layers=[128, 64, 32]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters")
    
    # 5. Train
    logger.info("=" * 60)
    logger.info("STEP 3: Training")
    logger.info("=" * 60)
    
    trainer = NCFTrainer(model)
    history = trainer.train(train_loader, val_loader, num_epochs=30, patience=10)
    
    # 6. Save final model and mappings
    logger.info("=" * 60)
    logger.info("STEP 4: Saving model and mappings")
    logger.info("=" * 60)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_products': num_products,
        'embed_dim': 32,
        'mlp_layers': [128, 64, 32]
    }, f'{save_dir}/ncf_model_final.pt')
    
    # Save mappings
    with open(f'{save_dir}/ncf_mappings.pkl', 'wb') as f:
        pickle.dump({
            'user_id_to_idx': dataset.user_id_to_idx,
            'product_id_to_idx': dataset.product_id_to_idx,
            'idx_to_user_id': dataset.idx_to_user_id,
            'idx_to_product_id': dataset.idx_to_product_id
        }, f)
    
    logger.info(f"✅ Model saved to {save_dir}/ncf_model_final.pt")
    logger.info(f"✅ Mappings saved to {save_dir}/ncf_mappings.pkl")
    
    return model, dataset, history


if __name__ == "__main__":
    # Example usage
    import psycopg2
    
    db = psycopg2.connect(
        host="localhost",
        database="your_database",
        user="your_user",
        password="your_password"
    )
    
    logging.basicConfig(level=logging.INFO)
    train_ncf_model(db, save_dir='./models')
