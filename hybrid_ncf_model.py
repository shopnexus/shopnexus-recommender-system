"""
hybrid_ncf_model.py

Hybrid NCF combining collaborative filtering + content features
Uses existing MGTE embeddings from Milvus
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class HybridNCF(nn.Module):
    """
    Hybrid Neural Collaborative Filtering
    
    Combines:
    1. User ID embeddings (learned)
    2. Product ID embeddings (collaborative signal)
    3. Product text embeddings (content signal from MGTE)
    """
    
    def __init__(
        self,
        num_users,
        num_products,
        embed_dim=32,  # ✅ Giảm từ 64 → 32
        text_embed_dim=768,
        dropout=0.3  # ✅ Tăng dropout
    ):
        super().__init__()
        
        # User embedding
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # Product ID embedding
        self.product_id_embedding = nn.Embedding(num_products, embed_dim)
        
        # Text projection (simplified)
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, 128),  # ✅ Giảm từ 256
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim)
        )
        
        # Simplified MLP (ít layers hơn)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),  # ✅ Giảm từ 256
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Proper initialization"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, user_ids, product_ids, product_text_embeddings):
        # Embeddings
        user_emb = self.user_embedding(user_ids)
        product_id_emb = self.product_id_embedding(product_ids)
        product_text_emb = self.text_projection(product_text_embeddings)
        
        # Concatenate
        combined = torch.cat([user_emb, product_id_emb, product_text_emb], dim=-1)
        
        # Predict
        score = torch.sigmoid(self.mlp(combined))
        return score.squeeze(-1)
    
    def predict_batch(self, user_ids, product_ids, product_text_embeddings):
        """Batch prediction wrapper"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(
                torch.LongTensor(user_ids),
                torch.LongTensor(product_ids),
                torch.FloatTensor(product_text_embeddings)
            )
            return scores.numpy()


class HybridNCFDataset:
    """
    Dataset for Hybrid NCF training
    Extends NCFDataset to include text embeddings
    """
    
    def __init__(
        self,
        mock_db,
        product_embeddings_dict: Dict[int, np.ndarray],  # product_id -> MGTE embedding
        days: int = 90,
        negative_ratio: int = 4
    ):
        """
        Args:
            mock_db: Mock database connection
            product_embeddings_dict: Dict mapping product_id to MGTE dense vector
            days: Days of interaction history
            negative_ratio: Negative samples per positive
        """
        self.negative_ratio = negative_ratio
        self.product_embeddings_dict = product_embeddings_dict
        
        logger.info(f"Loading interactions from last {days} days...")
        interactions = self._load_interactions(mock_db, days)
        
        logger.info("Building user/product mappings...")
        self._build_mappings(interactions)
        
        logger.info("Preparing training samples...")
        self.samples = self._prepare_samples(interactions)
        
        logger.info(f"✅ Hybrid NCF dataset ready: {len(self.user_id_to_idx)} users, "
                   f"{len(self.product_id_to_idx)} products, "
                   f"{len(self.samples)} samples")
    
    def _load_interactions(self, db, days):
        """Load interactions from mock DB"""
        cursor = db.cursor()
        cursor.execute(f"SELECT * FROM analytics_events WHERE timestamp >= NOW() - INTERVAL '{days} days'")
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
        """Build ID mappings"""
        unique_users = sorted(set(i['user_id'] for i in interactions))
        unique_products = sorted(set(i['product_id'] for i in interactions))
        
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(unique_products)}
        
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_product_id = {idx: pid for pid, idx in self.product_id_to_idx.items()}
        
        # User-product interactions
        self.user_interacted_products = {}
        for interaction in interactions:
            user_id = interaction['user_id']
            product_id = interaction['product_id']
            
            if user_id not in self.user_interacted_products:
                self.user_interacted_products[user_id] = set()
            self.user_interacted_products[user_id].add(product_id)
        
        self.all_product_ids = set(unique_products)
    
    def _prepare_samples(self, interactions):
        """Prepare training samples with text embeddings"""
        samples = []
        
        # Positive samples
        for interaction in interactions:
            user_id = interaction['user_id']
            product_id = interaction['product_id']
            
            # Skip if product doesn't have text embedding
            if product_id not in self.product_embeddings_dict:
                continue
            
            user_idx = self.user_id_to_idx[user_id]
            product_idx = self.product_id_to_idx[product_id]
            
            samples.append({
                'user_idx': user_idx,
                'product_idx': product_idx,
                'product_id': product_id,  # Keep for fetching text embedding
                'label': 1.0
            })
        
        # Negative samples
        logger.info("Generating negative samples...")
        for interaction in interactions:
            user_id = interaction['user_id']
            
            # Sample negative products
            negative_candidates = list(
                self.all_product_ids - self.user_interacted_products[user_id]
            )
            
            # Filter candidates that have text embeddings
            negative_candidates = [
                pid for pid in negative_candidates
                if pid in self.product_embeddings_dict
            ]
            
            if len(negative_candidates) >= self.negative_ratio:
                negative_products = np.random.choice(
                    negative_candidates,
                    size=self.negative_ratio,
                    replace=False
                )
                
                user_idx = self.user_id_to_idx[user_id]
                
                for neg_product_id in negative_products:
                    product_idx = self.product_id_to_idx[neg_product_id]
                    samples.append({
                        'user_idx': user_idx,
                        'product_idx': product_idx,
                        'product_id': neg_product_id,
                        'label': 0.0
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Fetch text embedding
        product_text_emb = self.product_embeddings_dict[sample['product_id']]
        
        return {
            'user_idx': sample['user_idx'],
            'product_idx': sample['product_idx'],
            'product_text_emb': torch.FloatTensor(product_text_emb),
            'label': float(sample['label'])
        }


class HybridNCFTrainer:
    """Training pipeline for Hybrid NCF"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # ✅ Lower learning rate
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,  # Giảm từ 0.001
            weight_decay=0.01  # Tăng regularization
        )
        
        # ✅ Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # ✅ Label smoothing để tránh overconfidence
        self.criterion = nn.BCELoss(reduction='mean')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            user_ids = batch['user_idx'].to(self.device)
            product_ids = batch['product_idx'].to(self.device)
            product_text_embs = batch['product_text_emb'].to(self.device)
            labels = batch['label'].float().to(self.device)
            
            # Forward
            predictions = self.model(user_ids, product_ids, product_text_embs)
            loss = self.criterion(predictions, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in val_loader:
            user_ids = batch['user_idx'].to(self.device)
            product_ids = batch['product_idx'].to(self.device)
            product_text_embs = batch['product_text_emb'].to(self.device)
            labels = batch['label'].float().to(self.device)
            
            predictions = self.model(user_ids, product_ids, product_text_embs)
            loss = self.criterion(predictions, labels)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_predictions)
        
        return total_loss / len(val_loader), auc
    
    def train(self, train_loader, val_loader, num_epochs=20, patience=5):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.evaluate(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
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
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'hybrid_ncf_best.pt')
                logger.info("✅ Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history


def train_hybrid_ncf(mock_db, product_embeddings_dict, save_dir='./models'):
    """
    Train Hybrid NCF model
    
    Args:
        mock_db: Mock database connection
        product_embeddings_dict: Dict[product_id -> MGTE embedding (768 dims)]
        save_dir: Directory to save models
    """
    from torch.utils.data import DataLoader, random_split
    import os
    import pickle
    
    # Create dataset
    dataset = HybridNCFDataset(
        mock_db,
        product_embeddings_dict,
        days=90,
        negative_ratio=4
    )
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)
    
    # Initialize model
    model = HybridNCF(
        num_users=len(dataset.user_id_to_idx),
        num_products=len(dataset.product_id_to_idx),
        embed_dim=32,  # Match constructor parameter
        text_embed_dim=768,
        dropout=0.3  # Match constructor parameter
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = HybridNCFTrainer(model)
    history = trainer.train(train_loader, val_loader, num_epochs=20)
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': len(dataset.user_id_to_idx),
        'num_products': len(dataset.product_id_to_idx),
        'embed_dim': 32,
        'text_embed_dim': 768,
        'dropout': 0.3
    }, f'{save_dir}/hybrid_ncf_final.pt')
    
    with open(f'{save_dir}/hybrid_ncf_mappings.pkl', 'wb') as f:
        pickle.dump({
            'user_id_to_idx': dataset.user_id_to_idx,
            'product_id_to_idx': dataset.product_id_to_idx,
            'idx_to_user_id': dataset.idx_to_user_id,
            'idx_to_product_id': dataset.idx_to_product_id
        }, f)
    
    logger.info(f"✅ Model saved to {save_dir}")
    
    return model, dataset, history