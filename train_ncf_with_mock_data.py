"""
train_ncf_with_mock_data.py (UPDATED)

Train NCF model with improved mock data
"""

import logging
from mock_data_generator import generate_mock_data_for_ncf
from ncf_model import train_ncf_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Train NCF model with improved mock data"""
    
    logger.info("=" * 70)
    logger.info("NCF TRAINING WITH IMPROVED MOCK DATA")
    logger.info("=" * 70)
    
    # Generate mock data with better patterns
    logger.info("\nStep 1: Generating realistic interaction data...")
    
    interactions, mock_db = generate_mock_data_for_ncf(
        service=None,
        num_users=5000,
        num_products=2000,
        min_interactions_per_user=15,  # Increased
        max_interactions_per_user=100,
        days=90,
        save_csv=True
    )
    
    logger.info(f"✅ Generated {len(interactions)} interactions")
    
    # Train NCF model
    logger.info("\nStep 2: Training NCF model with fixes...")
    
    try:
        model, dataset, history = train_ncf_model(
            db_connection=mock_db,
            save_dir='./models'
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Users: {len(dataset.user_id_to_idx)}")
        logger.info(f"Products: {len(dataset.product_id_to_idx)}")
        logger.info(f"Training samples: {len(dataset)}")
        logger.info(f"Best val AUC: {max(history['val_auc']):.4f}")
        logger.info("\nModel saved to: ./models/ncf_model_final.pt")
        logger.info("Mappings saved to: ./models/ncf_mappings.pkl")
        
        # Plot training curves
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Loss
            axes[0].plot(history['train_loss'], label='Train')
            axes[0].plot(history['val_loss'], label='Val')
            axes[0].set_title('Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # AUC
            axes[1].plot(history['val_auc'])
            axes[1].axhline(y=0.5, color='r', linestyle='--', label='Random')
            axes[1].set_title('Validation AUC')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('AUC')
            axes[1].legend()
            axes[1].grid(True)
            
            # Learning Rate
            axes[2].plot(history['learning_rates'])
            axes[2].set_title('Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('LR')
            axes[2].set_yscale('log')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig('training_results.png', dpi=300)
            logger.info("\n✅ Training curves saved to training_results.png")
        except Exception as e:
            logger.warning(f"Could not plot results: {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n✅ ALL DONE! Model ready to use.")


if __name__ == "__main__":
    main()