keywords:
- Warm-start CF from previous embeddings; fix seed and hyperparams.
- Use early stopping/weight decay to prevent large drifts.
- Fuse cf model item_embedding with MGTE embedding for recommendation (CF + content-based)
- CF model (Matrix Factorization)






------------------
RESUME TRAINING:

Step 1. Load old model
old_model = tf.keras.models.load_model("cf_model_best.keras")

Step 2. Detect new users/items
new_n_users = self.client.hybrid_customers_collection.num_entities
new_n_items = self.client.hybrid_products_collection.num_entities

Step 3. Create a new model with larger embeddings
new_model = CFModel(
    n_users=new_n_users,
    n_items=new_n_items,
    embedding_dim=old_model.embedding_dim,
    ...
).build_model()

Step 4. Copy old weights

For both embeddings:

# Get old weights
old_user_weights = old_model.get_layer("user_embedding").get_weights()[0]
old_item_weights = old_model.get_layer("item_embedding").get_weights()[0]

# Get new embedding layers
new_user_emb = new_model.get_layer("user_embedding")
new_item_emb = new_model.get_layer("item_embedding")

# Copy old weights into the front
new_user_weights = new_user_emb.get_weights()[0]
new_item_weights = new_item_emb.get_weights()[0]

new_user_weights[:old_user_weights.shape[0]] = old_user_weights
new_item_weights[:old_item_weights.shape[0]] = old_item_weights

# Set them back
new_user_emb.set_weights([new_user_weights])
new_item_emb.set_weights([new_item_weights])


This way:

Old users/items keep their trained embeddings.

New users/items get new random rows (untrained yet).

Then you can continue training:

new_model.fit([...])


Finally, replace your old model reference:

self.model = new_model