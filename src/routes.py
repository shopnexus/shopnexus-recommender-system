import time
import logging
from datetime import datetime
from typing import Dict
from flask import request, jsonify

logger = logging.getLogger(__name__)


def register_routes(app, service):
    """Register all Flask routes"""
    
    @app.route("/search", methods=["POST"])
    def search():
        """Product semantic search endpoint"""
        data = request.json
        query = data.get("query")
        if not query:
            return jsonify({"error": "Missing query"}), 400

        limit = data.get("limit", 10)
        weights = data.get("weights", {"dense": 1.0, "sparse": 1.0})
        offset = data.get("offset", 0)

        result = service.semantic_search(
            query,
            dense_weight=weights.get("dense", 1.0),
            sparse_weight=weights.get("sparse", 1.0),
            offset=offset,
            limit=limit,
        )

        return jsonify(result)

    @app.route("/user/<int:account_id>/recommendations", methods=["GET"])
    def get_recommendations(account_id):
        """Get product recommendations for a user based on their preferences"""
        limit = request.args.get("limit", 10, type=int)
        results = service.recommend(account_id, limit=limit)
        return jsonify(results)


    @app.route("/analytics/process", methods=["POST"])
    def process_events():
        """Process analytics events endpoint"""
        start_time = time.time()
        data = request.json
        events = data.get("events", [])

        if not events:
            return jsonify({"error": "No events provided"}), 400

        process_start = time.time()
        service.process_events_batch(events)
        process_time = time.time() - process_start
        total_time = time.time() - start_time

        logger.info(f"Analytics processing completed: {len(events)} events in {process_time:.3f}s (total: {total_time:.3f}s)")

        return jsonify(create_performance_response(
            f"Processed {len(events)} events", len(events), process_time, total_time
        ))


    @app.route("/products", methods=["POST"])
    def update_products():
        """Update products in Milvus collection"""
        start_time = time.time()
        data = request.json
        products = data.get("products", [])
        metadata_only = data.get("metadata_only", False)

        if not products:
            return jsonify({"error": "No products provided"}), 200

        process_start = time.time()
        processed_count = service.update_products_batch(products, metadata_only=metadata_only)
        process_time = time.time() - process_start
        total_time = time.time() - start_time

        logger.info(f"Products update completed: {processed_count} products processed in {process_time:.3f}s (total: {total_time:.3f}s)")

        return jsonify(create_performance_response(
            f"Successfully updated {processed_count} products", processed_count, process_time, total_time
        ))

    @app.route("/training/train", methods=["POST"])
    def train_cf_model():
        """Trigger CF model training"""
        start_time = time.time()
        data = request.json or {}
        
        epochs = data.get("epochs", 50)
        batch_size = data.get("batch_size", 256)
        embedding_dim = data.get("embedding_dim", 64)

        try:
            process_start = time.time()
            history = service.train_cf_model(
                epochs=epochs,
                batch_size=batch_size,
                embedding_dim=embedding_dim
            )
            process_time = time.time() - process_start
            total_time = time.time() - start_time

            logger.info(f"CF model training completed in {process_time:.3f}s (total: {total_time:.3f}s)")

            return jsonify({
                "message": "CF model training completed",
                "processed_at": datetime.now().isoformat(),
                "training_history": history,
                "performance": {
                    "training_time_seconds": round(process_time, 3),
                    "total_time_seconds": round(total_time, 3)
                }
            })
        except Exception as e:
            logger.error(f"Error training CF model: {e}")
            return jsonify({"error": f"Failed to train CF model: {str(e)}"}), 500

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "collections": {
                "content_products": service.client.content_products_collection.num_entities,
                "hybrid_products": service.client.hybrid_products_collection.num_entities,
                "hybrid_customers": service.client.hybrid_customers_collection.num_entities
            }
        })


def create_performance_response(message: str, count: int, processing_time: float, 
                                  total_time: float) -> Dict:
  """Create standardized performance response"""
  return {
      "message": message,
      "processed_at": datetime.now().isoformat(),
      "performance": {
          "items_count": count,
          "processing_time_seconds": round(processing_time, 3),
          "total_time_seconds": round(total_time, 3),
          "items_per_second": round(count / processing_time, 2) if processing_time > 0 else 0
      }
  }