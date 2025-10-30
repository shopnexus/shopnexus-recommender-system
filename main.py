import time
import logging
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify

from service import ResponseUtils, service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH_EMBED = 4048
DESCRIPTION_LENGTH = 10 * 1024

# Flask app
app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    """Product search endpoint"""
    start = time.time()
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    limit = data.get("limit", 10)
    weights = data.get("weights", {"dense": 1.0, "sparse": 1.0})
    offset = data.get("offset", 0)

    query_embeddings = service.generate_embeddings([query])
    logger.info(f"Embedding time: {(time.time() - start) * 1000:.2f} ms")

    result = service.hybrid_search(
        query_embeddings["dense"][0],
        query_embeddings["sparse"][[0]],
        dense_weight=weights.get("dense", 1.0),
        sparse_weight=weights.get("sparse", 1.0),
        offset=offset,
        limit=limit,
    )

    return jsonify([{"id": hit["id"], "score": hit.score} for hit in result])


@app.route("/analytics/process", methods=["POST"])
def process_analytics():
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

    return jsonify(ResponseUtils.create_performance_response(
        f"Processed {len(events)} events", len(events), process_time, total_time
    ))

@app.route("/user/<int:account_id>/recommendations", methods=["GET"])
def get_recommendations(account_id):
    """Get product recommendations for a user based on their preferences"""
    limit = request.args.get("limit", 10, type=int)

    try:
        user_vector = service.get_user_vector(account_id)
        if user_vector is None:
            return jsonify([])

        results = service.dense_search(user_vector.tolist(), limit=limit)
        return jsonify([{"id": hit["id"], "score": float(hit.score)} for hit in results])

    except Exception as e:
        logger.error(f"Error getting recommendations for user {account_id}: {e}")
        return jsonify({"error": "Failed to get recommendations"}), 500


@app.route("/products", methods=["POST"])
def update_products():
    """Update products in Milvus collection"""
    start_time = time.time()
    data = request.json
    products = data.get("products", [])
    metadata_only = data.get("metadata_only", False)

    if not products:
        return jsonify({"error": "No products provided"}), 200

    try:
        process_start = time.time()
        processed_count = service.update_products_batch(products, metadata_only=metadata_only)
        process_time = time.time() - process_start
        total_time = time.time() - start_time

        logger.info(f"Products update completed: {processed_count} products processed in {process_time:.3f}s (total: {total_time:.3f}s)")

        return jsonify(ResponseUtils.create_performance_response(
            f"Successfully updated {processed_count} products", 
            processed_count, process_time, total_time
        ))

    except Exception as e:
        logger.error(f"Error updating products: {e}")
        return jsonify({"error": "Failed to update products"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "collections": {
            "products": service.products_collection.num_entities,
            "customer": service.customer_collection.num_entities
        }
    })

def main():
    """Main function - run as Flask API server"""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()