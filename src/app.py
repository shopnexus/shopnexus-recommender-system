import logging
from flask import Flask
from service import Service
from routes import register_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize service
service = Service("localhost")

# Register routes
register_routes(app, service)


def main():
    """Main function - run as Flask API server"""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()

