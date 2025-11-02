# Constants
MAX_LENGTH_EMBED = 4048
DESCRIPTION_LENGTH = 10 * 1024

event_weights = {
    # === HIGH INTENT EVENTS (Strong purchase/interest signals) ===
    "purchase": 1.0,                    # Strongest signal - actual conversion
    "repeat_purchase": 1.2,             # Even stronger - proven preference
    "add_to_cart": 0.5,                 # Strong intent but not committed
    # "checkout_initiated": 0.7,          # Started checkout process
    # "add_to_wishlist": 0.6,             # Save for future - strong interest
    "add_to_favorites": 0.6,            # Permanent save - strong signal
    
    # === SOCIAL & VALIDATION EVENTS ===
    "write_review": 0.5,                # Active engagement post-purchase
    "rating_high": 0.4,                 # Rating 4-5 stars
    "rating_medium": 0.1,               # Rating 3 stars - neutral
    "ask_question": 0.25,               # Considering purchase
    
    # === DISCOVERY EVENTS (Lower but important) ===
    "click_from_recommendation": 0.15,  # Responded to algo suggestion
    "click_from_search": 0.2,           # Intent-driven discovery
    "click_from_category": 0.12,        # Browsing discovery
    "view_similar_products": 0.15,      # Exploring options
    
    # === NEGATIVE SIGNALS (Important to prevent bad recommendations) ===
    "remove_from_cart": -0.25,          # Changed mind
    "return_product": -0.4,             # Strong negative
    "rating_low": -0.4,                 # Rating 1-2 stars
    "report_product": -1.0,             # Strongest negative
}