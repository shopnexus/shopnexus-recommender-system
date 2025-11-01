# Constants
MAX_LENGTH_EMBED = 4048
DESCRIPTION_LENGTH = 10 * 1024

# Event weights for user vector calculation
event_weights = {
    # === HIGH INTENT EVENTS (Strong purchase/interest signals) ===
    "purchase": 1.0,                    # Strongest signal - actual conversion
    "repeat_purchase": 1.2,             # Even stronger - proven preference
    "add_to_cart": 0.5,                 # Strong intent but not committed
    # "checkout_initiated": 0.7,          # Started checkout process
    # "add_to_wishlist": 0.6,             # Save for future - strong interest
    "add_to_favorites": 0.6,            # Permanent save - strong signal
    
    # === ENGAGEMENT EVENTS (Moderate interest) ===
    "product_detail_view": 0.25,        # Clicked into full product page
    "time_spent_30s_plus": 0.2,         # Spent significant time viewing
    "time_spent_60s_plus": 0.3,         # Deep engagement
    "image_zoom": 0.15,                 # Examining product closely
    "view_multiple_images": 0.18,       # Looking at all angles
    "video_watch": 0.25,                # High engagement content
    "video_watch_complete": 0.35,       # Watched entire product video
    "view_specifications": 0.2,         # Technical interest
    # "view_size_guide": 0.15,            # Considering purchase
    
    # === SOCIAL & VALIDATION EVENTS ===
    "write_review": 0.5,                # Active engagement post-purchase
    "rating_high": 0.4,                 # Rating 4-5 stars
    "rating_medium": 0.1,               # Rating 3 stars - neutral
    "share_product": 0.35,              # Social endorsement
    "ask_question": 0.25,               # Considering purchase
    "helpful_review_vote": 0.08,        # Engaged with product content
    
    # === DISCOVERY EVENTS (Lower but important) ===
    "click_from_recommendation": 0.15,  # Responded to algo suggestion
    "click_from_search": 0.2,           # Intent-driven discovery
    "click_from_category": 0.12,        # Browsing discovery
    "view_similar_products": 0.15,      # Exploring options
    # "add_to_compare": 0.22,             # Serious consideration
    "quick_view": 0.08,                 # Light interest
    "category_browse": 0.05,            # General browsing
    
    # === NEGATIVE SIGNALS (Important to prevent bad recommendations) ===
    "remove_from_cart": -0.25,          # Changed mind
    # "remove_from_wishlist": -0.3,       # Lost interest
    "abandoned_cart": -0.15,            # Didn't complete purchase
    "return_product": -0.6,             # Strong negative
    "rating_low": -0.5,                 # Rating 1-2 stars
    "report_product": -1.0,             # Strongest negative
    "skip_recommendation": -0.05,       # Soft negative
    "quick_bounce": -0.08,              # Viewed < 3 seconds
    
    # === POST-PURCHASE BEHAVIOR ===
    "reorder": 1.1,                     # Repeat purchase of exact item
    # "subscription_signup": 1.3,         # Highest commitment
    # "bundle_purchase": 0.8,             # Bought with related items
}

