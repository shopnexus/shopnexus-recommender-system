import logging
from dateutil import parser as dateutil_parser
from datetime import datetime, timezone
from config import DECAY_ENABLED, DECAY_HALF_LIFE_DAYS, DECAY_MIN_FACTOR, event_weights
from typing import Any, Dict, List
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def decay_factor(ts_str: str) -> float:
    """Calculate decay factor based on timestamp"""
    decay_factor = 1.0
    now = datetime.now(timezone.utc)
    try:
        if not ts_str or not isinstance(ts_str, str):
            raise ValueError("timestamp is empty or not a string")

        # dateutil handles Z, offsets, and fractional seconds
        ts = dateutil_parser.isoparse(ts_str)
        # Assume UTC if no timezone provided
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        if DECAY_ENABLED and DECAY_HALF_LIFE_DAYS > 0:
            decay_factor = 0.5 ** (age_days / float(DECAY_HALF_LIFE_DAYS))
            decay_factor = max(decay_factor, float(DECAY_MIN_FACTOR))
    except Exception as e:
        logger.error(f"Failed to calculate decay factor for timestamp {ts_str}: {e}")
        pass
    return decay_factor


def get_event_weight(event_type: str, ts_str: str) -> float:
    return event_weights.get(event_type, 0.0) * decay_factor(ts_str)


def avg_vec_by_event(
    item_vectors: Dict[int, np.ndarray], events: List[Dict], fallback_vec: np.ndarray
) -> np.ndarray:
    # Aggregate weights per product with event weights and time decay
    weighted_item_scores: Dict[int, float] = defaultdict(float)

    for e in events:
        item_id = e.get("ref_id")
        if item_id is None:
            logger.warning(f"Skipping event {e}, no item_id found")
            continue
        ev_type = (e.get("event_type") or "").lower()
        weight = get_event_weight(ev_type, e.get("date_created"))
        weighted_item_scores[item_id] += weight

    if not weighted_item_scores:
        return fallback_vec

    # Compute weighted average with signed weights and L2 normalize
    sum_weight_abs = 0.0
    accum = np.zeros_like(fallback_vec, dtype=np.float32)
    for item_id, w in weighted_item_scores.items():
        vec = item_vectors.get(item_id)
        if vec is None or len(vec) == 0:
            logger.warning(f"Skipping item {item_id}, vector is None or empty")
            continue
        accum += w * vec
        sum_weight_abs += abs(w)

    if sum_weight_abs == 0.0:
        logger.warning("Sum of weight absolute is 0")
        return fallback_vec

    user_cf_vec = accum / sum_weight_abs
    return user_cf_vec


# Average vectors by weight (score) [{'id': int, 'vector': np.ndarray, 'score': float}]
def avg_vec_by_weight(
    vectors: List[Dict[str, Any]], fallback_vec: np.ndarray
) -> np.ndarray:
    if len(vectors) == 0:
        return fallback_vec

    accum = np.zeros_like(fallback_vec, dtype=np.float32)
    sum_weight = 0.0
    for v in vectors:
        score = v.get("score", 0.0)
        vector = v.get("vector")
        if vector is None:
            continue
        accum += score * np.asarray(vector)
        sum_weight += abs(score)  # Use absolute value to handle negative weights
    
    if sum_weight == 0.0:
        logger.warning("Sum of weights is 0 in avg_vec_by_weight, returning fallback")
        return fallback_vec
    
    return accum / sum_weight
