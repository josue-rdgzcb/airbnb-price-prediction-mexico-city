
import numpy as np 

# ============== FEATURE TRANSFORMATIONS ==============

# LOG TRANSFORM
LOG_FEATURES = [
    "accommodates",
    "beds",
    "bedrooms",
    "bathrooms",
    "dist_to_nearest_attraction",
]

# WINSORIZATION
WINSOR_FEATURES = {
    "beds": 0.99,
    "host_total_listings_count": 0.95,
}

# BINNING
BINNING_FEATURES = {
    "minimum_nights": {
        "bins": [0, 7, 30, np.inf],
        "labels": ["short_stay", "medium_stay", "long_term"]
    }
}


# ============== FEATURE SCALING ==============

# STANDARD SCALING
STANDARD_SCALE_FEATURES = [
    "amenity_count",
    "amenity_score",
]

# MINMAX SCALING
MINMAX_SCALE_FEATURES = [
    "accommodates",
    "commercial_within_radius",
]

# ROBUST SCALING
ROBUST_SCALE_FEATURES = [
    "beds",
    "host_total_listings_count",
]
