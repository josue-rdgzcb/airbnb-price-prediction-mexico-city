
# ============== FEATURE CLEANING ==============

PERCENTAGE_COLUMNS = [
]

BINARY_COLUMNS = [
    "host_is_superhost",
    "instant_bookable"
]

PARSE_COLUMNS = [
    "host_verifications",
    "amenities"

]

STRING_COLUMNS = [
    "room_type"
]


# ============== FEATURE IMPUTATION ==============

# MEDIAN IMPUTATION
MEDIAN_IMPUTE_FEATURES = [
    "bathrooms",
    "bedrooms",
    "beds",
    "host_total_listings_count"
]

# MEAN IMPUTATION
MEAN_IMPUTE_FEATURES = [
]

# MODE IMPUTATION
MOST_FREQUENT_IMPUTE_FEATURES = [
    "host_total_listings_segment",
    "host_is_superhost"
]

# MISSING CATEGORY IMPUTATION
MISSING_CATEGORY_IMPUTE_FEATURES = [
    "review_scores_mean_segment"
]

# ============== NUMERIC FEATURE TRANSFORMATIONS ==============

# LOG TRANSFORM
LOG_FEATURES = [
    "accommodates",
    "bathrooms",
    "beds",
    "bedrooms",
    "dist_to_nearest_attraction",
    "attractions_within_radius",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms"
]

# BINNING
BINNING_FEATURES = {

#    "review_scores_mean": {
#        "method": "qcut",
#        "bins": 4
#    }

}

# ============== FEATURE ENCODING ==============


# ONE HOT ENCODING
ONE_HOT_ENCODING_FEATURES = [
    "room_type",
    "property_group"
]


# FREQUENCY ENCODING
FREQUENCY_ENCODING_FEATURES = [
    "neighbourhood_cleansed",
    "property_group_room"
]


# ORDINAL ENCODING

ORDINAL_ENCODING_FEATURES = {

    "minimum_nights_segment": {
        "short_stay": 0,
        "medium_stay": 1,
        "long_term": 2
    },

    "host_verifications_grouped": {
        "none": 0,
        "low": 1,
        "basic": 2,
        "extended": 3
    },

    "review_scores_mean_segment": {
        "missing": -1,
        "low_review": 0,
        "medium_review": 1,
        "high_review": 2
    },

    "host_total_listings_segment": {
        "small_host": 0,
        "medium_host": 1,
        "large_host": 2,
        "professional_host": 3
    }
}


# ============== FEATURE SCALING ==============

# STANDARD SCALING
STANDARD_SCALE_FEATURES = [
    "accommodates_log",
    "bedrooms_log",
    "bathrooms_log",
    "amenity_count",
    "attractions_within_radius_log",
    "commercial_within_radius",
    "neighbourhood_cleansed_freq",
    "property_group_room_freq"
]

# MINMAX SCALING
MINMAX_SCALE_FEATURES = [
    "amenity_score"
    ]

# ROBUST SCALING
ROBUST_SCALE_FEATURES = [
    "calculated_host_listings_count_entire_homes_log",
    "calculated_host_listings_count_private_rooms_log",
    "dist_to_nearest_attraction_log",
    "beds_log"
    ]
