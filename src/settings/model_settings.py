# SELECTED MODEL FEATURES FOR TRAINING AND EVALUATION
MODEL_FEATURES = [

    # Numeric features
    'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms',
    'dist_to_nearest_attraction', 'beds', 'amenity_score',
    'accommodates', 'bedrooms', 'bathrooms',
    'attractions_within_radius', 'commercial_within_radius',

    # Categorical features
    'room_type','neighbourhood_cleansed', 'property_group_room',
    'host_total_listings_segment', 'review_scores_mean_segment', 
    'minimum_nights_segment', 'host_verifications_grouped',

    # Boolean features
    'host_is_superhost','instant_bookable', 
    'has_tv', 'has_elevator', 'has_free_parking', 'has_coffee_maker', 
    'has_outdoor_furniture', 'has_air_conditioning','has_self_check_in', 'has_pool'
    
]

# RANDOM FOREST SEARCH SPACE
RF_PARAM_GRID = {
    # Number of trees in the forest
    "n_estimators": [100, 200, 300, 500, 700],

    # Maximum depth of each tree
    "max_depth": [None, 10, 20, 30, 40],

    # Minimum samples required to split a node
    "min_samples_split": [2, 5, 10, 20],

    # Minimum samples required at each leaf node
    "min_samples_leaf": [1, 2, 4, 8],

    # Number of features considered at each split
    "max_features": ["sqrt", "log2", None]
}

# XGBOOST SEARCH SPACE
XGB_PARAM_GRID = {

    # Number of boosting rounds
    "n_estimators": [100, 200, 300, 500],

    # Learning rate
    "learning_rate": [0.01, 0.03, 0.05, 0.1],

    # Maximum tree depth
    "max_depth": [3, 4, 5, 6, 8],

    # Minimum child weight
    "min_child_weight": [1, 3, 5, 7],

    # Minimum loss reduction required for a split
    "gamma": [0, 0.1, 0.3, 0.5],

    # Row subsampling
    "subsample": [0.6, 0.8, 1.0],

    # Column subsampling
    "colsample_bytree": [0.6, 0.8, 1.0],

    # L2 regularization
    "reg_lambda": [0, 1, 5, 10],

    # L1 regularization
    "reg_alpha": [0, 0.1, 1]

}