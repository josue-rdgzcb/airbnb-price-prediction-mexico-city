
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