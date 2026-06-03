
MODEL_FEATURES = [

    # Numeric features
    'calculated_host_listings_count_entire_homes_log',
    'calculated_host_listings_count_private_rooms_log',
    'dist_to_nearest_attraction_log', 'beds_log', 'amenity_score',
    'accommodates_log', 'bedrooms_log', 'bathrooms_log',
    'attractions_within_radius_log', 'commercial_within_radius',

    # Categorical features
    'room_type_entire_home/apt','room_type_hotel_room', 'room_type_private_room','room_type_shared_room',
    'neighbourhood_cleansed_freq', 'property_group_room_freq',
    'host_total_listings_segment_ordinal', 'review_scores_mean_segment_ordinal', 
    'minimum_nights_segment_ordinal', 'host_verifications_grouped_ordinal',

    # Boolean features
    'host_is_superhost','instant_bookable', 
    'has_tv', 'has_elevator', 'has_free_parking', 'has_coffee_maker', 
    'has_outdoor_furniture', 'has_air_conditioning','has_self_check_in', 'has_pool',
    
]