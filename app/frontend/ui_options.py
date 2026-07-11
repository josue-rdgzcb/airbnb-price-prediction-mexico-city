"""
User interface options for the Streamlit application.

This module centralizes all selectable values used by the
frontend. Keeping them in a single place simplifies future
maintenance and avoids hardcoding options throughout the app.
"""

# ==========================================================
# LOCATION
# ==========================================================

NEIGHBOURHOODS = [
    'Álvaro Obregón',
    'Azcapotzalco',
    'Benito Juárez',
    'Coyoacán',
    'Cuajimalpa de Morelos',
    'Cuauhtémoc',
    'Gustavo A. Madero',
    'Iztacalco',
    'Iztapalapa',
    'La Magdalena Contreras',
    'Miguel Hidalgo',
    'Milpa Alta',
    'Tlalpan',
    'Tláhuac',
    'Venustiano Carranza',
    'Xochimilco'
]


# ==========================================================
# PROPERTY
# ==========================================================

ROOM_TYPES = [
    'Entire home/apt', 
    'Private room', 
    'Hotel room', 
    'Shared room'
]

PROPERTY_TYPES = [
    'Barn',
    'Camper/RV',
    'Casa particular',
    'Castle',
    'Dome',
    'Entire bungalow',
    'Entire cabin',
    'Entire chalet',
    'Entire condo',
    'Entire cottage',
    'Entire guest suite',
    'Entire guesthouse',
    'Entire home',
    'Entire home/apt',
    'Entire hostel',
    'Entire loft',
    'Entire place',
    'Entire rental unit',
    'Entire serviced apartment',
    'Entire townhouse',
    'Entire vacation home',
    'Entire villa',
    'Farm stay',
    'Holiday park',
    'Hut',
    'Private room',
    'Private room in barn',
    'Private room in bed and breakfast',
    'Private room in cabin',
    'Private room in casa particular',
    'Private room in castle',
    'Private room in condo',
    'Private room in cottage',
    'Private room in dome',
    'Private room in earthen home',
    'Private room in farm stay',
    'Private room in guest suite',
    'Private room in guesthouse',
    'Private room in home',
    'Private room in hostel',
    'Private room in hut',
    'Private room in lighthouse',
    'Private room in loft',
    'Private room in minsu',
    'Private room in nature lodge',
    'Private room in rental unit',
    'Private room in resort',
    'Private room in serviced apartment',
    'Private room in shipping container',
    'Private room in tent',
    'Private room in tiny home',
    'Private room in tower',
    'Private room in townhouse',
    'Private room in vacation home',
    'Private room in villa',
    'Private room in yurt',
    'Room in aparthotel',
    'Room in bed and breakfast',
    'Room in boutique hotel',
    'Room in casa particular',
    'Room in hotel',
    'Room in serviced apartment',
    'Shared room in bed and breakfast',
    'Shared room in casa particular',
    'Shared room in condo',
    'Shared room in guesthouse',
    'Shared room in home',
    'Shared room in hostel',
    'Shared room in hotel',
    'Shared room in loft',
    'Shared room in rental unit',
    'Shared room in townhouse',
    'Shipping container',
    'Tent',
    'Tiny home',
    'Tower'
]


# ==========================================================
# BOOLEAN OPTIONS
# ==========================================================

BOOLEAN_OPTIONS = {
    'Yes': 't',
    'No': 'f'
}


# ==========================================================
# HOST VERIFICATIONS
# ==========================================================

HOST_VERIFICATIONS = [
    'email',
    'phone',
    'work_email'
]


# ==========================================================
# AMENITIES
# ==========================================================

AMENITIES = [

    'washer',
    'pool',
    'netflix',
    'hbo',
    'prime video',
    'disney+',
    'apple tv',
    'hulu',
    'wifi',
    'kitchen',
    'hot water',
    'essentials',
    'bed linens',
    'microwave',
    'refrigerator',
    'air conditioning',
    'heating',
    'cooking basics',
    'dishes and silverware',
    'iron',
    'hair dryer',
    'dedicated workspace',
    'dining table',
    'dishwasher',
    'freezer',
    'coffee maker',
    'blender',
    'self check-in',
    'private entrance',
    'elevator',
    'free parking',
    'pets allowed',
    'cleaning available during stay',
    'tv',
    'pool table',
    'piano',
    'game console',
    'ping pong table',
    'patio or balcony',
    'backyard',
    'sauna',
    'city skyline view',
    'outdoor furniture',
    'outdoor dining area',
    'smoke alarm',
    'carbon monoxide alarm',
    'fire extinguisher',
    'exterior security cameras on property',
    'first aid kit'
]


# ==========================================================
# REVIEW SCORES
# ==========================================================

REVIEW_SCORE_MIN = 0.0
REVIEW_SCORE_MAX = 5.0
REVIEW_SCORE_STEP = 0.1