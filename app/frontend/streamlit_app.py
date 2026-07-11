"""
Streamlit frontend for Airbnb Smart Pricing.

This application collects listing information from the user,
sends it to the FastAPI backend and displays the pricing
recommendation together with market insights.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import requests
import streamlit as st
import numpy as np

from app.frontend import ui_labels as LABELS
from app.frontend import ui_options as OPTIONS


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title=LABELS.APP_TITLE,
    page_icon="🏠",
    layout="wide"
)


# ==========================================================
# API CONFIGURATION
# ==========================================================

API_URL = "http://127.0.0.1:8000/predict"


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def predict_listing(payload: dict) -> dict:
    """
    Send listing information to the FastAPI backend.

    Parameters
    ----------
    payload : dict
        Listing information.

    Returns
    -------
    dict
        API response.
    """

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def list_to_string(values: list[str]) -> str:
    """
    Convert a list into the string representation expected
    by the preprocessing pipeline.

    Example
    -------
    ["Wifi", "Kitchen"]

    becomes

    "['Wifi', 'Kitchen']"
    """

    return str(values)


# ==========================================================
# PAGE HEADER
# ==========================================================

st.title(LABELS.APP_TITLE)

st.caption(
    LABELS.APP_SUBTITLE
)

st.divider()

# ==========================================================
# SIDEBAR - LISTING INFORMATION FORM
# ==========================================================

st.sidebar.title("🏠 Listing Information")

st.sidebar.caption(
    "Complete the property information."
)

# ------------------------------------------------------
# LOCATION
# ------------------------------------------------------

st.sidebar.subheader(LABELS.LOCATION_SECTION)

latitude = st.sidebar.number_input(
    LABELS.LATITUDE,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    LABELS.LONGITUDE,
    format="%.6f"
)

neighbourhood = st.sidebar.selectbox(
    LABELS.NEIGHBOURHOOD,
    OPTIONS.NEIGHBOURHOODS
)

st.sidebar.divider()

# ------------------------------------------------------
# PROPERTY
# ------------------------------------------------------

st.sidebar.subheader(LABELS.PROPERTY_SECTION)

room_type = st.sidebar.selectbox(
    LABELS.ROOM_TYPE,
    OPTIONS.ROOM_TYPES
)

property_type = st.sidebar.selectbox(
    LABELS.PROPERTY_TYPE,
    OPTIONS.PROPERTY_TYPES
)

accommodates = st.sidebar.number_input(
    LABELS.ACCOMMODATES,
    min_value=1,
    step=1
)

bedrooms = st.sidebar.number_input(
    LABELS.BEDROOMS,
    min_value=0,
    step=1
)

bathrooms = st.sidebar.number_input(
    LABELS.BATHROOMS,
    min_value=0,
    step=1
)

beds = st.sidebar.number_input(
    LABELS.BEDS,
    min_value=0,
    step=1
)

minimum_nights = st.sidebar.number_input(
    LABELS.MINIMUM_NIGHTS,
    min_value=1,
    step=1
)

st.sidebar.divider()

# ------------------------------------------------------
# HOST
# ------------------------------------------------------

st.sidebar.subheader(LABELS.HOST_SECTION)

host_is_superhost = st.sidebar.selectbox(
    LABELS.HOST_IS_SUPERHOST,
    OPTIONS.BOOLEAN_OPTIONS.keys()
)

instant_bookable = st.sidebar.selectbox(
    LABELS.INSTANT_BOOKABLE,
    OPTIONS.BOOLEAN_OPTIONS.keys()
)

host_verifications = st.sidebar.multiselect(
    LABELS.HOST_VERIFICATIONS,
    OPTIONS.HOST_VERIFICATIONS
)

host_total_listings = st.sidebar.number_input(
    LABELS.HOST_TOTAL_LISTINGS,
    min_value=0,
    step=1
)

host_entire_homes = st.sidebar.number_input(
    LABELS.HOST_ENTIRE_HOMES,
    min_value=0,
    step=1
)

host_private_rooms = st.sidebar.number_input(
    LABELS.HOST_PRIVATE_ROOMS,
    min_value=0,
    step=1
)

st.sidebar.divider()

# ------------------------------------------------------
# AMENITIES
# ------------------------------------------------------

st.sidebar.subheader(LABELS.AMENITIES_SECTION)

amenities = st.sidebar.multiselect(
    LABELS.AMENITIES,
    OPTIONS.AMENITIES
)

st.sidebar.divider()

# ------------------------------------------------------
# REVIEWS
# ------------------------------------------------------

st.sidebar.subheader(LABELS.REVIEWS_SECTION)

has_reviews = st.sidebar.checkbox(
    LABELS.HAS_REVIEWS
)

if has_reviews:

    review_scores_rating = st.sidebar.number_input(
        LABELS.REVIEW_SCORES_RATING,
        min_value=OPTIONS.REVIEW_SCORE_MIN,
        max_value=OPTIONS.REVIEW_SCORE_MAX,
        step=OPTIONS.REVIEW_SCORE_STEP
    )

    review_scores_accuracy = st.sidebar.number_input(
        LABELS.REVIEW_SCORES_ACCURACY,
        min_value=OPTIONS.REVIEW_SCORE_MIN,
        max_value=OPTIONS.REVIEW_SCORE_MAX,
        step=OPTIONS.REVIEW_SCORE_STEP
    )

    review_scores_cleanliness = st.sidebar.number_input(
        LABELS.REVIEW_SCORES_CLEANLINESS,
        min_value=OPTIONS.REVIEW_SCORE_MIN,
        max_value=OPTIONS.REVIEW_SCORE_MAX,
        step=OPTIONS.REVIEW_SCORE_STEP
    )

    review_scores_checkin = st.sidebar.number_input(
        LABELS.REVIEW_SCORES_CHECKIN,
        min_value=OPTIONS.REVIEW_SCORE_MIN,
        max_value=OPTIONS.REVIEW_SCORE_MAX,
        step=OPTIONS.REVIEW_SCORE_STEP
    )

    review_scores_communication = st.sidebar.number_input(
        LABELS.REVIEW_SCORES_COMMUNICATION,
        min_value=OPTIONS.REVIEW_SCORE_MIN,
        max_value=OPTIONS.REVIEW_SCORE_MAX,
        step=OPTIONS.REVIEW_SCORE_STEP
    )

    review_scores_location = st.sidebar.number_input(
        LABELS.REVIEW_SCORES_LOCATION,
        min_value=OPTIONS.REVIEW_SCORE_MIN,
        max_value=OPTIONS.REVIEW_SCORE_MAX,
        step=OPTIONS.REVIEW_SCORE_STEP
    )

    review_scores_value = st.sidebar.number_input(
        LABELS.REVIEW_SCORES_VALUE,
        min_value=OPTIONS.REVIEW_SCORE_MIN,
        max_value=OPTIONS.REVIEW_SCORE_MAX,
        step=OPTIONS.REVIEW_SCORE_STEP
    )

else:

    review_scores_rating = None
    review_scores_accuracy = None
    review_scores_cleanliness = None
    review_scores_checkin = None
    review_scores_communication = None
    review_scores_location = None
    review_scores_value = None

# ==========================================================
# PREDICTION
# ==========================================================

submitted = st.button(
    LABELS.PREDICT_BUTTON,
    use_container_width=True,
    type="primary"
)

if submitted:

    # ------------------------------------------------------
    # BUILD PAYLOAD
    # ------------------------------------------------------

    payload = {

        # LOCATION
        "latitude": latitude,
        "longitude": longitude,
        "neighbourhood_cleansed": neighbourhood,

        # PROPERTY
        "room_type": room_type,
        "property_type": property_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "beds": beds,
        "minimum_nights": minimum_nights,

        # HOST
        "host_is_superhost":
            OPTIONS.BOOLEAN_OPTIONS[
                host_is_superhost
            ],

        "host_verifications":
            list_to_string(
                host_verifications
            ),

        "host_total_listings_count":
            host_total_listings,

        "calculated_host_listings_count_entire_homes":
            host_entire_homes,

        "calculated_host_listings_count_private_rooms":
            host_private_rooms,

        "instant_bookable":
            OPTIONS.BOOLEAN_OPTIONS[
                instant_bookable
            ],

        # REVIEWS
        "review_scores_rating":
            review_scores_rating,

        "review_scores_accuracy":
            review_scores_accuracy,

        "review_scores_cleanliness":
            review_scores_cleanliness,

        "review_scores_checkin":
            review_scores_checkin,

        "review_scores_communication":
            review_scores_communication,

        "review_scores_location":
            review_scores_location,

        "review_scores_value":
            review_scores_value,

        # AMENITIES
        "amenities":
            list_to_string(
                amenities
            )

    }

    # TEMPORAL
    st.subheader("Payload sent to API")
    st.json(payload)

    # ------------------------------------------------------
    # CALL API
    # ------------------------------------------------------

    try:

        with st.status(
            "Analyzing listing...",
            expanded=True
        ) as status:

            st.write("✓ Preparing listing features...")

            st.write("✓ Finding comparable listings...")

            st.write("✓ Estimating market price...")

            st.write("✓ Generating pricing insights...")

            results = predict_listing(
                payload
            )

            status.update(
                label="Analysis completed successfully.",
                state="complete",
                expanded=False
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the API.\n\n"
            "Verify that FastAPI is running."
        )

        st.stop()

    except Exception as e:

        st.error(
            f"Unexpected error:\n\n{e}"
        )

        st.stop()

# ==========================================================
# RESULTS
# ==========================================================

if submitted:

    st.markdown("---")

    st.header(LABELS.RESULTS_SECTION)

    # =====================================================
    # HERO PRICE
    # =====================================================

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:30px;
            border-radius:12px;
            border:1px solid #d9d9d9;
            background-color:#fafafa;
            margin-bottom:25px;
        ">

            <h4 style="margin-bottom:10px;">
                {LABELS.ESTIMATED_PRICE}
            </h4>

            <h1 style="
                font-size:52px;
                color:#00A699;
                margin-top:0px;
                margin-bottom:0px;
            ">
                ${results["estimated_price"]:,.2f} MXN
            </h1>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # MARKET METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            LABELS.TYPICAL_MARKET_PRICE,
            f"${results['typical_market_price']:,.2f}"
        )

    with col2:

        st.metric(
            LABELS.MARKET_PRICE_RANGE,
            (
                f"${results['market_price_lower']:,.0f}"
                " - "
                f"${results['market_price_upper']:,.0f}"
            )
        )

    with col3:

        st.metric(
            LABELS.LISTING_POSITION,
            results["listing_position"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            LABELS.COMPARABLE_LISTINGS,
            results["comparable_listings"]
        )

    with col2:

        st.metric(
            LABELS.CONFIDENCE,
            results["confidence"]
        )

    with col3:

        st.metric(
            LABELS.SEARCH_RADIUS,
            f"{results['search_radius_km']} km"
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    st.subheader(LABELS.SUMMARY)

    summary = fr"""
    The model recommends an estimated nightly price of **\${results['estimated_price']:,.2f} MXN**.

    This recommendation falls **{results['listing_position'].lower()}** within the market range of comparable listings.

    The analysis is based on **{results['comparable_listings']} comparable listings** located within a **{results['search_radius_km']} km** search radius.

    Among those listings, the **typical market price** is **\${results['typical_market_price']:,.2f} MXN**, with most listings ranging between **\${results['market_price_lower']:,.2f} MXN** and **\${results['market_price_upper']:,.2f} MXN**.

    Prediction confidence: **{results['confidence']}**.
    """

    with st.container(border=True):

        st.markdown(summary)

# ==========================================================
# LISTING PREVIEW
# ==========================================================

st.subheader("📋 Listing Preview")

with st.container(border=True):

    # --------------------------------------------------
    # LOCATION & PROPERTY
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 📍 Location")

        st.write(f"**Neighbourhood:** {neighbourhood}")

        st.caption(
            f"{latitude:.6f}, {longitude:.6f}"
        )

    with col2:

        st.markdown("#### 🏠 Property")

        st.write(f"**Room Type:** {room_type}")

        st.write(f"**Property Type:** {property_type}")

    st.divider()

    # --------------------------------------------------
    # PROPERTY DETAILS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Guests", accommodates)

    with col2:
        st.metric("Bedrooms", bedrooms)

    with col3:
        st.metric("Bathrooms", bathrooms)

    with col4:
        st.metric("Beds", beds)

    st.caption(
        f"Minimum stay: **{minimum_nights} night(s)**"
    )

    st.divider()

    # --------------------------------------------------
    # HOST & AMENITIES
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### ⭐ Host")

        if host_is_superhost == "t":
            st.success("⭐ Superhost")
        else:
            st.info("Standard Host")

        if instant_bookable == "t":
            st.success("⚡ Instant Booking")

        st.write(
            f"**Listings Managed:** {host_total_listings}"
        )

        st.write(
            f"Entire Homes: {host_entire_homes}"
        )

        st.write(
            f"Private Rooms: {host_private_rooms}"
        )

        st.write(
            f"Verifications: {len(host_verifications)}"
        )

    with col2:

        st.markdown("#### 🛋 Amenities")

        st.metric(
            "Selected",
            len(amenities)
        )

        if amenities:

            preview = ", ".join(amenities[:6])

            if len(amenities) > 6:

                preview += f" +{len(amenities)-6} more"

            st.caption(preview)

        else:

            st.caption("No amenities selected.")

    st.divider()

    # --------------------------------------------------
    # REVIEWS
    # --------------------------------------------------

    st.markdown("#### ⭐ Guest Reviews")

    if has_reviews:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Overall",
                review_scores_rating
            )

            st.metric(
                "Accuracy",
                review_scores_accuracy
            )

            st.metric(
                "Cleanliness",
                review_scores_cleanliness
            )

        with col2:

            st.metric(
                "Check-in",
                review_scores_checkin
            )

            st.metric(
                "Communication",
                review_scores_communication
            )

        with col3:

            st.metric(
                "Location",
                review_scores_location
            )

            st.metric(
                "Value",
                review_scores_value
            )

    else:

        st.info(
            "This listing has no reviews yet."
        )