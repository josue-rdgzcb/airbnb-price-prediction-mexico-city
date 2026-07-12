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
    layout="wide",
    initial_sidebar_state="expanded")


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

st.markdown(
    f"""
    <div style="margin-bottom: 25px;">
        <h1 style="
            font-size: 38px; 
            font-weight: 800; 
            color: #222222; 
            margin: 0 0 4px 0;
        ">
            {LABELS.APP_TITLE}
        </h1>
        <p style="
            font-size: 16px; 
            color: #717171; 
            margin: 0;
        ">
            {LABELS.APP_SUBTITLE}
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

welcome_message = st.empty()

welcome_message.info(
    "👈 **Welcome! Please fill in your listing details in the sidebar on the left**, "
    "then click the **'Estimate Price'** button to generate your market analysis.",
    icon="ℹ️"
)

# ==========================================================
# SIDEBAR - LISTING INFORMATION FORM
# ==========================================================

st.sidebar.title("🏠 Listing Information")

st.sidebar.caption(
    "Complete the property information.  \n"  # 👈 Fíjate en los dos espacios antes de \n
    "Provide accurate details to ensure a more precise price estimation."
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
    min_value=0.0,
    step=0.5
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

superhost_selection = st.sidebar.selectbox(
    LABELS.HOST_IS_SUPERHOST,
    list(OPTIONS.BOOLEAN_OPTIONS.keys())
)

host_is_superhost = OPTIONS.BOOLEAN_OPTIONS[superhost_selection]

instant_selection = st.sidebar.selectbox(
    LABELS.INSTANT_BOOKABLE,
    list(OPTIONS.BOOLEAN_OPTIONS.keys())
)
instant_bookable = OPTIONS.BOOLEAN_OPTIONS[instant_selection]

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

st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        height: 68px !important;
        background-color: #FF385C !important; 
    }
    div.stButton > button[kind="primary"] p {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: white !important;
    }
    
    /* Hover */
    div.stButton > button[kind="primary"]:hover {
        background-color: white !important;
    }
    /* Hover */
    div.stButton > button[kind="primary"]:hover p {
        color: #FF385C !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)


btn_col1, btn_col2, btn_col3 = st.columns([0.5, 1, 0.5])

with btn_col2:
    submitted = st.button(
        LABELS.PREDICT_BUTTON,
        use_container_width=True,
        type="primary"
    )


if submitted:

    # ------------------------------------------------------
    # BUILD PAYLOAD
    # ------------------------------------------------------

    welcome_message.empty()

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
        "host_is_superhost": host_is_superhost,
        "host_verifications": list_to_string(host_verifications),
        "host_total_listings_count": host_total_listings,
        "calculated_host_listings_count_entire_homes": host_entire_homes,
        "calculated_host_listings_count_private_rooms": host_private_rooms,
        "instant_bookable": instant_bookable,

        # REVIEWS
        "review_scores_rating": review_scores_rating,
        "review_scores_accuracy": review_scores_accuracy,
        "review_scores_cleanliness": review_scores_cleanliness,
        "review_scores_checkin": review_scores_checkin,
        "review_scores_communication": review_scores_communication,
        "review_scores_location": review_scores_location,
        "review_scores_value": review_scores_value,

        # AMENITIES
        "amenities": list_to_string(amenities)

    }

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

    st.subheader(LABELS.RESULTS_SECTION)

    # =====================================================
    # HERO PRICE
    # =====================================================

    st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 16px 20px;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        margin-bottom: 25px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <h4 style="
            margin: 0 0 2px 0;
            color: #484848;
            font-weight: 500;
            font-size: 16px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            line-height: 1.2;
        ">
            {LABELS.ESTIMATED_PRICE}
        </h4>
        <h1 style="
            font-size: 48px;
            font-weight: 700;
            color: #FF385C;
            margin: 0;
            line-height: 1.0;
        ">
            ${results["estimated_price"]:,.2f} <span style="font-size: 22px; font-weight: 500; color: #484848;">MXN</span>
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

    # =====================================================
    # MARKET METRICS
    # =====================================================

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #ffffff;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #f0f0f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            text-align: center;
            height: 105px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-label {
            font-size: 16px;
            color: #717171;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 700;
            color: #222222;
            margin: 0;
        }
        .metric-value-highlight {
            color: #FF385C;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.subheader(LABELS.INSIGHTS_SECTION)

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{LABELS.TYPICAL_MARKET_PRICE}</div>
                <div class="metric-value">${results['typical_market_price']:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{LABELS.MARKET_PRICE_RANGE}</div>
                <div class="metric-value" style="font-size: 26px;">
                    ${results['market_price_lower']:,.0f} - ${results['market_price_upper']:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{LABELS.LISTING_POSITION}</div>
                <div class="metric-value metric-value-highlight">{results["listing_position"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    d_col1, d_col2, d_col3 = st.columns(3)

    with d_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{LABELS.COMPARABLE_LISTINGS}</div>
                <div class="metric-value">{results["comparable_listings"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{LABELS.CONFIDENCE}</div>
                <div class="metric-value">{results["confidence"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{LABELS.SEARCH_RADIUS}</div>
                <div class="metric-value">{results['search_radius_km']} km</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    st.html("<div style='margin-block: 15px;'></div>")

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

st.html("<div style='margin-block: 15px;'></div>")

st.subheader(LABELS.PREVIEW_SECTION)

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

    st.markdown("#### 🛏️ Capacity & Space")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("**Guests**", accommodates)

    with col2:
        st.metric("**Bedrooms**", bedrooms)

    with col3:
        st.metric("**Bathrooms**", bathrooms)

    with col4:
        st.metric("**Beds**", beds)

    st.divider()

    # --------------------------------------------------
    # HOST
    # --------------------------------------------------

    st.markdown("#### 👤 Host")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        if host_is_superhost == "t":
            st.write("**Host Status:** ⭐ Superhost")
        else:
            st.write("**Host Status:** ☑️ Standard Host")

    with status_col2:
        if instant_bookable == "t":
            st.write("**Booking:** ⚡ Instant Booking")
        else:
            st.write("**Booking:** 📅 Request to Book")

    st.html("<div style='margin-block: 10px;'></div>")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("**Listings Managed**", host_total_listings)

    with col2:
        st.metric("**Entire Homes**", host_entire_homes)

    with col3:
        st.metric("**Private Rooms**", host_private_rooms)

    with col4:
        st.metric("**Verifications**", len(host_verifications))

        if host_verifications:
            preview = ", ".join(host_verifications)
            st.caption(preview)

        else:
            st.caption("No verifications selected.")

    st.divider()

    # --------------------------------------------------
    # AMENITIES & MINIMUM STAY
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 🌃 Minimum Stay")

        st.write(f"**Minimum nights:** {minimum_nights} night(s)")

    with col2:

        st.markdown("#### 📺 Amenities")

        st.metric("Selected", len(amenities))

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
                "Overall Rating",
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

# --------------------------------------------------
# Global Footer Signature (Sidebar Bottom)
# --------------------------------------------------

st.html("<div style='margin-block: 15px;'></div>")

st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 18px; font-weight: 500;">
        Developed by Josué Rodríguez
        <div style="margin-top: 6px;">
            <a href="https://github.com/josue-rdgzcb" target="_blank" style="color: #FF385C; font-weight: 700; text-decoration: underline;">
                🔗 GitHub
            </a>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)