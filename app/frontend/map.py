"""
Interactive location selector.

Allows the user to choose a listing location on a Folium map.
"""

import folium
import streamlit as st

from streamlit_folium import st_folium

from app.services.geolocation import get_neighbourhood


# ==========================================================
# MAP CONFIGURATION
# ==========================================================

DEFAULT_LATITUDE = 19.4326
DEFAULT_LONGITUDE = -99.1332
DEFAULT_ZOOM = 11


# ==========================================================
# MAP COMPONENT
# ==========================================================

def render_location_map():
    """
    Render an interactive map.

    Returns
    -------
    dict | None
    """

    # ------------------------------------------------------
    # Cursor style
    # ------------------------------------------------------

    st.markdown(
        """
        <style>
        .leaflet-container{
            cursor:crosshair !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # Session State
    # ------------------------------------------------------

    if "selected_location" not in st.session_state:

        st.session_state.selected_location = None

    # ------------------------------------------------------
    # Create map
    # ------------------------------------------------------

    m = folium.Map(

        location=[
            DEFAULT_LATITUDE,
            DEFAULT_LONGITUDE
        ],

        zoom_start=DEFAULT_ZOOM,

        control_scale=True

    )

    # ------------------------------------------------------
    # Existing marker
    # ------------------------------------------------------

    if st.session_state.selected_location is not None:

        folium.Marker(

            [

                st.session_state.selected_location["latitude"],

                st.session_state.selected_location["longitude"]

            ],

            tooltip="Selected Listing",

            icon=folium.Icon(
                color="red",
                icon="home",
                prefix="fa"
            )

        ).add_to(m)

    # ------------------------------------------------------
    # Render map
    # ------------------------------------------------------

    map_data = st_folium(

        m,

        width=700,

        height=500,

        key="listing_map",

        returned_objects=[
            "last_clicked"
        ]

    )

    # ------------------------------------------------------
    # New click
    # ------------------------------------------------------

    if (

        map_data

        and

        map_data["last_clicked"] is not None

    ):

        latitude = map_data["last_clicked"]["lat"]

        longitude = map_data["last_clicked"]["lng"]

        neighbourhood = get_neighbourhood(

            latitude=latitude,

            longitude=longitude

        )

        st.session_state.selected_location = {

            "latitude": latitude,

            "longitude": longitude,

            "neighbourhood": neighbourhood

        }

        st.rerun()

    # ------------------------------------------------------
    # Location summary
    # ------------------------------------------------------

    if st.session_state.selected_location is None:

        st.info(
            "Click on the map to choose the property location."
        )

        return None

    location = st.session_state.selected_location

    st.success("Location selected")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Latitude",

            f"{location['latitude']:.6f}"

        )

    with col2:

        st.metric(

            "Longitude",

            f"{location['longitude']:.6f}"

        )

    st.metric(

        "Neighbourhood",

        location["neighbourhood"]

    )

    st.divider()

    if st.button(

        "✅ Confirm Location",

        use_container_width=True

    ):

        return location

    return None