"""
Interactive location selector.

Allows the user to choose a listing location on a Folium map.
"""

import folium
import streamlit as st
from streamlit_folium import st_folium
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    # Session State Initialization
    # ------------------------------------------------------  
    if "selected_location" not in st.session_state:  
        st.session_state.selected_location = None  
        
    if "map_error" not in st.session_state:
        st.session_state.map_error = None

    # ------------------------------------------------------  
    # Adjusted CSS (We add classes for our custom alert)  
    # ------------------------------------------------------  
    st.markdown(  
        """  
        <style>  
        /* Reduce metric sizes */  
        [data-testid="stMetricLabel"] p {  
            font-size: 0.85rem !important;  
        }  
        [data-testid="stMetricValue"] div {  
            font-size: 1.05rem !important;  
        }  
          
        /* CUSTOM GREEN BOX (st.success replacement) */  
        .custom-success-box {  
            background-color: rgba(40, 167, 69, 0.12); /* Soft green background */  
            border: 1px solid rgba(40, 167, 69, 0.3);   /* Subtle green border */  
            color: #1e4620;                            /* Readable dark green text */  
            padding: 6px 10px;                         /* Ultra compact padding */  
            border-radius: 0.5rem;                     /* Rounded corners matching Streamlit */  
            font-size: 0.82rem;                        /* Optimized font size */  
            font-weight: 500;                          /* Semi-bold text */  
            margin-bottom: 1rem;                       /* Separation with the element below */  
            display: flex;  
            align-items: center;  
            gap: 6px;                                  /* Space between emoji and text */  
        }  
        </style>  
        """,  
        unsafe_allow_html=True  
    )  

    # ------------------------------------------------------  
    # 1. READ INTERACTION FIRST (Get data from the key state)
    # ------------------------------------------------------  
    # st_folium stores the output in session_state under its key name ("listing_map")
    map_state = st.session_state.get("listing_map")
    
    if map_state and map_state.get("last_clicked") is not None:  
        latitude = map_state["last_clicked"]["lat"]  
        longitude = map_state["last_clicked"]["lng"]  
        neighbourhood = get_neighbourhood(latitude=latitude, longitude=longitude)  
          
        if neighbourhood is None:  
            st.session_state.map_error = "Please select a location inside Mexico City."  
            st.session_state.selected_location = None  
        else:  
            st.session_state.map_error = None  
            st.session_state.selected_location = {  
                "latitude": latitude,  
                "longitude": longitude,  
                "neighbourhood": neighbourhood  
            }  

    # ------------------------------------------------------  
    # 2. CREATE MAP WITH UPDATED MARKER  
    # ------------------------------------------------------  
    m = folium.Map(  
        location=[DEFAULT_LATITUDE, DEFAULT_LONGITUDE],  
        zoom_start=DEFAULT_ZOOM,  
        control_scale=True  
    )  

    # Add marker using the newly updated session_state from step 1
    if st.session_state.selected_location is not None:  
        folium.Marker(  
            [  
                st.session_state.selected_location["latitude"],  
                st.session_state.selected_location["longitude"]  
            ],  
            tooltip="Selected Listing",  
            icon=folium.Icon(color="red", icon="home", prefix="fa")  
        ).add_to(m)  

    # ------------------------------------------------------  
    # Layout (Give more proportion to the map: 3 parts to 1)  
    # ------------------------------------------------------  
    col_map, col_info = st.columns([3, 1])   

    # ------------------------------------------------------  
    # 3. RENDER MAP  
    # ------------------------------------------------------  
    with col_map:  
        st_folium(  
            m,   
            width=None,  
            height=500,   
            key="listing_map",   
            returned_objects=["last_clicked"]  
        )  

    # ------------------------------------------------------  
    # Render info (Right section)  
    # ------------------------------------------------------  
    with col_info:  
        if st.session_state.get("map_error"):  
            st.warning(st.session_state.map_error)  
          
        elif st.session_state.get("selected_location") is None:  
            st.info("Click on the map to choose the property location.")  
          
        else:  
            location = st.session_state.selected_location  
              
            st.markdown(  
                '<div class="custom-success-box"><span>📍</span> Selected Location</div>',   
                unsafe_allow_html=True  
            )  
              
            st.metric("Latitude", f"{location['latitude']:.6f}")  
            st.metric("Longitude", f"{location['longitude']:.6f}")  
            st.metric("Neighbourhood", location["neighbourhood"])  
            st.divider()  
              
            if st.button("✅ Confirm Location", use_container_width=True):  
                
                return location

