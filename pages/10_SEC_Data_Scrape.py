import streamlit as st
import pydeck as pdk
import pandas as pd
import requests

st.set_page_config(page_title="Live Global Vessel Map", layout="wide")
st.title("Live Global Shipping Vessel Map (OpenShipMap API)")

# Fetch live vessel positions from OpenShipMap API
@st.cache_data(ttl=300)  # cache for 5 minutes to avoid excess hits
def get_ship_positions():
    url = "https://openshipmap.org/api/positions"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        # Flatten into DataFrame
        ships = []
        for ship in data["positions"]:
            ships.append({
                "lat": ship["latitude"],
                "lon": ship["longitude"],
                "name": ship.get("name", "Unknown"),
                "type": ship.get("type", "Unknown"),
                "heading": ship.get("heading", ""),
                "speed": ship.get("speed", 0)
            })
        return pd.DataFrame(ships)
    except Exception as e:
        st.error(f"Failed to fetch ship data: {e}")
        return pd.DataFrame()

df_ships = get_ship_positions()
if df_ships.empty:
    st.warning("No live vessel data available. Try again later.")
else:
    st.markdown(f"Showing **{len(df_ships)}** live vessel positions.")

    # Pydeck scatterplot layer for ship locations
    ship_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_ships,
        get_position='[lon, lat]',
        get_radius=6000,
        get_fill_color='[30, 144, 255, 180]',
        pickable=True,
        auto_highlight=True,
    )

    # World view settings
    view_state = pdk.ViewState(
        latitude=0,
        longitude=0,
        zoom=1,
        pitch=0,
        bearing=0
    )

    # Tooltip
    tooltip = {
        "html": "<b>{name}</b><br>Type: {type}<br>Speed: {speed} kn<br>Heading: {heading}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[ship_layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v10',
            tooltip=tooltip
        )
    )

st.caption("""
**Data:** [OpenShipMap.org](https://openshipmap.org/api/) (updated every few minutes)<br>
Bubble = live vessel. Hover for info.<br>
To refresh: rerun app or wait for cache to expire.
""", unsafe_allow_html=True)
