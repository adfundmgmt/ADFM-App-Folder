import streamlit as st
import pydeck as pdk
import pandas as pd

# App Title
st.set_page_config(page_title="NVIDIA Global Footprint Map", layout="wide")
st.title("NVIDIA (NVDA) Global AI/Data Center Footprint (Sample Visualization)")

# Mock data — Replace with real locations/metrics if available
data = {
    "lat": [37.7749, 40.7128, 51.5074, -23.5505, 48.8566, 1.3521, 35.6895],
    "lon": [-122.4194, -74.0060, -0.1278, -46.6333, 2.3522, 103.8198, 139.6917],
    "location": [
        "San Francisco", "New York", "London",
        "São Paulo", "Paris", "Singapore", "Tokyo"
    ],
    "gpu_capacity": [300, 500, 250, 150, 200, 180, 320]  # Example "capacity" metric
}
df = pd.DataFrame(data)

# Scatterplot layer for data center locations
bubble_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_radius="gpu_capacity * 1000",
    get_fill_color='[140, 30, 230, 180]',
    pickable=True,
    auto_highlight=True,
)

# (Optional) Example cable/traffic lines
# Lines from NYC to other locations for effect
arc_data = pd.DataFrame({
    'from_lon': [-74.0060]*3,
    'from_lat': [40.7128]*3,
    'to_lon': [-122.4194, -0.1278, 103.8198],
    'to_lat': [37.7749, 51.5074, 1.3521]
})
arc_layer = pdk.Layer(
    "ArcLayer",
    data=arc_data,
    get_source_position='[from_lon, from_lat]',
    get_target_position='[to_lon, to_lat]',
    get_width=3,
    get_tilt=15,
    get_source_color=[255, 140, 0, 180],
    get_target_color=[0, 140, 255, 180],
    pickable=False
)

# View settings (centered, zoomed out)
view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.3, pitch=30)

# Render pydeck map
st.pydeck_chart(
    pdk.Deck(
        layers=[bubble_layer, arc_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={"text": "{location}\nGPU Capacity: {gpu_capacity}"}
    )
)

# Optional: instructions
st.caption("""
This is a sample visualization of NVIDIA's global AI/data center presence.
Bubble size = capacity (mock numbers).
Lines = sample network/cable flows.
Swap in real data for a more meaningful map.
""")
