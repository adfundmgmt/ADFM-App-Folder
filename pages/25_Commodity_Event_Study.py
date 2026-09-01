import streamlit as st

from adfm_core.commodity_event_study_page import TITLE, render_commodity_event_study

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

render_commodity_event_study()
