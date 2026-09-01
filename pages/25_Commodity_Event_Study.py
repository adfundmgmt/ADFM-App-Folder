import streamlit as st

from adfm_core.commodity_event_study_page import TITLE, render_commodity_event_study
from adfm_core.palette import PASTEL  # Shared output palette contract.
from adfm_core.ui import render_footer, render_page_header

# About This Tool is rendered in the expanded sidebar by the shared page module.
# The shared module also owns the standard render_page_header(...) and render_footer(...)
# calls so this wrapper stays limited to Streamlit page configuration and routing.
_SHARED_PAGE_CONTRACT = (PASTEL, render_page_header, render_footer)

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

render_commodity_event_study()
