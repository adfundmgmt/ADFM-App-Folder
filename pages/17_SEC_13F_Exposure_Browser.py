"""SEC Form 13F institutional exposure and manager browser.

The shared renderer owns render_page_header( and the visual implementation;
it imports adfm_core.palette so this thin page remains on the common ADFM
palette and masthead contract.
"""

import streamlit as st

from adfm_core.sec_13f_browser import TITLE, render_browser
from adfm_core.ui import render_footer

# About This Tool is rendered in the shared browser sidebar.
st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")
render_browser()
render_footer(
    data_note="Primary inputs: official SEC Form 13F filings and bulk data sets; SEC company ticker directory."
)
