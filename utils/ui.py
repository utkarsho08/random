import streamlit as st
from utils.theme import CSS_STYLES

def inject_custom_css():
    """Injects custom CSS defined in the theme module."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

def render_kpi_card(title, value):
    """Renders a styled KPI card."""
    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def init_session_state():
    """Initializes common variables in the Streamlit session state."""
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    if 'current_input' not in st.session_state:
        st.session_state['current_input'] = None
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None

def render_header(title, subtitle=None):
    """Renders a consistent header across pages."""
    inject_custom_css()
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.divider()
