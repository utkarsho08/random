import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

def set_plotly_theme():
    """Configures the default Plotly theme for the application."""
    # Define a custom template based on the Plotly dark or a clean light theme
    custom_template = pio.templates["plotly_white"]
    
    # Customize layout
    custom_template.layout.font.family = "Inter, sans-serif"
    custom_template.layout.title.font.size = 20
    custom_template.layout.title.font.color = "#333333"
    
    # Set default color sequences
    custom_template.layout.colorway = px.colors.qualitative.Pastel
    
    # Apply as default
    pio.templates.default = "plotly_white"
    
def get_custom_color_scale():
    """Returns a custom color scale for heatmaps and sequential data."""
    return px.colors.sequential.Blues
    
# Application-wide CSS
CSS_STYLES = """
<style>
    /* Main typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* KPI Cards styling */
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #4CAF50;
    }
    .kpi-title {
        font-size: 14px;
        color: #666;
        margin-bottom: 10px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #1f2937;
    }
    
    /* Dark mode adjustments (if Streamlit theme is dark) */
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background-color: #1e1e1e;
            border-left: 5px solid #2e7d32;
        }
        .kpi-title {
            color: #aaa;
        }
        .kpi-value {
            color: #fff;
        }
    }
</style>
"""
