"""Centralized visual theme settings for the dashboard UI and charts.

Update these constants to change styles globally.
"""

import plotly.express as px

APP_FONT_STACK = '"Avenir", "Avenir Next", "Segoe UI", Tahoma, sans-serif'
PLOTLY_FONT_FAMILY = "Avenir, Avenir Next, Segoe UI, Tahoma, sans-serif"
APP_LAYOUT_MAX_WIDTH_PX = 1500
APP_CENTERED_CONTAINER_STYLE = {
    "width": "100%",
    "maxWidth": f"{APP_LAYOUT_MAX_WIDTH_PX}px",
    "marginLeft": "auto",
    "marginRight": "auto",
}

# Keep these unified so chart typography can be tuned from one place.
PLOTLY_BASE_FONT_SIZE = 14
PLOTLY_LEGEND_FONT_SIZE = PLOTLY_BASE_FONT_SIZE
PLOTLY_LABEL_FONT_SIZE = PLOTLY_BASE_FONT_SIZE + 2
PLOTLY_HOVER_FONT_SIZE = PLOTLY_BASE_FONT_SIZE
PLOTLY_AXIS_TICK_FONT_SIZE = 12

PLOTLY_TEMPLATE_NAME = "ipl"
PLOTLY_PALETTE = px.colors.qualitative.Prism

PLOTLY_COLORS = {
    "innings_1": PLOTLY_PALETTE[0],
    "innings_2": "#FF7F0E",
    "win_probability": PLOTLY_PALETTE[2],
    "reference_line": PLOTLY_PALETTE[2],
    "leaderboard_primary": PLOTLY_PALETTE[0],
    "leaderboard_secondary": PLOTLY_PALETTE[1],
    "leaderboard_impact": "#3498db",
    "success_card": "#28a745",
}

PLOTLY_HEADER_TITLE_Y = 0.980
PLOTLY_HEADER_LEGEND_Y = 0.950
PLOTLY_HEADER_PLOT_TOP = 1
PLOTLY_HEADER_MARGIN_TOP = 20
PLOTLY_HEADER_MARGIN_LEFT = 90
PLOTLY_HEADER_MARGIN_RIGHT = 90
PLOTLY_REFERENCE_LINE_WIDTH = 1
