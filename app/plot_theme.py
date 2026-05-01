from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

try:
    from theme_config import (
        PLOTLY_AXIS_TICK_FONT_SIZE,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_COLORS,
        PLOTLY_FONT_FAMILY,
        PLOTLY_HEADER_LEGEND_Y,
        PLOTLY_HOVER_FONT_SIZE,
        PLOTLY_LEGEND_FONT_SIZE,
        PLOTLY_PALETTE,
        PLOTLY_TEMPLATE_NAME,
    )
except ModuleNotFoundError:
    from app.theme_config import (
        PLOTLY_AXIS_TICK_FONT_SIZE,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_COLORS,
        PLOTLY_FONT_FAMILY,
        PLOTLY_HEADER_LEGEND_Y,
        PLOTLY_HOVER_FONT_SIZE,
        PLOTLY_LEGEND_FONT_SIZE,
        PLOTLY_PALETTE,
        PLOTLY_TEMPLATE_NAME,
    )


def ensure_plotly_theme_registered() -> None:
    if PLOTLY_TEMPLATE_NAME not in pio.templates:
        pio.templates[PLOTLY_TEMPLATE_NAME] = go.layout.Template(pio.templates["plotly_white"])

    template = pio.templates[PLOTLY_TEMPLATE_NAME]
    template.layout.colorway = PLOTLY_PALETTE
    template.layout.font = dict(
        family=PLOTLY_FONT_FAMILY,
        size=PLOTLY_BASE_FONT_SIZE,
    )
    template.layout.legend = dict(
        font=dict(family=PLOTLY_FONT_FAMILY, size=PLOTLY_LEGEND_FONT_SIZE),
    )
    template.layout.hoverlabel = dict(
        font=dict(family=PLOTLY_FONT_FAMILY, size=PLOTLY_HOVER_FONT_SIZE),
    )


def apply_plot_theme(fig):
    ensure_plotly_theme_registered()
    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        font=dict(family=PLOTLY_FONT_FAMILY, size=PLOTLY_BASE_FONT_SIZE),
        legend_font=dict(family=PLOTLY_FONT_FAMILY, size=PLOTLY_LEGEND_FONT_SIZE),
        hoverlabel=dict(font=dict(family=PLOTLY_FONT_FAMILY, size=PLOTLY_HOVER_FONT_SIZE)),
        xaxis=dict(tickfont=dict(size=PLOTLY_AXIS_TICK_FONT_SIZE, family=PLOTLY_FONT_FAMILY)),
        yaxis=dict(tickfont=dict(size=PLOTLY_AXIS_TICK_FONT_SIZE, family=PLOTLY_FONT_FAMILY)),
    )


def header_legend_layout():
    return dict(
        orientation="h",
        x=0.5,
        y=PLOTLY_HEADER_LEGEND_Y,
        xref="container",
        yref="container",
        xanchor="center",
        yanchor="top",
    )


def innings_color(innings: int) -> str:
    return PLOTLY_COLORS["innings_1"] if int(innings) == 1 else PLOTLY_COLORS["innings_2"]


def leaderboard_primary_color() -> str:
    return PLOTLY_COLORS["leaderboard_primary"]


ensure_plotly_theme_registered()
