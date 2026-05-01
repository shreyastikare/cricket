from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

try:
    from theme_config import APP_CENTERED_CONTAINER_STYLE, APP_FONT_STACK
except ModuleNotFoundError:
    from app.theme_config import APP_CENTERED_CONTAINER_STYLE, APP_FONT_STACK

NAV_ITEMS = [
    {"label": "Home", "href": "/"},
    {"label": "Match Analysis", "href": "/match-analysis"},
    {"label": "Team Analysis", "href": "/team-analysis"},
    {"label": "Season Leaderboard", "href": "/season-leaderboard"},
    {"label": "About", "href": "/about"},
]

PAGE_WRAPPER_STYLE = {
    "minHeight": "100vh",
    "backgroundColor": "#f8f9fa",
    "fontFamily": APP_FONT_STACK,
    "display": "flex",
    "flexDirection": "column",
}

APP_CONTENT_SHELL_STYLE = {
    **APP_CENTERED_CONTAINER_STYLE,
    "flex": "1 0 auto",
}

NAVBAR_INNER_STYLE = {
    **APP_CENTERED_CONTAINER_STYLE,
    "padding": "0 24px",
    "boxSizing": "border-box",
}

CONTENT_STYLE = {
    "padding": "18px 24px 28px",
    "boxSizing": "border-box",
}

PANE_STYLE = {
    "padding": "16px 0",
}

FILTER_ROW_STYLE = {
    "display": "flex",
    "gap": "12px",
    "alignItems": "flex-end",
    "flexWrap": "nowrap",
}

YEAR_FILTER_STYLE = {
    "flex": "0 0 110px",
}

TEAM_FILTER_STYLE = {
    "flex": "0 0 240px",
}

SEASON_FILTER_STYLE = {
    "flex": "0 0 160px",
}

MATCH_CELL_STYLE = {
    "flex": "1 1 auto",
    "minWidth": 0,
    "maxWidth": "660px",
}

FOOTER_DISCLAIMER = (
    "This website is an independent analytics project and is not affiliated with, "
    "endorsed by, or sponsored by the Indian Premier League (IPL), the BCCI, or any "
    "IPL franchise. Team names and logos are used solely for identification and "
    "informational purposes. All trademarks are the property of their respective owners."
)


def build_navbar(pathname: str | None) -> dbc.Navbar:
    nav_items = []
    for item in NAV_ITEMS:
        is_active = pathname == item["href"] or (item["href"] == "/" and pathname in (None, ""))
        active_style = (
            {
                "color": "white",
                "fontWeight": 700,
                "textDecoration": "underline",
                "textUnderlineOffset": "5px",
                "textDecorationThickness": "2px",
            }
            if is_active
            else None
        )
        nav_items.append(
            dbc.NavItem(
                dbc.NavLink(
                    item["label"],
                    href=item["href"],
                    active=is_active,
                    style=active_style,
                )
            )
        )

    return dbc.Navbar(
        html.Div(
            html.Div(
                [
                    dbc.NavbarBrand("IPL Match Analysis", href="/", class_name="fw-bold"),
                    dbc.Nav(nav_items, navbar=True, class_name="ms-3"),
                ],
                className="d-flex align-items-center",
            ),
            style=NAVBAR_INNER_STYLE,
        ),
    )


def build_footer() -> html.Footer:
    return html.Footer(
        html.Div(
            [
                html.Div(
                    [
                        "Created by ",
                        html.A(
                            "Shreyas Tikare",
                            href="https://www.shreyastikare.com",
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                    ],
                    className="site-footer-credit",
                ),
                html.Div(
                    [
                        html.A(
                            "LinkedIn",
                            href="https://www.linkedin.com/in/shreyastikare/",
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                        html.A(
                            "GitHub",
                            href="https://github.com/shreyastikare",
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                    ],
                    className="site-footer-links",
                ),
                html.P(FOOTER_DISCLAIMER, className="site-footer-disclaimer"),
            ],
            className="site-footer-inner",
        ),
        className="site-footer",
    )


def build_landing_page() -> html.Div:
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H1("IPL Match Analysis", style={"margin": "0 0 8px"}),
                    html.P(
                        "This landing page is intentionally minimal for now."
                        " It will hold the project overview and key write-up sections.",
                        style={"margin": 0},
                    ),
                ],
                style=PANE_STYLE,
            )
        ],
        style=CONTENT_STYLE,
    )


def build_about_page() -> html.Div:
    paragraph_style = {
        "margin": "0 0 14px",
        "lineHeight": 1.65,
        "maxWidth": "900px",
    }
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H1("IPL Match Analytics", style={"margin": "0 0 16px"}),
                    html.P(
                        (
                            "This project is an interactive analytics platform for exploring IPL "
                            "matches through a ball-by-ball lens. It combines historical data with "
                            "live match ingestion to generate real-time insights, including win "
                            "probability, projected scores, and team performance profiles."
                        ),
                        style=paragraph_style,
                    ),
                    html.P(
                        (
                            "At its core is a probabilistic modeling framework built on match "
                            "state variables such as runs, wickets, and overs remaining. A custom "
                            "resource-based feature engineering approach is used to quantify game "
                            "context, allowing the models to capture how match dynamics evolve "
                            "across innings and phases of play. The system is trained on historical "
                            "IPL seasons and validated on recent data to ensure strong predictive "
                            "performance and calibration."
                        ),
                        style=paragraph_style,
                    ),
                    html.P(
                        "The application is designed to make these insights intuitive and accessible. Users can:",
                        style=paragraph_style,
                    ),
                    html.Ul(
                        [
                            html.Li("Track win probability and momentum shifts throughout a match"),
                            html.Li("Analyze team strengths across batting and bowling phases"),
                            html.Li("Explore season-level comparisons through interactive leaderboards"),
                        ],
                        style={
                            "margin": "0 0 16px 22px",
                            "padding": 0,
                            "lineHeight": 1.65,
                            "maxWidth": "900px",
                        },
                    ),
                    html.P(
                        (
                            "Beyond visualization, the project emphasizes sound data engineering "
                            "and modeling practices. A unified SQLite backend supports both "
                            "historical and live data, while a modular pipeline handles ingestion, "
                            "feature construction, and prediction in a consistent and reproducible way."
                        ),
                        style=paragraph_style,
                    ),
                    html.P(
                        (
                            "This platform serves both as a tool for cricket analysis and as a "
                            "demonstration of applied data science, bringing together data "
                            "engineering, statistical modeling, and interactive visualization in a "
                            "cohesive system."
                        ),
                        style={**paragraph_style, "marginBottom": 0},
                    ),
                ],
                style=PANE_STYLE,
            )
        ],
        style=CONTENT_STYLE,
    )


def build_match_analysis_page(
    year_options: list[dict[str, int]],
    team_options: list[dict[str, str]],
) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Year", style={"fontWeight": 600}),
                                    dcc.Dropdown(
                                        id="year-dropdown",
                                        options=year_options,
                                        value=None,
                                        clearable=True,
                                        placeholder="All years",
                                        maxHeight=320,
                                    ),
                                ],
                                style=YEAR_FILTER_STYLE,
                            ),
                            html.Div(
                                children=[
                                    html.Label("Team (Optional)", style={"fontWeight": 600}),
                                    dcc.Dropdown(
                                        id="team-filter-dropdown",
                                        options=team_options,
                                        value=None,
                                        clearable=True,
                                        placeholder="All teams",
                                        maxHeight=320,
                                    ),
                                ],
                                style=TEAM_FILTER_STYLE,
                            ),
                            html.Div(
                                children=[
                                    html.Label("Match", style={"fontWeight": 600}),
                                    html.Div(
                                        children=[
                                            html.Div(
                                                dcc.Dropdown(
                                                    id="match-dropdown",
                                                    options=[],
                                                    value=None,
                                                    clearable=False,
                                                    placeholder="Select a match",
                                                    maxHeight=320,
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                style={"flex": "1 1 auto", "minWidth": 0},
                                            ),
                                            html.Div(
                                                "Match ID: -",
                                                id="match-id-inline",
                                                style={
                                                    "paddingLeft": "10px",
                                                    "fontSize": "16px",
                                                    "fontWeight": 400,
                                                    "whiteSpace": "nowrap",
                                                },
                                            ),
                                        ],
                                        style={"display": "flex", "alignItems": "center"},
                                    ),
                                ],
                                style=MATCH_CELL_STYLE,
                            ),
                            html.Div(
                                children=[
                                    dbc.Checklist(
                                        id="auto-refresh-checkbox",
                                        options=[{"label": "Refresh automatically", "value": 1}],
                                        value=[],
                                        switch=True,
                                        label_style={
                                            "fontSize": "16px",
                                            "fontWeight": 400,
                                            "marginBottom": 0,
                                            "cursor": "pointer",
                                        },
                                        input_style={
                                            "cursor": "pointer",
                                            "transform": "scale(1.18)",
                                            "marginRight": "10px",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "alignItems": "flex-end",
                                    "justifyContent": "flex-end",
                                    "flex": "0 0 auto",
                                    "marginLeft": "auto",
                                },
                            ),
                        ],
                        style={**FILTER_ROW_STYLE, "marginBottom": "14px"},
                    ),
                    html.Div(id="match-dashboard", style={"marginTop": "16px"}),
                ],
                style=PANE_STYLE,
            ),
        ],
        style=CONTENT_STYLE,
    )


def build_season_leaderboard_page(
    season_options: list[dict[str, int]],
    default_season: int | None,
) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H2("Season Leaderboard", style={"margin": 0}),
                ],
                style=PANE_STYLE,
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Label("Season", style={"fontWeight": 600}),
                            dcc.Dropdown(
                                id="leaderboard-season-dropdown",
                                options=season_options,
                                value=default_season,
                                clearable=False,
                                placeholder="Select season",
                                maxHeight=320,
                            ),
                        ],
                        style=SEASON_FILTER_STYLE,
                    ),
                    html.Div(id="season-leaderboard-content", style={"marginTop": "16px"}),
                ],
                style=PANE_STYLE,
            ),
        ],
        style=CONTENT_STYLE,
    )


def build_team_analysis_page(
    season_options: list[dict[str, int]],
    default_season: int | None,
    team_options: list[dict[str, str]],
    default_team: str | None,
) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        id="team-analysis-header",
                        children=html.Div(
                            [
                                html.H2("Team Analysis", style={"margin": "0 0 8px"}),
                                html.P(
                                    "Season-level team profile built from existing match impact data.",
                                    style={"margin": 0},
                                ),
                            ]
                        ),
                        style={"minWidth": 0},
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Year", style={"fontWeight": 600}),
                                    dcc.Dropdown(
                                        id="team-analysis-season-dropdown",
                                        options=season_options,
                                        value=default_season,
                                        clearable=False,
                                        placeholder="Select season",
                                        maxHeight=320,
                                    ),
                                ],
                                style={**SEASON_FILTER_STYLE, "maxWidth": "180px", "marginBottom": 0},
                            ),
                            html.Div(
                                children=[
                                    html.Label("Team", style={"fontWeight": 600}),
                                    dcc.Dropdown(
                                        id="team-analysis-team-dropdown",
                                        options=team_options,
                                        value=default_team,
                                        clearable=False,
                                        placeholder="Select team",
                                        maxHeight=320,
                                    ),
                                ],
                                style={**TEAM_FILTER_STYLE, "maxWidth": "260px", "marginBottom": 0},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "10px",
                            "justifyContent": "flex-end",
                            "alignItems": "flex-end",
                            "flexWrap": "nowrap",
                            "marginLeft": "auto",
                        },
                    ),
                ],
                style={
                    **PANE_STYLE,
                    "display": "flex",
                    "flexDirection": "row",
                    "gap": "16px",
                    "alignItems": "flex-start",
                    "justifyContent": "space-between",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                children=[
                    html.Div(id="team-analysis-content", style={"marginTop": "16px"}),
                ],
                style=PANE_STYLE,
            ),
        ],
        style=CONTENT_STYLE,
    )
