from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from math import ceil
import os
import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, quote

import ipl
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, dash_table
from ipl.storage.team_profile_metrics import load_team_profile_metrics, refresh_team_profile_metrics_for_season

try:
    from leaderboard import Leaderboard, get_default_leaderboard_season, get_finalized_season_options
    from match import Match, resource_params
    from match_selector import (
        get_match_options,
        get_team_options_for_year,
        get_year_options,
    )
    from ui import (
        APP_CONTENT_SHELL_STYLE,
        PAGE_WRAPPER_STYLE,
        build_about_page,
        build_footer,
        build_match_analysis_page,
        build_navbar,
        build_season_leaderboard_page,
        build_team_analysis_page,
    )
    from team_analysis import (
        BENCHMARK_END_SEASON,
        BENCHMARK_START_SEASON,
        BATTING_ROLE_ORDER,
        BOWLING_PHASE_ORDER,
        BOWLING_PRESSURE_HELP,
        PHASE_COLUMNS,
        aggregate_team_phase_impact,
        benchmark_seasons_2008_2025,
        best_phase_name,
        build_match_table_rows,
        compute_profile_percentiles,
        compute_profile_season_ranks,
        compute_team_profile_metrics,
        filter_team_season_matches,
        opponent_for_match,
        profile_metric_subtitle,
        select_profile_strength_cards,
        team_record,
        team_phase_heatmap_height,
        team_standings_rank,
        top_impact_player,
        top_impact_player_by_match,
    )
    from theme_config import (
        APP_FONT_STACK,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_COLORS,
        PLOTLY_FONT_FAMILY,
        PLOTLY_HEADER_MARGIN_LEFT,
        PLOTLY_HEADER_MARGIN_RIGHT,
        PLOTLY_HEADER_MARGIN_TOP,
        PLOTLY_HEADER_TITLE_Y,
    )
    from team_logos import team_logo_path
except ModuleNotFoundError:
    from app.leaderboard import Leaderboard, get_default_leaderboard_season, get_finalized_season_options
    from app.match import Match, resource_params
    from app.match_selector import (
        get_match_options,
        get_team_options_for_year,
        get_year_options,
    )
    from app.ui import (
        APP_CONTENT_SHELL_STYLE,
        PAGE_WRAPPER_STYLE,
        build_about_page,
        build_footer,
        build_match_analysis_page,
        build_navbar,
        build_season_leaderboard_page,
        build_team_analysis_page,
    )
    from app.team_analysis import (
        BENCHMARK_END_SEASON,
        BENCHMARK_START_SEASON,
        BATTING_ROLE_ORDER,
        BOWLING_PHASE_ORDER,
        BOWLING_PRESSURE_HELP,
        PHASE_COLUMNS,
        aggregate_team_phase_impact,
        benchmark_seasons_2008_2025,
        best_phase_name,
        build_match_table_rows,
        compute_profile_percentiles,
        compute_profile_season_ranks,
        compute_team_profile_metrics,
        filter_team_season_matches,
        opponent_for_match,
        profile_metric_subtitle,
        select_profile_strength_cards,
        team_record,
        team_phase_heatmap_height,
        team_standings_rank,
        top_impact_player,
        top_impact_player_by_match,
    )
    from app.theme_config import (
        APP_FONT_STACK,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_COLORS,
        PLOTLY_FONT_FAMILY,
        PLOTLY_HEADER_MARGIN_LEFT,
        PLOTLY_HEADER_MARGIN_RIGHT,
        PLOTLY_HEADER_MARGIN_TOP,
        PLOTLY_HEADER_TITLE_Y,
    )
    from app.team_logos import team_logo_path

try:
    from plot_theme import apply_plot_theme
except ModuleNotFoundError:
    try:
        from app.plot_theme import apply_plot_theme
    except ModuleNotFoundError:
        def apply_plot_theme(fig):
            return fig

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="IPL Match Analysis",
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder=str(Path(__file__).resolve().parents[1] / "assets"),
)
server = app.server

CARD_VALUE_FONT_SIZE = "16px"
LIVE_REFRESH_INTERVAL_MS = 60_000
TERMINAL_MATCH_STATUSES = {"complete", "abandoned", "no_result"}
DB_PATH = os.getenv('DB_PATH', 'data/sqlite/ipl.db')
PLAYBYPLAY_DESC_OVERS_PER_PAGE = 3
PLAYBYPLAY_RIBBON_OVERS_PER_PAGE = 4


def _team_logo_img(team: object, class_name: str = "team-logo", *, alt_team: object | None = None):
    path = team_logo_path(team)
    if not path:
        return None
    alt_text = "" if alt_team is None else str(alt_team)
    if not alt_text:
        alt_text = str(team or "").strip()
    return html.Img(src = app.get_asset_url(path), alt = f"{alt_text} logo", className = class_name)


def _team_text_with_logo(team: object, text_component, class_name: str = "team-logo-inline") -> html.Div:
    logo = _team_logo_img(team)
    children = [child for child in [logo, text_component] if child is not None]
    return html.Div(children, className = class_name)

app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="selected-match-store", data={"match_id": None, "auto_refresh": False}, storage_type="local"),
        dcc.Store(id="playbyplay-desc-page-store", data=1, storage_type="memory"),
        dcc.Store(id="playbyplay-ribbon-page-store", data=1, storage_type="memory"),
        dcc.Interval(id="live-refresh-interval", interval=LIVE_REFRESH_INTERVAL_MS, n_intervals=0),
        html.Div(id="nav-container"),
        html.Div(id="page-content", style=APP_CONTENT_SHELL_STYLE),
        build_footer(),
        html.Div(id="refresh-client-hook", style={"display": "none"}),
    ],
    style=PAGE_WRAPPER_STYLE,
)


@lru_cache(maxsize = 64)
def load_match(match_id: int, cache_token: int = 0) -> Match:
    return Match(match_id)


@lru_cache(maxsize = 32)
def load_leaderboard(season: int, cache_token: int = 0) -> Leaderboard:
    return Leaderboard(season=season)


@lru_cache(maxsize = 32)
def load_leaderboard_overview(season: int, cache_token: int = 0) -> Leaderboard:
    leaderboard = Leaderboard(season=season, include_impact=False)
    leaderboard.player_impact_stats = _cached_player_impact_stats_for_season(int(season))
    return leaderboard


def _cached_player_impact_stats_for_season(season: int) -> pd.DataFrame:
    query = """
        WITH season_matches AS (
            SELECT CAST(match_id AS INTEGER) AS match_id
            FROM match_list
            WHERE CAST(year AS INTEGER) = ?
              AND LOWER(TRIM(COALESCE(status, ''))) = 'complete'
        ), version_rank AS (
            SELECT
                CAST(match_id AS INTEGER) AS match_id,
                model_version,
                MAX(COALESCE(updated_at, created_at, prediction_ts)) AS latest_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(match_id AS INTEGER)
                    ORDER BY MAX(COALESCE(updated_at, created_at, prediction_ts)) DESC
                ) AS rn
            FROM prediction_history
            WHERE model_type = 'win_probability'
              AND CAST(match_id AS INTEGER) IN (SELECT match_id FROM season_matches)
            GROUP BY CAST(match_id AS INTEGER), model_version
        ), latest_predictions AS (
            SELECT p.*
            FROM prediction_history p
            JOIN version_rank v
              ON CAST(p.match_id AS INTEGER) = v.match_id
             AND p.model_version = v.model_version
             AND v.rn = 1
            WHERE p.model_type = 'win_probability'
        ), player_impacts AS (
            SELECT
                CASE
                    WHEN b.batter_player_id IS NOT NULL THEN 'id:' || CAST(CAST(b.batter_player_id AS INTEGER) AS TEXT)
                    ELSE 'name:' || TRIM(COALESCE(b.batter, ''))
                END AS player_key,
                COALESCE(NULLIF(TRIM(b.batter_canonical_name), ''), TRIM(COALESCE(b.batter, ''))) AS player,
                TRIM(COALESCE(b.batting_team, '')) AS team,
                CASE
                    WHEN CAST(COALESCE(b.wicket_taken, 0) AS INTEGER) = 1 THEN 0.0
                    WHEN b.extra_type IS NULL OR CAST(COALESCE(b.batter_runs, 0) AS REAL) > 0 THEN
                        CASE CAST(COALESCE(b.innings, 0) AS INTEGER)
                            WHEN 1 THEN CAST(COALESCE(p.prob_batting_diff, 0) AS REAL)
                            WHEN 2 THEN CAST(COALESCE(p.prob_bowling_diff, 0) AS REAL)
                            ELSE 0.0
                        END
                    ELSE 0.0
                END AS batting_impact,
                0.0 AS bowling_impact
            FROM ball_by_ball b
            JOIN latest_predictions p
              ON p.ball_id = b.ball_id
             AND CAST(p.match_id AS INTEGER) = CAST(b.match_id AS INTEGER)
            WHERE CAST(b.innings AS INTEGER) IN (1, 2)
            UNION ALL
            SELECT
                CASE
                    WHEN b.bowler_player_id IS NOT NULL THEN 'id:' || CAST(CAST(b.bowler_player_id AS INTEGER) AS TEXT)
                    ELSE 'name:' || TRIM(COALESCE(b.bowler, ''))
                END AS player_key,
                COALESCE(NULLIF(TRIM(b.bowler_canonical_name), ''), TRIM(COALESCE(b.bowler, ''))) AS player,
                TRIM(COALESCE(b.bowling_team, '')) AS team,
                0.0 AS batting_impact,
                CASE CAST(COALESCE(b.innings, 0) AS INTEGER)
                    WHEN 1 THEN CAST(COALESCE(p.prob_bowling_diff, 0) AS REAL)
                    WHEN 2 THEN -CAST(COALESCE(p.prob_bowling_diff, 0) AS REAL)
                    ELSE 0.0
                END AS bowling_impact
            FROM ball_by_ball b
            JOIN latest_predictions p
              ON p.ball_id = b.ball_id
             AND CAST(p.match_id AS INTEGER) = CAST(b.match_id AS INTEGER)
            WHERE CAST(b.innings AS INTEGER) IN (1, 2)
        )
        SELECT
            player_key AS "Player Key",
            player AS "Player",
            GROUP_CONCAT(DISTINCT team) AS "Team",
            0 AS "Matches Played",
            SUM(batting_impact) AS "Batting Impact",
            SUM(bowling_impact) AS "Bowling Impact",
            SUM(batting_impact + bowling_impact) AS "Total Impact",
            0.0 AS "Avg Total Impact / Match"
        FROM player_impacts
        WHERE TRIM(player_key) != '' AND TRIM(player) != ''
        GROUP BY player_key
        ORDER BY "Total Impact" DESC, "Player"
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(query, conn, params=[int(season)])
    except Exception:
        return pd.DataFrame()


def _match_needs_live_reload(match_id: int) -> bool:
    query = """
        SELECT status
        FROM match_list
        WHERE match_id = ?
        LIMIT 1
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(query, (int(match_id),)).fetchone()
    except Exception:
        return True

    if not row:
        return True
    status = "" if row[0] is None else str(row[0]).strip().lower()
    return status not in TERMINAL_MATCH_STATUSES


def _match_cache_token(match_id: int) -> int:
    query = """
        SELECT
            COALESCE(COUNT(b.ball_id), 0),
            COALESCE(MAX(b.updated_at), ''),
            COALESCE(m.status, ''),
            COALESCE(m.updated_at, ''),
            COALESCE(m.last_successful_fetch_ts, '')
        FROM match_list m
        LEFT JOIN ball_by_ball b
          ON CAST(b.match_id AS INTEGER) = m.match_id
        WHERE m.match_id = ?
        GROUP BY m.match_id
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(query, (int(match_id),)).fetchone()
    except Exception:
        return 0

    if not row:
        return 0

    token_raw = "|".join("" if value is None else str(value) for value in row)
    return abs(hash(token_raw))


def _leaderboard_cache_token(season: int) -> int:
    query = """
        SELECT
            COUNT(*) AS n_rows,
            COALESCE(MAX(updated_at), '') AS max_updated_at,
            COALESCE(MAX(last_successful_fetch_ts), '') AS max_fetch_ts
        FROM match_list
        WHERE year = ?
          AND LOWER(TRIM(COALESCE(status, ''))) IN ('complete', 'abandoned', 'no_result')
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(query, (int(season),)).fetchone()
    except Exception:
        return 0

    if not row:
        return 0
    token_raw = "|".join("" if value is None else str(value) for value in row)
    return abs(hash(token_raw))


def _format_date(value: str | None) -> str:
    if value is None:
        return "-"
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%m/%d/%Y")
    except (TypeError, ValueError):
        return str(value)


def _system_timezone():
    return datetime.now().astimezone().tzinfo


def _abbreviate_timezone_name(name: str | None) -> str:
    text = "" if name is None else str(name).strip()
    if not text:
        return ""
    if " " not in text and 2 <= len(text) <= 6:
        return text.upper()

    known = {
        "Eastern Standard Time": "EST",
        "Eastern Daylight Time": "EDT",
        "Central Standard Time": "CST",
        "Central Daylight Time": "CDT",
        "Mountain Standard Time": "MST",
        "Mountain Daylight Time": "MDT",
        "Pacific Standard Time": "PST",
        "Pacific Daylight Time": "PDT",
        "Coordinated Universal Time": "UTC",
        "Greenwich Mean Time": "GMT",
        "India Standard Time": "IST",
    }
    if text in known:
        return known[text]

    parts = [part for part in text.replace("-", " ").split() if part]
    initials = "".join(part[0].upper() for part in parts if part[0].isalpha())
    if 2 <= len(initials) <= 6:
        return initials
    return text


def _format_local_time(dt_local: datetime) -> str:
    clock_text = dt_local.strftime("%I:%M %p").lstrip("0")
    tz_text = _abbreviate_timezone_name(dt_local.tzname())
    return f"{clock_text} {tz_text}".strip()


def _parse_dt_utc(value: str | None):
    if value is None:
        return None
    ts = pd.to_datetime(value, utc = True, errors = "coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _format_local_dt(value: str | None) -> tuple[str, str]:
    dt_utc = _parse_dt_utc(value)
    if dt_utc is None:
        return "-", "-"
    dt_local = dt_utc.astimezone(_system_timezone())
    return dt_local.strftime("%Y-%m-%d"), _format_local_time(dt_local)


def _format_starts_in(scheduled_start_ts: str | None) -> str:
    dt_utc = _parse_dt_utc(scheduled_start_ts)
    if dt_utc is None:
        return "starts in -"

    delta = dt_utc - datetime.now(tz = UTC)
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return "starts in 0m"

    days, rem = divmod(total_seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes = rem // 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return f"starts in {' '.join(parts)}"


def _landing_row_season(match_date, scheduled_start_ts) -> int | None:
    for value in [match_date, scheduled_start_ts]:
        if value is None:
            continue
        text = str(value).strip()
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
    return None


def _landing_sections() -> dict[str, list[dict[str, str | int | None]]]:
    now_utc = datetime.now(tz = UTC)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    today_utc = now_utc.date()
    two_days_date = (today_utc + timedelta(days = 2)).strftime("%Y-%m-%d")
    today_date = today_utc.strftime("%Y-%m-%d")

    live_query = """
        SELECT
            CAST(ms.match_id AS INTEGER) AS match_id,
            COALESCE(ms.match_date, ml.date) AS match_date,
            ms.scheduled_start_ts,
            COALESCE(ms.team1, ml.bat_first) AS team1,
            COALESCE(ms.team2, ml.bowl_first) AS team2
        FROM match_schedule ms
        LEFT JOIN match_list ml
          ON ml.match_id = CAST(ms.match_id AS INTEGER)
        WHERE LOWER(TRIM(COALESCE(ml.status, ms.status, ''))) IN ('live', 'innings_break', 'delayed')
        ORDER BY ms.scheduled_start_ts DESC, CAST(ms.match_id AS INTEGER) DESC;
    """

    recent_query = """
        SELECT
            ml.match_id AS match_id,
            COALESCE(ms.match_date, ml.date) AS match_date,
            ms.scheduled_start_ts,
            COALESCE(ms.team1, ml.bat_first) AS team1,
            COALESCE(ms.team2, ml.bowl_first) AS team2
        FROM match_list ml
        LEFT JOIN match_schedule ms
          ON CAST(ms.match_id AS INTEGER) = ml.match_id
        WHERE LOWER(TRIM(COALESCE(ml.status, ''))) IN ('complete', 'abandoned', 'no_result')
        ORDER BY
            COALESCE(ms.scheduled_start_ts, ml.date || 'T00:00:00Z') DESC,
            ml.match_id DESC
        LIMIT 5;
    """

    upcoming_query = """
        SELECT
            CAST(ms.match_id AS INTEGER) AS match_id,
            ms.match_date AS match_date,
            ms.scheduled_start_ts,
            ms.team1 AS team1,
            ms.team2 AS team2
        FROM match_schedule ms
        WHERE ms.scheduled_start_ts IS NOT NULL
          AND ms.match_date IS NOT NULL
          AND ms.match_date BETWEEN ? AND ?
          AND ms.scheduled_start_ts >= ?
          AND LOWER(TRIM(COALESCE(ms.status, ''))) NOT IN ('live', 'complete', 'abandoned', 'no_result')
        ORDER BY ms.scheduled_start_ts ASC, CAST(ms.match_id AS INTEGER) ASC;
    """

    empty = {"live": [], "recent": [], "upcoming": []}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            live_df = pd.read_sql_query(live_query, conn)
            recent_df = pd.read_sql_query(recent_query, conn)
            upcoming_df = pd.read_sql_query(upcoming_query, conn, params = [today_date, two_days_date, now_iso])
    except Exception:
        return empty

    def _rows(df: pd.DataFrame, *, include_starts_in: bool) -> list[dict[str, str | int | None]]:
        if df.empty:
            return []
        out = []
        for row in df.to_dict("records"):
            scheduled = row.get("scheduled_start_ts")
            date_text, time_text = _format_local_dt(None if scheduled is None else str(scheduled))
            if date_text == "-" and row.get("match_date") is not None:
                date_text = _format_date(str(row.get("match_date")))
            season = _landing_row_season(row.get("match_date"), scheduled)
            out.append(
                {
                    "match_id": int(row.get("match_id")),
                    "season": season,
                    "date": date_text,
                    "time": time_text,
                    "team1": "-" if row.get("team1") is None else str(row.get("team1")),
                    "team2": "-" if row.get("team2") is None else str(row.get("team2")),
                    "starts_in": _format_starts_in(None if scheduled is None else str(scheduled)) if include_starts_in else None,
                }
            )
        return out

    return {
        "live": _rows(live_df, include_starts_in = False),
        "recent": _rows(recent_df, include_starts_in = False),
        "upcoming": _rows(upcoming_df, include_starts_in = True),
    }


def _landing_match_line(item: dict[str, str | int | None]) -> str:
    return f"{item['date']} | {item['time']} | {item['team1']} vs. {item['team2']}"


def _team_analysis_href(team: str | int | None, season: str | int | None) -> str:
    team_text = "" if team is None else str(team).strip()
    if not team_text or team_text == "-":
        return "/team-analysis"

    parts = []
    if season is not None:
        parts.append(f"season={quote(str(season))}")
    parts.append(f"team={quote(team_text)}")
    return f"/team-analysis?{'&'.join(parts)}"


def _landing_team_link(team: str | int | None, season: str | int | None):
    team_text = "-" if team is None else str(team)
    if not team_text.strip() or team_text == "-":
        return html.Span(team_text)
    return dcc.Link(team_text, href = _team_analysis_href(team_text, season))


def _landing_match_node(
    item: dict[str, str | int | None],
    *,
    link_date_to_match: bool,
    include_starts_in: bool,
    show_time: bool = True,
):
    date_text = str(item.get("date", "-"))
    date_node = (
        dcc.Link(date_text, href = f"/match-analysis?match_id={item['match_id']}")
        if link_date_to_match and item.get("match_id") is not None
        else html.Span(date_text)
    )
    children = [
        date_node,
    ]
    if show_time:
        children.append(html.Span(f" | {item.get('time', '-')} | "))
    else:
        children.append(html.Span(" | "))
    children.extend([
        _landing_team_link(item.get("team1"), item.get("season")),
        html.Span(" vs. "),
        _landing_team_link(item.get("team2"), item.get("season")),
    ])
    if include_starts_in and item.get("starts_in"):
        children.append(html.Span(f" ({item['starts_in']})"))
    return html.Span(children)


def _landing_list(
    title: str,
    rows: list[dict[str, str | int | None]],
    *,
    link_items: bool,
    include_starts_in: bool = False,
    show_time: bool = True,
):
    if not rows:
        body = html.Div("No matches found.", style={"fontSize": "14px"})
    else:
        bullets = []
        for item in rows:
            node = _landing_match_node(
                item,
                link_date_to_match = link_items,
                include_starts_in = include_starts_in,
                show_time = show_time,
            )
            bullets.append(html.Li(node, style={"marginBottom": "6px"}))
        body = html.Ul(bullets, style={"margin": 0, "paddingLeft": "20px"})

    return _card(
        [
            html.Div(title, style={"fontWeight": 700, "fontSize": "18px", "marginBottom": "8px"}),
            body,
        ],
        body_class_name = "p-3",
    )


def _build_landing_page() -> html.Div:
    sections = _landing_sections()
    return html.Div(
        children=[
            html.Div(
                [
                    html.H1("IPL Match Analysis", style={"margin": "0 0 8px"}),
                    html.P(
                        (
                            "This project grew out of a personal interest in cricket and data analytics while completing my "
                            "Master's in Analytics at the Institute for Advanced Analytics at NC State. It is an interactive "
                            "platform for exploring IPL matches through real-time analytics and predictive modeling, bringing "
                            "ball-by-ball data to life with win probability tracking, momentum shifts, and team performance "
                            "insights. The goal is to combine intuitive visualizations with rigorous modeling to better "
                            "understand how matches evolve and what drives outcomes in T20 cricket."
                        ),
                        style={
                            "maxWidth": "980px",
                            "margin": "10px 0 0",
                            "fontSize": "16px",
                            "lineHeight": "1.5",
                            "color": "#34495e",
                        },
                    ),
                ],
                style={"padding": "16px 0"},
            ),
            html.Div(
                [
                    _landing_list("Live Matches", sections["live"], link_items = True),
                    _landing_list("Recent Matches", sections["recent"], link_items = True, show_time = False),
                    _landing_list("Upcoming Matches", sections["upcoming"], link_items = False, include_starts_in = True),
                ],
                style={"display": "grid", "gap": "12px"},
            ),
        ],
        style={"padding": "18px 24px 28px", "boxSizing": "border-box"},
    )


def _card(
    children,
    *,
    body_class_name: str = "p-3",
    class_name: str | None = "h-100",
    color: str | None = None,
    inverse: bool | None = None,
):
    return dbc.Card(
        dbc.CardBody(children, class_name = body_class_name),
        class_name = class_name,
        color = color,
        inverse = inverse,
    )


def _summary_card(label: str, value: str):
    return _summary_card_with_style(label, value)


def _summary_card_with_style(
    label: str,
    value: str,
    *,
    color: str | None = None,
    inverse: bool | None = None,
):
    return _card(
        children=[
            html.Div(label, style={"fontSize": "12px", "fontWeight": 600}),
            html.Div(
                value,
                style={
                    "fontSize": CARD_VALUE_FONT_SIZE,
                    "fontWeight": 600,
                    "marginTop": "6px",
                },
            ),
        ],
        body_class_name = "p-3",
        color = color,
        inverse = inverse,
    )


def _score_summary_card(
    summary: dict,
    status_text: str | None = None,
    *,
    card_color: str = "primary",
    muted_team: str | None = None,
):
    innings1_score = f"{summary['innings1']['runs']}/{summary['innings1']['wickets']} ({summary['innings1']['overs']})"
    innings2_score = f"{summary['innings2']['runs']}/{summary['innings2']['wickets']} ({summary['innings2']['overs']})"
    muted_color = "#d9dee3"
    season = _landing_row_season(summary.get("date"), summary.get("scheduled_start_ts"))

    def _team_line(team: str, score: str) -> html.Div:
        is_muted = bool(muted_team) and str(team).strip() == str(muted_team).strip()
        text_color = muted_color if is_muted else "#ffffff"
        score_main = str(score)
        score_overs = ""
        if " (" in score_main and score_main.endswith(")"):
            split_at = score_main.rfind(" (")
            if split_at > 0:
                score_overs = score_main[split_at:]
                score_main = score_main[:split_at]
        return html.Div(
            children=[
                _team_text_with_logo(
                    team,
                    html.A(
                        team,
                        href = _team_analysis_href(team, season),
                        className = "team-analysis-link-plain score-team-link",
                        style = {
                            "fontWeight": 700,
                            "fontSize": "18px",
                            "color": text_color,
                        },
                    ),
                    "team-logo-inline score-team-logo-inline",
                ),
                html.Span(
                    [
                        html.Span(score_main, style={"fontWeight": 700}),
                        html.Span(score_overs, style={"fontWeight": 400}),
                    ],
                    style={"fontSize": "20px", "color": text_color},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "gap": "12px",
            },
        )

    return _card(
        children=[
            html.Div(
                "Score",
                style={"fontSize": "12px", "fontWeight": 600, "marginBottom": "8px"},
            ),
            _team_line(str(summary["team1"]), innings1_score),
            html.Div(style={"height": "8px"}),
            _team_line(str(summary["team2"]), innings2_score),
            html.Div(
                "" if status_text is None else str(status_text),
                style={
                    "fontSize": "15px",
                    "fontWeight": 400,
                    "marginTop": "10px",
                    "lineHeight": "1.2",
                },
            ),
        ],
        body_class_name = "p-3",
        color = card_color,
        inverse = True,
    )


def _info_summary_card(summary: dict):
    stage = "" if summary.get("stage") is None else str(summary.get("stage")).strip()
    match_no = "" if summary.get("event_match_no") is None else str(summary.get("event_match_no")).strip()
    playoff_raw = pd.to_numeric(pd.Series([summary.get("playoff_match")]), errors = "coerce").iloc[0]
    is_playoff = pd.notna(playoff_raw) and int(playoff_raw) == 1

    if is_playoff and stage and stage.lower() != "unknown":
        match_descriptor = stage
    elif match_no and match_no.lower() != "unknown":
        match_descriptor = f"Match {match_no}"
    elif stage and stage.lower() != "unknown":
        match_descriptor = stage
    else:
        match_descriptor = "Match"

    scheduled_ts = summary.get("scheduled_start_ts")
    date_text = _format_date(summary.get("date"))
    time_text = "-"
    if scheduled_ts is not None:
        dt_utc = _parse_dt_utc(str(scheduled_ts))
        if dt_utc is not None:
            dt_local = dt_utc.astimezone(_system_timezone())
            date_text = dt_local.strftime("%m/%d/%Y")
            time_text = _format_local_time(dt_local)
    match_date_time_line = f"{match_descriptor} | {date_text} | {time_text}" if time_text != "-" else f"{match_descriptor} | {date_text}"
    toss_winner = summary.get("toss_winner")
    toss_decision = summary.get("toss_decision")
    toss_line = "-"
    if toss_winner is not None and toss_decision is not None:
        winner_text = str(toss_winner).strip()
        decision_text = str(toss_decision).strip().lower()
        if winner_text and decision_text:
            toss_display = "field" if decision_text in {"bowl", "field"} else "bat"
            toss_line = f"{winner_text} chose to {toss_display}"
    line_style = {
        "fontSize": CARD_VALUE_FONT_SIZE,
        "marginTop": "6px",
        "fontWeight": 400,
    }

    return _card(
        children=[
            html.Div("Info", style={"fontSize": "12px", "fontWeight": 600}),
            html.Div(match_date_time_line, style=line_style),
            html.Div(str(summary.get("venue", "-")), style=line_style),
            html.Div(toss_line, style=line_style),
        ],
        body_class_name = "p-3",
        color = "primary",
        inverse = True,
    )


def _format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    if "Strike Rate" in formatted.columns:
        formatted["Strike Rate"] = pd.to_numeric(formatted["Strike Rate"], errors = "coerce").round(2)
    if "Economy" in formatted.columns:
        formatted["Economy"] = pd.to_numeric(formatted["Economy"], errors = "coerce").round(2)
    return formatted


def _data_table(
    df: pd.DataFrame,
    *,
    annotate_not_out: bool = False,
    center_numeric_columns: bool = False,
) -> dash_table.DataTable:
    formatted = _format_numeric_columns(df)
    style_data_conditional = []
    style_header_conditional = []
    if center_numeric_columns:
        numeric_columns = [
            str(col)
            for col in formatted.columns
            if pd.api.types.is_numeric_dtype(formatted[col])
        ]
        for col in numeric_columns:
            style_data_conditional.append(
                {
                    "if": {"column_id": col},
                    "textAlign": "center",
                }
            )
            style_header_conditional.append(
                {
                    "if": {"column_id": col},
                    "textAlign": "center",
                }
            )

    if annotate_not_out and {"Status", "Batter", "Runs"}.issubset(formatted.columns):
        not_out_mask = (
            formatted["Status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("not out")
        )

        if not_out_mask.any():
            # Allow visual asterisk formatting (e.g., 79*) without mutating source stats.
            formatted["Runs"] = formatted["Runs"].astype(object)

            def _format_runs_with_asterisk(value) -> str:
                num = pd.to_numeric(pd.Series([value]), errors = "coerce").iloc[0]
                if pd.notna(num):
                    num_float = float(num)
                    if num_float.is_integer():
                        return f"{int(num_float)}*"
                    return f"{num_float:g}*"
                text = str(value).strip()
                return f"{text}*" if text else "*"

            formatted.loc[not_out_mask, "Runs"] = formatted.loc[not_out_mask, "Runs"].apply(_format_runs_with_asterisk)
            style_data_conditional.append(
                {
                    "if": {"column_id": "Batter", "filter_query": '{Status} = "not out"'},
                    "fontWeight": 700,
                }
            )

    return dash_table.DataTable(
        data = formatted.to_dict("records"),
        columns = [{"name": col, "id": col} for col in formatted.columns],
        page_action = "none",
        style_table = {"overflowX": "auto", "border": "1px solid #dee2e6", "borderRadius": "8px"},
        style_header = {
            "fontWeight": 700,
            "fontSize": "14px",
            "borderBottom": "1px solid #dee2e6",
        },
        style_cell = {
            "fontSize": "14px",
            "padding": "6px 8px",
            "whiteSpace": "normal",
            "height": "auto",
            "textAlign": "left",
            "fontFamily": APP_FONT_STACK,
            "borderLeft": "1px solid #dee2e6",
            "borderRight": "1px solid #dee2e6",
            "borderBottom": "1px solid #dee2e6",
        },
        style_data_conditional = style_data_conditional,
        style_header_conditional = style_header_conditional,
    )


def _flatly_table(df: pd.DataFrame):
    if df.empty:
        return html.Div("No data available.", style={"padding": "8px 0"})
    return dbc.Table.from_dataframe(
        df,
        striped = True,
        bordered = False,
        hover = True,
        size = "sm",
        responsive = True,
        class_name = "mb-0",
    )


def _team_ranking_table(df: pd.DataFrame, *, season: int | None = None):
    if df.empty:
        return html.Div("No data available.", style={"padding": "8px 0"})

    numeric_columns = {"Rank", "Matches", "Wins", "Losses", "No Result", "Points"}

    def _display_value(value):
        if pd.isna(value):
            return ""
        num = pd.to_numeric(pd.Series([value]), errors = "coerce").iloc[0]
        if pd.notna(num):
            num_f = float(num)
            return str(int(num_f)) if num_f.is_integer() else f"{num_f:.2f}"
        return str(value)

    columns = [str(col) for col in df.columns]
    header_cells = []
    for col in columns:
        th_style = {
            "fontSize": "16px",
            "fontWeight": 700,
            "fontFamily": APP_FONT_STACK,
            "verticalAlign": "middle",
        }
        if col in numeric_columns:
            th_style["textAlign"] = "center"
        if col == "Rank":
            th_style.update(
                {
                    "width": "64px",
                    "minWidth": "64px",
                    "maxWidth": "64px",
                    "whiteSpace": "nowrap",
                }
            )
        header_cells.append(html.Th(col, style = th_style))

    body_rows = []
    for _, row in df.iterrows():
        row_cells = []
        for col in columns:
            td_style = {
                "fontSize": "16px",
                "fontFamily": APP_FONT_STACK,
                "padding": "6px 8px",
                "verticalAlign": "middle",
            }
            if col in numeric_columns:
                td_style["textAlign"] = "center"
            if col in {"Rank", "Points"}:
                td_style["fontWeight"] = 700
            if col == "Rank":
                td_style.update(
                    {
                        "width": "64px",
                        "minWidth": "64px",
                        "maxWidth": "64px",
                        "whiteSpace": "nowrap",
                    }
                )
            cell_value = _display_value(row[col])
            if col == "Team" and cell_value:
                cell_value = html.A(
                    _team_text_with_logo(
                        cell_value,
                        html.Span(cell_value),
                        "team-logo-inline leaderboard-team-logo-inline",
                    ),
                    href = _team_analysis_href(cell_value, season),
                    className = "team-ranking-link",
                )
            row_cells.append(html.Td(cell_value, style = td_style))
        body_rows.append(html.Tr(row_cells))

    return html.Div(
        dbc.Table(
            [html.Thead(html.Tr(header_cells)), html.Tbody(body_rows)],
            striped = True,
            bordered = False,
            hover = True,
            responsive = True,
            class_name = "mb-0",
        ),
        style = {"overflowX": "auto"},
    )


def _leaderboard_data_table(
    df: pd.DataFrame,
    *,
    page_size: int = 15,
    paginated: bool = True,
) -> dash_table.DataTable:
    style_cell_conditional = []
    style_header_conditional = []
    text_columns = {"Batter", "Bowler", "Team"}
    centered_columns = [str(col) for col in df.columns if str(col) not in text_columns]
    for col in centered_columns:
        style_cell_conditional.append(
            {
                "if": {"column_id": col},
                "textAlign": "center",
            }
        )
        style_header_conditional.append(
            {
                "if": {"column_id": col},
                "textAlign": "center",
            }
        )
    if "Rank" in df.columns:
        style_cell_conditional.append(
            {
                "if": {"column_id": "Rank"},
                "minWidth": "64px",
                "width": "64px",
                "maxWidth": "64px",
                "whiteSpace": "nowrap",
                "textAlign": "center",
            }
        )
        style_header_conditional.append(
            {
                "if": {"column_id": "Rank"},
                "minWidth": "64px",
                "width": "64px",
                "maxWidth": "64px",
                "whiteSpace": "nowrap",
                "textAlign": "center",
            }
        )
    return dash_table.DataTable(
        data = df.to_dict("records"),
        columns = [{"name": col, "id": col} for col in df.columns],
        page_action = "native" if paginated else "none",
        page_current = 0,
        page_size = int(page_size),
        style_table = {"overflowX": "auto", "border": "1px solid #dee2e6", "borderRadius": "8px"},
        style_header = {
            "fontWeight": 700,
            "fontSize": "14px",
            "borderLeft": "1px solid #dee2e6",
            "borderRight": "1px solid #dee2e6",
            "borderBottom": "1px solid #dee2e6",
        },
        style_cell = {
            "fontSize": "14px",
            "padding": "6px 8px",
            "whiteSpace": "normal",
            "height": "auto",
            "textAlign": "left",
            "fontFamily": APP_FONT_STACK,
            "borderLeft": "1px solid #dee2e6",
            "borderRight": "1px solid #dee2e6",
            "borderBottom": "1px solid #dee2e6",
        },
        style_cell_conditional = style_cell_conditional,
        style_header_conditional = style_header_conditional,
        css = [
            {
                "selector": ".dash-table-container .previous-next-container, .previous-next-container",
                "rule": "display: flex !important; justify-content: center !important; align-items: center !important; float: none !important; width: max-content !important; margin: 8px auto 0 auto !important; gap: 8px;",
            },
            {
                "selector": ".dash-table-container .previous-next-container .page-number, .previous-next-container .page-number",
                "rule": "margin: 0 !important;",
            },
            {
                "selector": ".previous-next-container .page-number .current-page-container input.current-page",
                "rule": "text-align: center;",
            },
        ],
    )


def _leaderboard_table_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for hidden_col in ["Player Key"]:
        if hidden_col in out.columns:
            out = out.drop(columns=[hidden_col])
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(2)
    return out


def _leaderboard_table_card(
    title: str,
    df: pd.DataFrame,
    *,
    paginated: bool = False,
    season: int | None = None,
    color: str | None = None,
    inverse: bool | None = None,
):
    table_df = _leaderboard_table_df(df)
    if table_df.empty:
        table_component = html.Div("No data available.", style={"padding": "8px 0"})
    elif paginated:
        table_component = _leaderboard_data_table(table_df, page_size = 15, paginated = True)
    elif title == "Team Ranking":
        table_component = _team_ranking_table(table_df, season = season)
    else:
        table_component = _flatly_table(table_df)

    return _card(
        [
            html.Div(title, style={"fontWeight": 700, "fontSize": "18px", "marginBottom": "8px"}),
            table_component,
        ],
        body_class_name = "p-3",
        color = color,
        inverse = inverse,
    )


def _leaderboard_graph_card(figure):
    figure.update_layout(
        width = None,
        height = 320,
        margin = {"l": 40, "r": 20, "t": 50, "b": 35},
    )
    return _card(
        [
            dcc.Graph(
                figure = figure,
                config = {"displayModeBar": False},
            ),
        ],
        body_class_name = "p-2",
        class_name = "h-100",
    )


def _leaderboard_placeholder_card(title: str):
    return _card(
        [
            html.Div(title, style={"fontWeight": 700, "fontSize": "18px", "marginBottom": "8px"}),
            html.Div(style={"minHeight": "160px"}),
        ],
        body_class_name = "p-3",
    )


OVERALL_STRENGTH_METRICS = {
    "overall_strength": {
        "label": "Overall Strength",
        "color": PLOTLY_COLORS.get("leaderboard_impact", "#3498db"),
        "description": "A weighted average of batting and bowling strength compared with team seasons from across IPL history.",
    },
    "top_order_batting": {
        "label": "Top-Order Batting",
        "color": PLOTLY_COLORS.get("innings_2", "#FF7F0E"),
        "description": "The impact generated by each team's batters in positions 1-3 compared with past IPL team seasons.",
    },
    "middle_order_batting": {
        "label": "Middle-Order Batting",
        "color": PLOTLY_COLORS.get("innings_2", "#FF7F0E"),
        "description": "The impact generated by each team's batters in positions 4-6 compared with past IPL team seasons.",
    },
    "finisher_batting": {
        "label": "Finisher Batting",
        "color": PLOTLY_COLORS.get("innings_2", "#FF7F0E"),
        "description": "The impact generated by each team's batters in position 7 or lower compared with past IPL team seasons.",
    },
    "wicket_taking_bowling": {
        "label": "Wicket-Taking",
        "color": PLOTLY_COLORS.get("innings_1", "#1F77B4"),
        "description": "How much bowling impact each team generated through wickets against the all-season IPL benchmark.",
    },
    "bowling_pressure": {
        "label": "Bowling Pressure",
        "color": PLOTLY_COLORS.get("innings_1", "#1F77B4"),
        "description": "How much non-wicket bowling impact each team created through pressure against the all-season IPL benchmark.",
    },
}


def _overall_strength_metric_options() -> list[dict[str, str]]:
    return [
        {"label": config["label"], "value": metric_key}
        for metric_key, config in OVERALL_STRENGTH_METRICS.items()
    ]


def _overall_strength_description(selected_metric: str | None):
    metric_key = selected_metric if selected_metric in OVERALL_STRENGTH_METRICS else "overall_strength"
    metric_config = OVERALL_STRENGTH_METRICS[metric_key]
    return html.Div(
        html.Div(metric_config["description"]),
        style = {
            "fontSize": "13px",
            "fontWeight": 400,
            "lineHeight": "1.35",
            "color": "#6c757d",
            "borderTop": "1px solid rgba(108, 117, 125, 0.16)",
            "borderBottom": "1px solid rgba(108, 117, 125, 0.16)",
            "padding": "12px 0 18px",
            "marginTop": "18px",
        },
    )


def _overall_strength_summary_item(
    label: str,
    value: str,
    descriptor: str | None = None,
    *,
    value_font_size: str = "25px",
    logo_team: str | None = None,
) -> html.Div:
    value_node = html.Div(
        value,
        style = {
            "fontSize": value_font_size,
            "fontWeight": 700,
            "lineHeight": "1.12",
            "color": "#212529",
        },
    )
    if logo_team:
        value_node = html.Div(
            [value_node, _team_logo_img(logo_team)],
            className = "team-logo-inline overall-strength-logo-inline",
        )
    return html.Div(
        [
            html.Div(
                label,
                style = {
                    "fontSize": "12px",
                    "fontWeight": 700,
                    "lineHeight": "1.1",
                    "color": "#6B7C8F",
                    "textTransform": "uppercase",
                    "letterSpacing": 0,
                    "marginBottom": "5px",
                },
            ),
            value_node,
            *(
                [
                    html.Div(
                        descriptor,
                        style = {
                            "fontSize": "12px",
                            "fontWeight": 400,
                            "lineHeight": "1.2",
                            "marginTop": "4px",
                            "color": "#5D728A",
                        },
                    )
                ]
                if descriptor
                else []
            ),
        ],
        style = {
            "padding": "8px 0",
        },
    )


def _overall_strength_side_panel(df: pd.DataFrame, selected_metric: str | None, season: int | None = None):
    metric_key = selected_metric if selected_metric in OVERALL_STRENGTH_METRICS else "overall_strength"
    season_label = "-" if season is None else str(int(season))

    def _summary_stack(items: list[html.Div]) -> html.Div:
        return html.Div(
            items,
            style = {
                "display": "flex",
                "flexDirection": "column",
                "gap": "12px",
            },
        )

    if df.empty or metric_key not in df.columns:
        return [
            _summary_stack(
                [
                    _overall_strength_summary_item("Top Team", "-"),
                    _overall_strength_summary_item(f"League Average ({season_label})", "-", value_font_size = "27px"),
                    _overall_strength_summary_item(f"Spread ({season_label})", "-", "First-to-last gap", value_font_size = "27px"),
                ]
            ),
            _overall_strength_description(metric_key),
        ]

    values_df = df.copy()
    values_df["Team"] = values_df["Team"].fillna("").astype(str)
    values_df[metric_key] = pd.to_numeric(values_df[metric_key], errors = "coerce")
    values_df = values_df.dropna(subset = [metric_key])
    values_df = values_df.loc[values_df["Team"].str.strip().ne("")]

    if values_df.empty:
        return [
            _summary_stack(
                [
                    _overall_strength_summary_item("Top Team", "-"),
                    _overall_strength_summary_item(f"League Average ({season_label})", "-", value_font_size = "27px"),
                    _overall_strength_summary_item(f"Spread ({season_label})", "-", "First-to-last gap", value_font_size = "27px"),
                ]
            ),
            _overall_strength_description(metric_key),
        ]

    values_df[metric_key] = values_df[metric_key].clip(0, 100)
    top_row = values_df.sort_values([metric_key, "Team"], ascending = [False, True], kind = "mergesort").iloc[0]
    top_team = str(top_row.get("Team", "-"))
    league_average = int(round(float(values_df[metric_key].mean())))
    spread = int(round(float(values_df[metric_key].max() - values_df[metric_key].min())))

    return [
        _summary_stack(
            [
                _overall_strength_summary_item("Top Team", top_team, logo_team = top_team),
                _overall_strength_summary_item(f"League Average ({season_label})", str(league_average), value_font_size = "27px"),
                _overall_strength_summary_item(f"Spread ({season_label})", str(spread), "First-to-last gap", value_font_size = "27px"),
            ]
        ),
        _overall_strength_description(metric_key),
    ]


def _team_strength_percentile_rows(leaderboard: Leaderboard) -> pd.DataFrame:
    ranking = getattr(leaderboard, "team_ranking", pd.DataFrame())
    if not isinstance(ranking, pd.DataFrame) or ranking.empty or "Team" not in ranking.columns:
        return pd.DataFrame(columns=["Team", *OVERALL_STRENGTH_METRICS.keys()])

    season_metrics = _team_identity_season_metrics(leaderboard)
    try:
        benchmark_metrics = _load_team_identity_benchmarks(_team_identity_benchmark_cache_token())
    except Exception:
        benchmark_metrics = pd.DataFrame()

    rows = []
    for _, team_row in ranking.iterrows():
        team_name = str(team_row.get("Team", "") or "").strip()
        if not team_name:
            continue

        selected_metrics = pd.DataFrame()
        if isinstance(season_metrics, pd.DataFrame) and not season_metrics.empty and "team" in season_metrics.columns:
            team_key = team_name.strip().lower()
            selected_metrics = season_metrics.loc[
                season_metrics["team"].fillna("").astype(str).str.strip().str.lower().eq(team_key)
            ].copy()
        if selected_metrics.empty:
            continue

        profile_metrics = compute_profile_percentiles(selected_metrics, benchmark_metrics)
        if profile_metrics.empty or "metric_key" not in profile_metrics.columns:
            continue

        profile_metrics["metric_key"] = profile_metrics["metric_key"].fillna("").astype(str)
        profile_metrics["percentile"] = pd.to_numeric(profile_metrics.get("percentile"), errors = "coerce").fillna(0)
        metric_values = profile_metrics.set_index("metric_key")["percentile"].to_dict()
        row = {
            "Team": team_name,
            "overall_strength": _dominance_index(profile_metrics),
        }
        for metric_key in OVERALL_STRENGTH_METRICS:
            if metric_key == "overall_strength":
                continue
            row[metric_key] = float(np.clip(metric_values.get(metric_key, 0.0), 0.0, 100.0))
        rows.append(row)

    return pd.DataFrame(rows)


def _overall_strength_figure(df: pd.DataFrame, selected_metric: str | None):
    metric_key = selected_metric if selected_metric in OVERALL_STRENGTH_METRICS else "overall_strength"
    metric_config = OVERALL_STRENGTH_METRICS[metric_key]
    metric_label = metric_config["label"]
    color = metric_config["color"]
    x_max = 100

    if df.empty or metric_key not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            x = 0.5,
            y = 0.5,
            text = "No data available",
            showarrow = False,
            xref = "paper",
            yref = "paper",
            font = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY),
        )
        fig.update_xaxes(visible = False)
        fig.update_yaxes(visible = False)
    else:
        plot_df = df.copy()
        plot_df[metric_key] = pd.to_numeric(plot_df[metric_key], errors = "coerce").fillna(0).clip(0, 100)
        plot_df["Team"] = plot_df["Team"].fillna("").astype(str)
        plot_df = plot_df.loc[plot_df["Team"].str.strip().ne("")]
        plot_df = plot_df.sort_values([metric_key, "Team"], ascending = [False, True], kind = "mergesort")
        max_metric_value = float(plot_df[metric_key].max()) if not plot_df.empty else 0.0
        x_max = min(100, int(ceil(max_metric_value / 10.0) * 10 + 10))
        x_max = max(10, x_max)

        labels = plot_df[metric_key].round(0).astype(int).astype(str)
        fig = go.Figure(
            data = [
                go.Bar(
                    x = plot_df[metric_key],
                    y = plot_df["Team"],
                    orientation = "h",
                    marker = dict(color = color),
                    text = labels,
                    textposition = "outside",
                    textfont = dict(size = PLOTLY_BASE_FONT_SIZE, color = "#1F2A37", family = PLOTLY_FONT_FAMILY),
                    cliponaxis = False,
                    hovertemplate = "Team = %{y}<br>" + f"{metric_label} = " + "%{x:.0f}<extra></extra>",
                    showlegend = False,
                )
            ]
        )
        fig.update_yaxes(
            categoryorder = "array",
            categoryarray = plot_df["Team"].tolist(),
            autorange = "reversed",
            fixedrange = True,
            ticks = "",
            ticklen = 0,
            tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY),
        )

    fig.update_layout(
        title = dict(text = f"<b>{metric_label}</b>", x = 0.5, xanchor = "center", y = 0.98, yanchor = "top"),
        height = 504,
        margin = {"t": 64, "l": 160, "r": 44, "b": 48},
        transition = {"duration": 250, "easing": "cubic-in-out"},
        hoverlabel = dict(font = dict(color = "#ffffff")),
    )
    apply_plot_theme(fig)
    fig.update_xaxes(
        range = [0, x_max],
        fixedrange = True,
        showgrid = True,
        gridcolor = "rgba(31, 42, 55, 0.08)",
        gridwidth = 1,
        zeroline = False,
        tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY),
    )
    fig.update_yaxes(
        showgrid = False,
        zeroline = False,
        tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY),
    )
    return fig


def _build_leaderboard_overall_strength_tab(leaderboard: Leaderboard) -> html.Div:
    strength_df = _team_strength_percentile_rows(leaderboard)
    default_metric = "overall_strength"
    return _card(
        [
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id = "overall-strength-chart",
                            figure = _overall_strength_figure(strength_df, default_metric),
                            config = {"displayModeBar": False},
                        ),
                        style = {"flex": "3 1 0", "minWidth": 0},
                    ),
                    html.Div(
                        [
                                    html.Div(
                                dcc.Dropdown(
                                    id = "overall-strength-metric-dropdown",
                                    options = _overall_strength_metric_options(),
                                    value = default_metric,
                                    clearable = False,
                                    maxHeight = 320,
                                ),
                                style = {
                                    "flex": "0 0 82px",
                                    "display": "flex",
                                    "alignItems": "flex-start",
                                    "justifyContent": "center",
                                },
                            ),
                                    html.Div(
                                html.Div(
                                    _overall_strength_side_panel(
                                        strength_df,
                                        default_metric,
                                        getattr(leaderboard, "season", None),
                                    ),
                                    id = "overall-strength-metric-description",
                                    style = {
                                        "width": "100%",
                                        "maxWidth": "340px",
                                    },
                                ),
                                style = {
                                    "flex": "1 1 auto",
                                    "paddingTop": "8px",
                                    "display": "flex",
                                    "justifyContent": "center",
                                    "alignItems": "flex-start",
                                },
                            ),
                        ],
                        style = {
                            "flex": "1 1 0",
                            "minWidth": "260px",
                            "height": "504px",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                ],
                style = {"display": "flex", "gap": "16px", "alignItems": "stretch"},
            ),
        ],
        body_class_name = "p-3",
    )


def _build_leaderboard_batting_tab(leaderboard: Leaderboard) -> html.Div:
    return html.Div(
        children = [
            _leaderboard_table_card("Batting Stats", leaderboard.batter_stats, paginated = True),
            html.Div(
                children = [
                    html.Div(
                        _leaderboard_graph_card(leaderboard.plot_batting_average()),
                        style = {"flex": "1 1 320px", "minWidth": 0},
                    ),
                    html.Div(
                        _leaderboard_graph_card(leaderboard.plot_strike_rate()),
                        style = {"flex": "1 1 320px", "minWidth": 0},
                    ),
                    html.Div(
                        _leaderboard_graph_card(leaderboard.plot_total_boundaries_stacked()),
                        style = {"flex": "1 1 320px", "minWidth": 0},
                    ),
                ],
                style = {"display": "flex", "gap": "12px", "marginTop": "12px", "flexWrap": "wrap"},
            ),
        ]
    )


def _build_leaderboard_bowling_tab(leaderboard: Leaderboard) -> html.Div:
    return html.Div(
        children = [
            _leaderboard_table_card("Bowling Stats", leaderboard.bowler_stats, paginated = True),
            html.Div(
                children = [
                    html.Div(
                        _leaderboard_graph_card(leaderboard.plot_bowling_average(ascending = True)),
                        style = {"flex": "1 1 320px", "minWidth": 0},
                    ),
                    html.Div(
                        _leaderboard_graph_card(leaderboard.plot_economy(ascending = True)),
                        style = {"flex": "1 1 320px", "minWidth": 0},
                    ),
                    html.Div(
                        _leaderboard_graph_card(leaderboard.plot_dot_ball_pct(descending = True)),
                        style = {"flex": "1 1 320px", "minWidth": 0},
                    ),
                ],
                style = {"display": "flex", "gap": "12px", "marginTop": "12px", "flexWrap": "wrap"},
            ),
        ]
    )


DOMINANCE_BATTING_METRICS = ["top_order_batting", "middle_order_batting", "finisher_batting"]
DOMINANCE_BOWLING_METRICS = ["wicket_taking_bowling", "bowling_pressure"]


def _dominance_index(profile_metrics: pd.DataFrame) -> float:
    if profile_metrics.empty or "metric_key" not in profile_metrics.columns:
        return 0.0

    metrics = profile_metrics.copy()
    metrics["metric_key"] = metrics["metric_key"].fillna("").astype(str)
    value_col = "percentile" if "percentile" in metrics.columns else "raw_value"
    metrics[value_col] = pd.to_numeric(metrics.get(value_col), errors = "coerce").fillna(0)
    values = metrics.set_index("metric_key")[value_col].to_dict()

    def _score(metric_keys: list[str]) -> float:
        metric_values = [
            float(np.clip(values.get(metric_key, 0.0), 0.0, 100.0))
            for metric_key in metric_keys
        ]
        return float(np.mean(metric_values)) if metric_values else 0.0

    batting_score = _score(DOMINANCE_BATTING_METRICS)
    bowling_score = _score(DOMINANCE_BOWLING_METRICS)
    return float(np.clip(0.5 * batting_score + 0.5 * bowling_score, 0.0, 100.0))


def _build_season_leader_cards(leaderboard: Leaderboard) -> html.Div:
    batter_stats = getattr(leaderboard, "batter_stats", pd.DataFrame())
    bowler_stats = getattr(leaderboard, "bowler_stats", pd.DataFrame())
    impact_stats = getattr(leaderboard, "player_impact_stats", pd.DataFrame())
    matches = getattr(leaderboard, "matches", pd.DataFrame())
    impact_balls = getattr(leaderboard, "_impact_balls", pd.DataFrame())

    dominant_team_name = "-"
    dominant_team_value = "-"
    dominant_team_raw = 0.0
    ranking = getattr(leaderboard, "team_ranking", pd.DataFrame())
    if (
        isinstance(ranking, pd.DataFrame)
        and not ranking.empty
        and isinstance(matches, pd.DataFrame)
        and isinstance(impact_balls, pd.DataFrame)
    ):
        season_metrics = _team_identity_season_metrics(leaderboard)
        try:
            benchmark_metrics = _load_team_identity_benchmarks(_team_identity_benchmark_cache_token())
        except Exception:
            benchmark_metrics = pd.DataFrame()
        team_rows = []
        for _, team_row in ranking.iterrows():
            team_name = str(team_row.get("Team", "") or "").strip()
            if not team_name:
                continue
            if isinstance(season_metrics, pd.DataFrame) and not season_metrics.empty and "team" in season_metrics.columns:
                team_key = team_name.strip().lower()
                selected_metrics = season_metrics.loc[
                    season_metrics["team"].fillna("").astype(str).str.strip().str.lower().eq(team_key)
                ].copy()
                profile_metrics = compute_profile_percentiles(selected_metrics, benchmark_metrics)
            else:
                profile_metrics = compute_team_profile_metrics(impact_balls, matches, team_name)
            if profile_metrics.empty:
                continue
            team_rows.append(
                {
                    "Team": team_name,
                    "Dominance Index": _dominance_index(profile_metrics),
                }
            )
        if team_rows:
            dominant_row = (
                pd.DataFrame(team_rows)
                .sort_values(["Dominance Index", "Team"], ascending = [False, True], kind = "mergesort")
                .iloc[0]
            )
            dominant_team_name = str(dominant_row.get("Team", "-"))
            dominant_team_raw = float(dominant_row.get("Dominance Index", 0))
            dominant_team_value = f"{dominant_team_raw:.2f}"

    top_batter_name = "-"
    top_batter_value = "-"
    top_batter_team = "-"
    if isinstance(batter_stats, pd.DataFrame) and not batter_stats.empty:
        batters = batter_stats.copy()
        batters["Runs"] = pd.to_numeric(batters.get("Runs"), errors = "coerce").fillna(0)
        batters["Strike Rate"] = pd.to_numeric(batters.get("Strike Rate"), errors = "coerce").fillna(0)
        row = batters.sort_values(["Runs", "Strike Rate", "Batter"], ascending = [False, False, True], kind = "mergesort").iloc[0]
        top_batter_name = str(row.get("Batter", "-"))
        top_batter_value = str(int(row.get("Runs", 0)))
        top_batter_team = str(row.get("Team", "-") or "-")

    top_bowler_name = "-"
    top_bowler_value = "-"
    top_bowler_team = "-"
    if isinstance(bowler_stats, pd.DataFrame) and not bowler_stats.empty:
        bowlers = bowler_stats.copy()
        bowlers["Wickets"] = pd.to_numeric(bowlers.get("Wickets"), errors = "coerce").fillna(0)
        bowlers["Economy"] = pd.to_numeric(bowlers.get("Economy"), errors = "coerce").fillna(0)
        row = bowlers.sort_values(["Wickets", "Economy", "Bowler"], ascending = [False, True, True], kind = "mergesort").iloc[0]
        top_bowler_name = str(row.get("Bowler", "-"))
        top_bowler_value = str(int(row.get("Wickets", 0)))
        top_bowler_team = str(row.get("Team", "-") or "-")

    top_impact_name = "-"
    top_impact_value = "-"
    top_impact_team = "-"
    top_impact_raw = 0.0
    if isinstance(impact_stats, pd.DataFrame) and not impact_stats.empty:
        impacts = impact_stats.copy()
        impacts["Total Impact"] = pd.to_numeric(impacts.get("Total Impact"), errors = "coerce").fillna(0)
        row = impacts.sort_values(["Total Impact", "Player"], ascending = [False, True], kind = "mergesort").iloc[0]
        top_impact_name = str(row.get("Player", "-"))
        top_impact_team = str(row.get("Team", "-") or "-")
        top_impact_raw = float(row.get("Total Impact", 0))
        top_impact_value = _format_signed_impact(top_impact_raw)

    cards = [
        _team_analysis_leader_item(
            "Most Complete Team",
            dominant_team_name,
            dominant_team_value,
            subtitle = "Averaged Batting and Bowling Strength",
            color = "#ffffff",
            name_font_size = "19px",
            value_font_size = "24px",
            min_height = "72px",
            card_color = "info",
            inverse = True,
            value_subtitle = "Overall Strength",
            logo_team = dominant_team_name,
        ),
        _team_analysis_leader_item(
            "Top Run-Scorer",
            top_batter_name,
            top_batter_value,
            subtitle = top_batter_team,
            color = PLOTLY_COLORS.get("innings_2", "#FF7F0E"),
            value_subtitle = "Runs",
        ),
        _team_analysis_leader_item(
            "Top Wicket-Taker",
            top_bowler_name,
            top_bowler_value,
            subtitle = top_bowler_team,
            color = PLOTLY_COLORS.get("leaderboard_secondary", "#e74c3c"),
            value_subtitle = "Wickets",
        ),
        _team_analysis_leader_item(
            "Most Impactful Player",
            top_impact_name,
            top_impact_value,
            subtitle = top_impact_team,
            color = _team_identity_delta_color(top_impact_raw),
            value_subtitle = "Total Impact",
        ),
    ]
    card_flex_values = ["1 1 0", "1 1 0", "1 1 0", "1 1 0"]
    return html.Div(
        [
            html.Div(card, style = {"flex": card_flex, "minWidth": 0})
            for card, card_flex in zip(cards, card_flex_values, strict = True)
        ],
        style = {"display": "flex", "flexDirection": "column", "gap": "12px", "height": "100%"},
    )


def _build_season_leaderboard_dashboard(leaderboard: Leaderboard) -> html.Div:
    return html.Div(
        children = [
            html.Div(
                children = [
                    html.Div(
                        _leaderboard_table_card(
                            "Team Ranking",
                            leaderboard.team_ranking,
                            season = getattr(leaderboard, "season", None),
                        ),
                        style = {"flex": "3 1 0", "minWidth": "520px"},
                    ),
                    html.Div(
                        _build_season_leader_cards(leaderboard),
                        style = {"flex": "2 1 0", "minWidth": "360px"},
                    ),
                ],
                style = {"display": "flex", "gap": "12px", "alignItems": "stretch", "flexWrap": "wrap"},
            ),
            html.Div(
                children = [
                    dcc.Tabs(
                        id = "season-leaderboard-tabs",
                        value = "overall-strength",
                        children = [
                            dcc.Tab(label = "Overall Strength", value = "overall-strength"),
                            dcc.Tab(label = "Batting Stats", value = "batting"),
                            dcc.Tab(label = "Bowling Stats", value = "bowling"),
                        ],
                    ),
                    html.Div(id = "season-leaderboard-tab-content", style = {"paddingTop": "8px"}),
                ],
                style = {"marginTop": "12px"},
            ),
        ]
    )


def _team_analysis_team_options_from_leaderboard(leaderboard: Leaderboard) -> list[dict[str, str]]:
    teams: list[str] = []
    ranking = getattr(leaderboard, "team_ranking", pd.DataFrame())
    if isinstance(ranking, pd.DataFrame) and not ranking.empty and "Team" in ranking.columns:
        teams = ranking["Team"].dropna().astype(str).str.strip().tolist()

    if not teams:
        matches = getattr(leaderboard, "matches", pd.DataFrame())
        if isinstance(matches, pd.DataFrame) and not matches.empty:
            team_values = []
            for col in ["bat_first", "bowl_first"]:
                if col in matches.columns:
                    team_values.extend(matches[col].dropna().astype(str).str.strip().tolist())
            teams = sorted({team for team in team_values if team})

    return [{"label": team, "value": team} for team in teams if team]


def _load_team_analysis_leaderboard(season: int) -> Leaderboard:
    cache_token = _leaderboard_cache_token(int(season))
    return load_leaderboard(int(season), cache_token)


def _team_identity_benchmark_seasons() -> list[int]:
    seasons = []
    for option in get_finalized_season_options():
        value = option.get("value") if isinstance(option, dict) else option
        seasons.append(value)
    return benchmark_seasons_2008_2025(seasons)


def _team_identity_benchmark_cache_token() -> int:
    seasons = _team_identity_benchmark_seasons()
    if not seasons:
        return 0
    placeholders = ",".join(["?"] * len(seasons))
    query = f"""
        SELECT COUNT(*), MAX(updated_at)
        FROM team_profile_metrics
        WHERE season IN ({placeholders})
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(query, [int(season) for season in seasons]).fetchone()
    except Exception:
        return 0
    count = 0 if row is None or row[0] is None else int(row[0])
    max_updated = "" if row is None or row[1] is None else str(row[1])
    return abs(hash(f"{','.join(str(season) for season in seasons)}:{count}:{max_updated}"))


@lru_cache(maxsize=8)
def _load_team_identity_benchmarks(cache_token: int = 0) -> pd.DataFrame:
    _ = cache_token
    return load_team_profile_metrics(db_path=DB_PATH, seasons=_team_identity_benchmark_seasons())


def _team_identity_season_metrics(leaderboard: Leaderboard) -> pd.DataFrame:
    season = getattr(leaderboard, "season", None)
    if season is None:
        return pd.DataFrame()
    season_int = int(season)

    try:
        cached = load_team_profile_metrics(db_path=DB_PATH, season=season_int)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
    except Exception:
        pass

    try:
        refresh_team_profile_metrics_for_season(season=season_int, db_path=DB_PATH, force=False)
        cached = load_team_profile_metrics(db_path=DB_PATH, season=season_int)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
    except Exception:
        pass

    matches = getattr(leaderboard, "matches", pd.DataFrame())
    impact_balls = getattr(leaderboard, "_impact_balls", pd.DataFrame())

    live_rows: list[pd.DataFrame] = []
    if isinstance(matches, pd.DataFrame) and isinstance(impact_balls, pd.DataFrame) and (not matches.empty):
        for option in _team_analysis_team_options_from_leaderboard(leaderboard):
            team_name = str(option.get("value", "")).strip()
            if not team_name:
                continue
            team_metrics = compute_team_profile_metrics(impact_balls, matches, team_name)
            if team_metrics.empty:
                continue
            metrics = team_metrics.copy()
            metrics["season"] = season_int
            metrics["team"] = team_name
            live_rows.append(metrics)
    if live_rows:
        return pd.concat(live_rows, ignore_index=True)

    try:
        return load_team_profile_metrics(db_path=DB_PATH, season=season_int)
    except Exception:
        return pd.DataFrame()


def _team_identity_selected_metrics(leaderboard: Leaderboard, team: str) -> pd.DataFrame:
    season_metrics = _team_identity_season_metrics(leaderboard)
    if season_metrics.empty or "team" not in season_metrics.columns:
        return pd.DataFrame()
    team_key = str(team or "").strip().lower()
    selected = season_metrics.loc[
        season_metrics["team"].fillna("").astype(str).str.strip().str.lower().eq(team_key)
    ].copy()
    return selected.reset_index(drop=True)


def _team_analysis_options_and_default(season: int | None, current_team: str | None = None):
    if season is None:
        return [], None

    try:
        leaderboard = _load_team_analysis_leaderboard(int(season))
    except Exception:
        return [], None

    options = _team_analysis_team_options_from_leaderboard(leaderboard)
    valid = {option["value"] for option in options}
    if current_team in valid:
        return options, current_team
    if options:
        return options, options[0]["value"]
    return options, None


def _team_analysis_graph_card(
    title: str,
    figure,
    *,
    show_card_title: bool = True,
    height: int = 340,
    graph_top_margin: int | None = None,
):
    top_margin = 52 if show_card_title else PLOTLY_HEADER_MARGIN_TOP
    if graph_top_margin is not None:
        top_margin = int(graph_top_margin)
    left_margin = 48 if show_card_title else PLOTLY_HEADER_MARGIN_LEFT
    right_margin = 20 if show_card_title else PLOTLY_HEADER_MARGIN_RIGHT
    figure.update_layout(
        width = None,
        height = height,
        margin = {"l": left_margin, "r": right_margin, "t": top_margin, "b": 42},
    )
    children = []
    if show_card_title:
        children.append(html.Div(title, style={"fontWeight": 700, "fontSize": "18px", "marginBottom": "4px"}))
    children.append(dcc.Graph(figure = figure, config = {"displayModeBar": False}))
    return _card(
        children,
        body_class_name = "p-2",
        class_name = "h-100",
    )


def _team_analysis_summary_strip(
    leaderboard: Leaderboard,
    team: str,
    phase_averages: dict[str, float],
) -> html.Div:
    matches = getattr(leaderboard, "matches", pd.DataFrame())
    impact_balls = getattr(leaderboard, "_impact_balls", pd.DataFrame())
    record = team_record(matches, team)
    top_player = top_impact_player(impact_balls, matches, team)
    top_player_text = "-" if top_player is None else str(top_player.get("Player", "-"))
    best_phase = best_phase_name(phase_averages)

    cards = [
        _summary_card("Matches played", str(record["matches"])),
        _summary_card("Wins / losses", f"{record['wins']} / {record['losses']}"),
        _summary_card("Top impact player", top_player_text),
        _summary_card("Best phase", best_phase),
    ]
    return html.Div(
        children=[html.Div(card, style={"flex": "1 1 180px", "minWidth": 0}) for card in cards],
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "stretch"},
    )


def _team_analysis_header(leaderboard: Leaderboard, team: str, season: int | None) -> html.Div:
    matches = getattr(leaderboard, "matches", pd.DataFrame())
    ranking = getattr(leaderboard, "team_ranking", pd.DataFrame())
    record = team_record(matches, team)
    rank = team_standings_rank(ranking, team)
    season_text = "-" if season is None else str(int(season))
    rank_text = "-" if rank is None else str(int(rank))
    return html.Div(
        [
            _team_logo_img(team, "team-logo team-analysis-header-logo"),
            html.Div(
                [
                    html.H2(team, style = {"margin": "0 0 8px", "fontWeight": 400}),
                    html.Div(
                        f"Season: {season_text}",
                        style = {"fontSize": "15px", "fontWeight": 600, "color": "#34495e", "lineHeight": "1.35"},
                    ),
                    html.Div(
                        f"Wins: {record['wins']}, Losses: {record['losses']}, No Result: {record['no_results']}",
                        style = {"fontSize": "15px", "fontWeight": 400, "color": "#34495e", "lineHeight": "1.35"},
                    ),
                    html.Div(
                        f"Season Rank: {rank_text}",
                        style = {"fontSize": "15px", "fontWeight": 400, "color": "#34495e", "lineHeight": "1.35"},
                    ),
                ],
                style = {"minWidth": 0},
            ),
        ],
        className = "team-analysis-header-with-logo",
    )


TEAM_IDENTITY_CONTEXT_LABELS = {
    "Top-order batting impact": "Positions 1-3",
    "Middle-order batting impact": "Positions 4-6",
    "Finisher batting impact": "Positions 7+",
    "Powerplay bowling impact": "Overs 1-6",
    "Middle-overs bowling impact": "Overs 7-15",
    "Death-overs bowling impact": "Overs 16-20",
}

TEAM_ANALYSIS_PIE_COLORS = [
    PLOTLY_COLORS.get("leaderboard_primary", "#1f77b4"),
    PLOTLY_COLORS.get("leaderboard_secondary", "#ff7f0e"),
    PLOTLY_COLORS.get("win_probability", "#2ca02c"),
    "#3498db",
    "#e74c3c",
    "#6c757d",
]


def _team_identity_display_label(category: str) -> str:
    return str(category or "-").replace(" impact", "").replace(" Impact", "")


def _team_identity_delta_color(value) -> str:
    num = _numeric_or_nan(value)
    if pd.isna(num):
        return "#6c757d"
    if float(num) > 0:
        return "#3498db"
    if float(num) < 0:
        return "#e74c3c"
    return "#6c757d"


def _team_identity_item(identity: dict, category: str) -> dict:
    for item in identity.get("ranked", []):
        if str(item.get("category", "")) == category:
            return item
    return {"category": category, "value": 0.0}


def _team_identity_feature(label: str, item: dict) -> html.Div:
    category = str(item.get("category", "-"))
    value = item.get("value", 0)
    return html.Div(
        [
            html.Div(label, style = {"fontSize": "11px", "fontWeight": 700, "color": "#6B7C8F", "textTransform": "uppercase"}),
            html.Div(
                [
                    html.Div(
                        _team_identity_display_label(category),
                        style = {"fontWeight": 700, "fontSize": "18px", "lineHeight": "1.25", "minWidth": 0},
                    ),
                    html.Div(
                        _format_signed_impact(value),
                        style = {
                            "fontSize": "30px",
                "fontWeight": 700,
                            "lineHeight": "1.05",
                            "color": _team_identity_delta_color(value),
                            "whiteSpace": "nowrap",
                        },
                    ),
                ],
                style = {"display": "flex", "alignItems": "center", "justifyContent": "space-between", "gap": "12px"},
            ),
        ],
        style = {"minWidth": 0},
    )


def _build_team_identity_card(team: str, identity: dict) -> html.Div:
    ranked = identity.get("ranked", [])
    rows = []
    for index, item in enumerate(ranked, start = 1):
        category = str(item.get("category", "-"))
        value = item.get("value", 0)
        rows.append(
            html.Div(
                [
                    html.Div(str(index), style={"fontSize": "12px", "fontWeight": 700, "color": "#6B7C8F"}),
                    html.Div(
                        _team_identity_display_label(category),
                        style={"fontWeight": 600, "minWidth": 0},
                    ),
                    html.Div(
                        _format_signed_impact(value),
                        style={"fontWeight": 700, "color": _team_identity_delta_color(value), "whiteSpace": "nowrap"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "24px minmax(0, 1fr) auto",
                    "alignItems": "center",
                    "gap": "12px",
                    "padding": "7px 0",
                    "borderBottom": "1px solid #edf1f5",
                },
            )
        )

    primary = _team_identity_item(identity, str(identity.get("primary_strength", "-")))
    secondary = _team_identity_item(identity, str(identity.get("secondary_strength", "-")))
    weakest = _team_identity_item(identity, str(identity.get("weakest_area", "-")))
    sentence = str(identity.get("sentence", ""))
    for category in TEAM_IDENTITY_CONTEXT_LABELS:
        sentence = sentence.replace(category, _team_identity_display_label(category))

    return _card(
        [
            html.Div("How This Team Wins", style={"fontWeight": 700, "fontSize": "18px", "marginBottom": "10px"}),
            html.Div(
                [
                    _team_identity_feature("Primary strength", primary),
                    _team_identity_feature("Secondary strength", secondary),
                    _team_identity_feature("Weakest area", weakest),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "12px"},
            ),
            html.Div(sentence, style={"fontSize": "15px", "marginTop": "12px", "lineHeight": "1.4"}),
            html.Div(rows, style={"marginTop": "8px"}) if rows else html.Div(),
        ],
        body_class_name = "p-3",
    )


def _profile_metric_display_category(category: str) -> str:
    return str(category or "-").replace(" - ", " – ")


def _ordinal_number(value) -> str:
    num = pd.to_numeric(pd.Series([value]), errors = "coerce").iloc[0]
    if pd.isna(num):
        return "-"
    number = int(num)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _season_rank_text(row: pd.Series | dict) -> str:
    rank = row.get("season_rank")
    season = row.get("season")
    rank_text = _ordinal_number(rank)
    if rank_text == "-":
        return "-"
    season_num = pd.to_numeric(pd.Series([season]), errors = "coerce").iloc[0]
    if pd.isna(season_num):
        return rank_text
    return f"{rank_text} in {int(season_num)}"


def _profile_strength_card(row: pd.Series | dict) -> html.Div:
    label = str(row.get("card_label", "-"))
    metric_name = str(row.get("metric_name", "-"))
    category = profile_metric_subtitle(row.get("metric_key"), row.get("category"))
    raw_value = _safe_float_text(row.get("raw_value"), digits = 2)
    rank_text = _season_rank_text(row)
    color = _team_identity_delta_color(row.get("raw_value"))
    return _card(
        [
            html.Div(
                [
                    html.Div(
                        [
                                    html.Div(
                                label,
                                style = {
                                    "fontSize": "11px",
                                    "fontWeight": 700,
                                    "color": "#6B7C8F",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0",
                                },
                            ),
                            html.Div(
                                metric_name,
                style = {"fontWeight": 700, "fontSize": "19px", "lineHeight": "1.2", "marginTop": "3px"},
                            ),
                            html.Div(category, style = {"fontSize": "12px", "color": "#6B7C8F", "marginTop": "2px"}),
                        ],
                        style = {"flex": "0 0 50%", "minWidth": 0},
                    ),
                    html.Div(
                        [
                            html.Div(
                                _format_signed_impact(raw_value),
                style = {"fontWeight": 700, "fontSize": "26px", "lineHeight": "1.05", "color": color},
                            ),
                            html.Div(
                                rank_text,
                                style = {"fontWeight": 700, "fontSize": "14px", "color": "#34495e", "whiteSpace": "nowrap"},
                            ),
                        ],
                        style = {
                            "display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "textAlign": "center",
                            "gap": "2px",
                            "flex": "0 0 50%",
                            "minWidth": 0,
                        },
                    ),
                ],
                style = {
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "stretch",
                    "gap": "12px",
                },
            ),
        ],
        body_class_name = "p-3",
    )


def _team_stat_team_mask(frame: pd.DataFrame, team: str) -> pd.Series:
    if frame.empty or "Team" not in frame.columns:
        return pd.Series(False, index = frame.index)
    team_key = str(team or "").strip().lower()
    return frame["Team"].fillna("").astype(str).apply(
        lambda value: team_key in {token.strip().lower() for token in value.split(",") if token.strip()}
    )


def _team_analysis_leader_item(
    label: str,
    player_name: str,
    value_text: str,
    *,
    subtitle: str | None = None,
    color: str = "#34495e",
    name_font_size: str = "19px",
    value_font_size: str = "24px",
    min_height: str = "72px",
    card_color: str | None = None,
    inverse: bool | None = None,
    label_color: str | None = None,
    name_color: str | None = None,
    subtitle_color: str | None = None,
    show_value: bool = True,
    value_subtitle: str | None = None,
    logo_team: str | None = None,
) -> html.Div:
    label_text_color = label_color or ("rgba(255, 255, 255, 0.82)" if inverse else "#6B7C8F")
    name_text_color = name_color or ("#ffffff" if inverse else None)
    subtitle_text_color = subtitle_color or ("rgba(255, 255, 255, 0.82)" if inverse else "#5D728A")
    value_text_color = "#ffffff" if inverse else color

    name_node = html.Div(
        player_name or "-",
        style = {
            "fontWeight": 700,
            "fontSize": name_font_size,
            "lineHeight": "1.2",
            "marginTop": "3px",
            **({"color": name_text_color} if name_text_color else {}),
        },
    )
    if logo_team:
        name_children = [
            html.Div([name_node, _team_logo_img(logo_team)], className = "team-logo-inline leader-card-team-logo-inline")
        ]
        if subtitle:
            name_children.append(
                html.Div(
                    subtitle,
                    style = {"fontSize": "14px", "fontWeight": 400, "color": subtitle_text_color, "marginTop": "2px", "lineHeight": "1.2"},
                )
            )
    else:
        name_children = [name_node]
        if subtitle:
            name_children.append(
                html.Div(
                    subtitle,
                    style = {"fontSize": "14px", "fontWeight": 400, "color": subtitle_text_color, "marginTop": "2px", "lineHeight": "1.2"},
                )
            )

    return _card(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                label,
                                style = {
                                    "fontSize": "11px",
                                    "fontWeight": 700,
                                    "color": label_text_color,
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0",
                                },
                            ),
                            *name_children,
                        ],
                        style = {"flex": "1 1 100%" if not show_value else "0 0 50%", "minWidth": 0},
                    ),
                    *(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        value_text or "-",
                                        style = {
                                            "fontWeight": 700,
                                            "fontSize": value_font_size,
                                            "lineHeight": "1.1",
                                            "color": value_text_color,
                                            "textAlign": "center",
                                        },
                                    ),
                                    *(
                                        [
                                            html.Div(
                                                value_subtitle,
                                                style = {
                                                    "fontSize": "12px",
                                                    "fontWeight": 600,
                                                    "lineHeight": "1.1",
                                                    "marginTop": "4px",
                                                    "color": subtitle_text_color,
                                                    "textAlign": "center",
                                                    "textTransform": "uppercase",
                                                },
                                            )
                                        ]
                                        if value_subtitle
                                        else []
                                    ),
                                ],
                                style = {
                                    "flex": "0 0 50%",
                                    "minWidth": 0,
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                },
                            ),
                        ]
                        if show_value
                        else []
                    ),
                ],
                style = {
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "gap": "12px",
                    "width": "100%",
                    "minHeight": min_height,
                },
            )
        ],
        body_class_name = "p-3 h-100 d-flex align-items-center",
        color = card_color,
        inverse = inverse,
    )


def _build_team_analysis_leader_cards(leaderboard: Leaderboard, team: str) -> html.Div:
    batter_stats = getattr(leaderboard, "batter_stats", pd.DataFrame())
    bowler_stats = getattr(leaderboard, "bowler_stats", pd.DataFrame())
    impact_stats = getattr(leaderboard, "player_impact_stats", pd.DataFrame())

    top_batter_name = "-"
    top_batter_value = "-"
    if isinstance(batter_stats, pd.DataFrame) and not batter_stats.empty:
        batters = batter_stats.loc[_team_stat_team_mask(batter_stats, team)].copy()
        if not batters.empty:
            batters["Runs"] = pd.to_numeric(batters.get("Runs"), errors = "coerce").fillna(0)
            batters["Strike Rate"] = pd.to_numeric(batters.get("Strike Rate"), errors = "coerce").fillna(0)
            row = batters.sort_values(["Runs", "Strike Rate", "Batter"], ascending = [False, False, True], kind = "mergesort").iloc[0]
            top_batter_name = str(row.get("Batter", "-"))
            top_batter_value = f"{int(row.get('Runs', 0))} (SR: {float(row.get('Strike Rate', 0)):.2f})"

    top_bowler_name = "-"
    top_bowler_value = "-"
    if isinstance(bowler_stats, pd.DataFrame) and not bowler_stats.empty:
        bowlers = bowler_stats.loc[_team_stat_team_mask(bowler_stats, team)].copy()
        if not bowlers.empty:
            bowlers["Wickets"] = pd.to_numeric(bowlers.get("Wickets"), errors = "coerce").fillna(0)
            bowlers["Economy"] = pd.to_numeric(bowlers.get("Economy"), errors = "coerce").fillna(0)
            row = bowlers.sort_values(["Wickets", "Economy", "Bowler"], ascending = [False, True, True], kind = "mergesort").iloc[0]
            top_bowler_name = str(row.get("Bowler", "-"))
            top_bowler_value = f"{int(row.get('Wickets', 0))} (Econ: {float(row.get('Economy', 0)):.2f})"

    top_impact_name = "-"
    top_impact_value = "-"
    top_impact_raw = 0.0
    if isinstance(impact_stats, pd.DataFrame) and not impact_stats.empty:
        impacts = impact_stats.loc[_team_stat_team_mask(impact_stats, team)].copy()
        if not impacts.empty:
            impacts["Total Impact"] = pd.to_numeric(impacts.get("Total Impact"), errors = "coerce").fillna(0)
            row = impacts.sort_values(["Total Impact", "Player"], ascending = [False, True], kind = "mergesort").iloc[0]
            top_impact_name = str(row.get("Player", "-"))
            top_impact_raw = float(row.get("Total Impact", 0))
            top_impact_value = _format_signed_impact(top_impact_raw)

    cards = [
        _team_analysis_leader_item("Top Run-Scorer", top_batter_name, top_batter_value, color = PLOTLY_COLORS.get("innings_2", "#FF7F0E")),
        _team_analysis_leader_item("Top Wicket-Taker", top_bowler_name, top_bowler_value, color = PLOTLY_COLORS.get("leaderboard_secondary", "#e74c3c")),
        _team_analysis_leader_item("Most Impactful Player", top_impact_name, top_impact_value, color = _team_identity_delta_color(top_impact_raw)),
    ]
    return html.Div(
        [html.Div(card, style = {"flex": "1 1 calc(33.333% - 8px)", "minWidth": "260px"}) for card in cards],
        style = {"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "stretch"},
    )


def _profile_radar_figure(profile_metrics: pd.DataFrame, profile: str, title: str):
    fig = go.Figure()
    plot_df = profile_metrics[profile_metrics.get("profile").eq(profile)].copy() if not profile_metrics.empty else pd.DataFrame()
    if plot_df.empty:
        fig.update_layout(
            annotations = [
                dict(text = "No profile metrics available.", x = 0.5, y = 0.5, xref = "paper", yref = "paper", showarrow = False)
            ],
            title = dict(
                text = f"<b>{title}</b>",
                x = 0.5,
                xanchor = "center",
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = "top",
            ),
            margin = {"t": 72, "l": PLOTLY_HEADER_MARGIN_LEFT, "r": PLOTLY_HEADER_MARGIN_RIGHT, "b": 42},
        )
        apply_plot_theme(fig)
        return fig

    plot_df["display_order"] = pd.to_numeric(plot_df.get("display_order"), errors = "coerce").fillna(9999)
    plot_df["percentile"] = pd.to_numeric(plot_df.get("percentile"), errors = "coerce").fillna(0)
    plot_df["raw_value"] = pd.to_numeric(plot_df.get("raw_value"), errors = "coerce").fillna(0)
    plot_df = plot_df.sort_values("display_order", kind = "mergesort").reset_index(drop = True)
    closed_df = pd.concat([plot_df, plot_df.iloc[[0]]], ignore_index = True)
    plot_radius = closed_df["percentile"].clip(lower = 0, upper = 100)

    hover_text = []
    for _, row in closed_df.iterrows():
        description = str(row.get("description") or "")
        lines = [
            f"{row.get('metric_name', '-')}",
            f"Percentile = {float(row.get('percentile', 0)):.0f}",
            f"Impact / Match = {_format_signed_impact(row.get('raw_value'))}",
        ]
        if description:
            lines.append(description)
        hover_text.append("<br>".join(lines))

    color = PLOTLY_COLORS.get("leaderboard_primary", "#3498db") if profile == "batting" else PLOTLY_COLORS.get("leaderboard_secondary", "#e74c3c")
    fill = "rgba(52, 152, 219, 0.18)" if profile == "batting" else "rgba(231, 76, 60, 0.14)"
    fig.add_trace(
        go.Scatterpolar(
            r = plot_radius,
            theta = closed_df["metric_name"],
            mode = "lines+markers",
            fill = "toself",
            line = dict(color = color, width = 2),
            marker = dict(size = 6, color = color),
            fillcolor = fill,
            text = hover_text,
            hovertemplate = "%{text}<extra></extra>",
            showlegend = False,
        )
    )
    fig.update_layout(
        title = dict(
            text = f"<b>{title}</b>",
            x = 0.5,
            xanchor = "center",
            y = PLOTLY_HEADER_TITLE_Y,
            yanchor = "top",
        ),
        polar = dict(
            radialaxis = dict(
                range = [0, 100],
                tickvals = [0, 25, 50, 75, 100],
                showline = False,
                showticklabels = False,
                showgrid = True,
                tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY),
            ),
            angularaxis = dict(
                tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY),
            ),
        ),
        showlegend = False,
        margin = {"t": 72, "l": PLOTLY_HEADER_MARGIN_LEFT, "r": PLOTLY_HEADER_MARGIN_RIGHT, "b": 42},
    )
    apply_plot_theme(fig)
    return fig


def _build_team_profile_identity_section(profile_metrics: pd.DataFrame) -> html.Div:
    strength_rows = select_profile_strength_cards(profile_metrics)
    cards = [
        _profile_strength_card(row)
        for _, row in strength_rows.iterrows()
    ]
    if not cards:
        cards = [html.Div("Cached profile metrics are unavailable for this season.", style = {"padding": "8px 0"})]

    return html.Div(
        children = [
            html.Div(
                [html.Div(card, style = {"flex": "1 1 0", "minWidth": "260px"}) for card in cards],
                style = {"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "stretch"},
            ),
            html.Div(style = {"height": "12px"}),
            html.Div(
                [
                    html.Div(
                        _team_analysis_graph_card(
                            "Batting Profile",
                            _profile_radar_figure(profile_metrics, "batting", "Batting Profile"),
                            show_card_title = False,
                            height = 520,
                            graph_top_margin = 72,
                        ),
                        style = {"flex": "1 1 420px", "minWidth": 0},
                    ),
                    html.Div(
                        _team_analysis_graph_card(
                            "Bowling Profile",
                            _profile_radar_figure(profile_metrics, "bowling", "Bowling Profile"),
                            show_card_title = False,
                            height = 520,
                            graph_top_margin = 72,
                        ),
                        style = {"flex": "1 1 420px", "minWidth": 0},
                    ),
                ],
                style = {"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "stretch"},
            ),
        ],
    )


def _team_analysis_heatmap_display_data(heatmap_df: pd.DataFrame, matches: pd.DataFrame, team: str) -> pd.DataFrame:
    if heatmap_df.empty:
        return heatmap_df.copy()

    display = heatmap_df.copy()
    display["match_id"] = pd.to_numeric(display["match_id"], errors = "coerce")

    team_matches = filter_team_season_matches(matches, team)
    if not team_matches.empty and "match_id" in team_matches.columns:
        order = team_matches.copy()
        order["match_id"] = pd.to_numeric(order["match_id"], errors = "coerce")
        order["Opponent"] = order.apply(lambda row: opponent_for_match(row, team), axis = 1)
        order_cols = [col for col in ["match_id", "date", "Opponent"] if col in order.columns]
        display = order[order_cols].merge(display, on = "match_id", how = "left")
        display["Match"] = display["Match"].fillna("")
        for column in PHASE_COLUMNS:
            display[column] = pd.to_numeric(display[column], errors = "coerce").fillna(0)
        sort_cols = [col for col in ["date", "match_id"] if col in display.columns]
        if sort_cols:
            display = display.sort_values(sort_cols, ascending = [False] * len(sort_cols), kind = "mergesort")
    elif "Opponent" not in display.columns:
        display["Opponent"] = "-"

    return display.reset_index(drop = True)


def _team_phase_heatmap_figure(heatmap_df: pd.DataFrame):
    fig = go.Figure()
    height = team_phase_heatmap_height(len(heatmap_df))
    heatmap_legend_tick_size = max(10, int(PLOTLY_BASE_FONT_SIZE) - 2)
    if heatmap_df.empty:
        fig.update_layout(
            annotations = [
                dict(text = "No phase impact available.", x = 0.5, y = 0.5, xref = "paper", yref = "paper", showarrow = False)
            ],
            title = dict(
                text = "<b>Phase Impact Heatmap</b>",
                x = 0.5,
                xanchor = "center",
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = "top",
            ),
            height = height,
        )
        apply_plot_theme(fig)
        return fig

    z = heatmap_df[PHASE_COLUMNS].apply(pd.to_numeric, errors = "coerce").fillna(0).to_numpy()
    max_abs = float(np.nanmax(np.abs(z))) if z.size else 0.0
    bound = max_abs if max_abs > 0 else 1.0
    customdata = np.dstack(
        [
            np.tile(heatmap_df["Match"].astype(str).to_numpy()[:, None], (1, len(PHASE_COLUMNS))),
            np.tile(heatmap_df.get("Opponent", pd.Series("-", index = heatmap_df.index)).astype(str).to_numpy()[:, None], (1, len(PHASE_COLUMNS))),
        ]
    )
    fig.add_trace(
        go.Heatmap(
            z = z,
            x = PHASE_COLUMNS,
            y = heatmap_df["Match"].astype(str),
            colorscale = [[0.0, "#e74c3c"], [0.5, "#ffffff"], [1.0, "#3498db"]],
            zmid = 0,
            zmin = -bound,
            zmax = bound,
            colorbar = dict(
                title = dict(text = "Impact", font = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY)),
                tickfont = dict(size = heatmap_legend_tick_size, family = PLOTLY_FONT_FAMILY),
            ),
            customdata = customdata,
            hovertemplate = (
                "Match = %{customdata[0]}<br>"
                "Opponent = %{customdata[1]}<br>"
                "Phase = %{x}<br>"
                "Impact = %{z:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title = dict(
            text = "<b>Phase Impact Heatmap</b>",
            x = 0.5,
            xanchor = "center",
            y = PLOTLY_HEADER_TITLE_Y,
            yanchor = "top",
        ),
        height = height,
        xaxis = dict(side = "top", tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY)),
        yaxis = dict(autorange = "reversed", tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY)),
        margin = {"t": 72, "l": PLOTLY_HEADER_MARGIN_LEFT, "r": PLOTLY_HEADER_MARGIN_RIGHT, "b": 42},
    )
    apply_plot_theme(fig)
    # Keep heatmap axis and colorbar text at the same size as radar/other dashboard graph labels.
    fig.update_xaxes(tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY))
    fig.update_yaxes(tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY))
    return fig


def _team_batting_role_figure(role_impact: pd.DataFrame):
    fig = go.Figure()
    plot_df = role_impact.copy()
    plot_df["Role"] = pd.Categorical(plot_df["Role"], categories = BATTING_ROLE_ORDER, ordered = True)
    plot_df = plot_df.sort_values("Role")
    values = pd.to_numeric(plot_df["Avg Impact / Match"], errors = "coerce").fillna(0)
    slice_values = values.abs()
    if float(slice_values.sum()) <= 0:
        fig.update_layout(
            annotations = [
                dict(text = "No batting impact available.", x = 0.5, y = 0.5, xref = "paper", yref = "paper", showarrow = False)
            ],
            font = dict(family = APP_FONT_STACK),
        )
        return fig
    fig.add_trace(
        go.Pie(
            labels = plot_df["Role"].astype(str),
            values = slice_values,
            customdata = values,
            marker = dict(colors = TEAM_ANALYSIS_PIE_COLORS[: len(plot_df)]),
            text = [_format_signed_impact(value) for value in values],
            textinfo = "label+text+percent",
            sort = False,
            direction = "clockwise",
            hovertemplate = "Role = %{label}<br>Impact / Match = %{customdata:.2f}<br>Share = %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend = True,
        font = dict(family = APP_FONT_STACK),
    )
    return fig


def _team_bowling_phase_figure(phase_averages: dict[str, float], bowling_hover: pd.DataFrame):
    phase_columns = ["PP Bowling", "Middle Bowling", "Death Bowling"]
    labels = BOWLING_PHASE_ORDER
    values = [float(phase_averages.get(column, 0.0)) for column in phase_columns]
    slice_values = [abs(value) for value in values]
    hover_lookup = {}
    if isinstance(bowling_hover, pd.DataFrame) and not bowling_hover.empty:
        hover_lookup = {str(row.get("Phase")): row for _, row in bowling_hover.iterrows()}

    customdata = []
    for label in ["Powerplay", "Middle overs", "Death overs"]:
        row = hover_lookup.get(label, {})
        customdata.append(
            [
                _safe_float_text(row.get("Economy") if isinstance(row, pd.Series) else None, digits = 2),
                _safe_float_text(row.get("Wickets") if isinstance(row, pd.Series) else None, digits = 0),
                _safe_float_text(row.get("Dot Ball %") if isinstance(row, pd.Series) else None, digits = 1),
            ]
        )

    fig = go.Figure()
    if float(sum(slice_values)) <= 0:
        fig.update_layout(
            annotations = [
                dict(text = "No bowling impact available.", x = 0.5, y = 0.5, xref = "paper", yref = "paper", showarrow = False)
            ],
            font = dict(family = APP_FONT_STACK),
        )
        return fig

    fig.add_trace(
        go.Pie(
            labels = labels,
            values = slice_values,
            text = [_format_signed_impact(value) for value in values],
            textinfo = "label+text+percent",
            sort = False,
            direction = "clockwise",
            marker = dict(colors = TEAM_ANALYSIS_PIE_COLORS[: len(labels)]),
            customdata = [[values[index], *customdata[index]] for index in range(len(values))],
            hovertemplate = (
                "Phase = %{label}<br>"
                "Impact / Match = %{customdata[0]:.2f}<br>"
                "Share = %{percent}<br>"
                "Economy = %{customdata[1]}<br>"
                "Wickets = %{customdata[2]}<br>"
                "Dot Ball % = %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        showlegend = True,
        font = dict(family = APP_FONT_STACK),
    )
    return fig


def _team_match_table(table_df: pd.DataFrame, *, season: int | None = None) -> dbc.Table | html.Div:
    if table_df.empty:
        return html.Div("No matches found for this team and season.", style={"padding": "8px 0"})

    display_df = table_df.copy()
    if "Team Total Impact" in display_df.columns:
        display_df["Team Total Impact"] = display_df["Team Total Impact"].apply(_format_signed_impact)

    rendered_columns = [col for col in display_df.columns if col != "Match"]

    def _match_href(row: pd.Series) -> str | None:
        if "Match" not in row.index:
            return None
        match_text = "" if pd.isna(row["Match"]) else str(row["Match"])
        if match_text.startswith("[Open](") and match_text.endswith(")"):
            return match_text[len("[Open]("):-1]
        return None

    def _cell_value(column: str, value):
        text = "" if pd.isna(value) else str(value)
        return text

    def _cell_style(column: str, value) -> dict:
        if column != "Team Total Impact":
            return {}
        text = "" if pd.isna(value) else str(value)
        color = "#34495e"
        if text.startswith("+"):
            color = "#3498db"
        elif text.startswith("-"):
            color = "#e74c3c"
        return {"fontWeight": 700, "color": color}

    def _render_cell(column: str, row: pd.Series):
        value = _cell_value(column, row[column])
        if column == "Date":
            href = _match_href(row)
            if href:
                value = html.A(value, href = href)
        elif column == "Opponent":
            value = html.A(
                _team_text_with_logo(
                    value,
                    html.Span(value),
                    "team-logo-inline recent-match-team-logo-inline",
                ),
                href = _team_analysis_href(value, season),
                className = "team-analysis-link-plain",
            )
        return html.Td(value, style = _cell_style(column, row[column]))

    header = html.Thead(html.Tr([html.Th(col) for col in rendered_columns]))
    body = html.Tbody(
        [
            html.Tr(
                [_render_cell(col, row) for col in rendered_columns]
            )
            for _, row in display_df.iterrows()
        ]
    )
    return dbc.Table(
        [header, body],
        striped = True,
        bordered = False,
        hover = True,
        size = "sm",
        responsive = True,
        class_name = "mb-0",
    )


def _build_team_analysis_dashboard(leaderboard: Leaderboard, team: str) -> html.Div:
    matches = getattr(leaderboard, "matches", pd.DataFrame())
    impact_balls = getattr(leaderboard, "_impact_balls", pd.DataFrame())
    team_matches = filter_team_season_matches(matches, team)
    if team_matches.empty:
        return html.Div("No finalized matches found for this team and season.", style={"padding": "8px 0"})

    phase_data = aggregate_team_phase_impact(impact_balls, matches, team)
    heatmap_df = phase_data["heatmap"]
    heatmap_display_df = _team_analysis_heatmap_display_data(heatmap_df, matches, team)
    selected_profile_metrics = _team_identity_selected_metrics(leaderboard, team)
    try:
        benchmark_metrics = _load_team_identity_benchmarks(_team_identity_benchmark_cache_token())
    except Exception:
        benchmark_metrics = pd.DataFrame()
    profile_metrics = compute_profile_percentiles(selected_profile_metrics, benchmark_metrics)
    season_metrics = _team_identity_season_metrics(leaderboard)
    profile_metrics = compute_profile_season_ranks(
        profile_metrics,
        season_metrics,
        team,
        getattr(leaderboard, "season", None),
    )
    match_table_df = build_match_table_rows(
        matches,
        team,
        heatmap_df,
        top_impact_player_by_match(impact_balls, matches, team),
    )

    return html.Div(
        children = [
            _build_team_analysis_leader_cards(leaderboard, team),
            html.Div(style = {"height": "12px"}),
            _card(
                [
                    html.Div("Recent matches", style={"fontWeight": 700, "fontSize": "18px", "marginBottom": "8px"}),
                    _team_match_table(match_table_df, season = getattr(leaderboard, "season", None)),
                ],
                body_class_name = "p-3",
            ),
            html.Div(style = {"height": "12px"}),
            _build_team_profile_identity_section(profile_metrics),
            html.Div(style = {"height": "12px"}),
            _team_analysis_graph_card(
                "Phase Impact Heatmap",
                _team_phase_heatmap_figure(heatmap_display_df),
                show_card_title = False,
                height = team_phase_heatmap_height(len(heatmap_display_df)),
                graph_top_margin = 72,
            ),
        ]
    )


def _playbyplay_total_pages(total_overs: int, overs_per_page: int = PLAYBYPLAY_DESC_OVERS_PER_PAGE) -> int:
    if total_overs <= 0:
        return 1
    return max(1, int(ceil(float(total_overs) / float(max(1, overs_per_page)))))


def _playbyplay_token_badge(token_text: str, token_style: str):
    base_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "justifyContent": "center",
        "minWidth": "34px",
        "height": "34px",
        "padding": "0 8px",
        "borderRadius": "7px",
        "fontSize": "13px",
        "fontWeight": 700,
        "border": "1px solid #d0d7de",
        "backgroundColor": "#f8f9fa",
        "color": "#212529",
    }

    if token_style == "wicket":
        base_style.update({"backgroundColor": "#dc3545", "color": "#ffffff", "border": "1px solid #dc3545"})
    elif token_style == "six":
        base_style.update({"backgroundColor": "#0d6efd", "color": "#ffffff", "border": "1px solid #0d6efd"})
    elif token_style == "four":
        base_style.update({"backgroundColor": "#28a745", "color": "#ffffff", "border": "1px solid #28a745"})

    return html.Span(token_text, style = base_style)


def _playbyplay_delta_style(delta_value):
    delta = pd.to_numeric(pd.Series([delta_value]), errors = "coerce").iloc[0]
    if pd.isna(delta):
        return "-", "#6c757d", "rgba(108, 117, 125, 0.10)"

    delta_float = float(delta)
    text = f"{delta_float:+.1f}%"
    if delta_float > 1:
        return text, "#3498db", "rgba(52, 152, 219, 0.10)"
    if delta_float < -1:
        return text, "#e74c3c", "rgba(231, 76, 60, 0.10)"
    return text, "#6c757d", "rgba(108, 117, 125, 0.10)"


def _playbyplay_secondary_delta_style(
    delta_value,
    *,
    favorable_when_positive: bool,
    threshold: float,
    digits: int,
    suffix: str = "",
):
    delta = pd.to_numeric(pd.Series([delta_value]), errors = "coerce").iloc[0]
    if pd.isna(delta):
        return "-", "#6c757d"

    delta_float = float(delta)
    text = f"{delta_float:+.{digits}f}{suffix}"
    favorable = delta_float > threshold if favorable_when_positive else delta_float < -threshold
    unfavorable = delta_float < -threshold if favorable_when_positive else delta_float > threshold
    if favorable:
        return text, "#3498db"
    if unfavorable:
        return text, "#e74c3c"
    return text, "#6c757d"


def _playbyplay_value_text(value, *, digits: int = 1, fallback: str = "-") -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors = "coerce").iloc[0]
    if pd.isna(parsed):
        return fallback
    if int(digits) <= 0:
        return str(int(round(float(parsed))))
    return f"{float(parsed):.{int(digits)}f}"


def _playbyplay_delta_panel(over: dict, delta_text: str, delta_color: str) -> html.Div:
    innings = int(over.get("innings", 0) or 0)
    batting_team_name = str(over.get("team", "") or "").strip() or "Batting team"
    secondary_label = ""
    secondary_value_text = "-"
    secondary_value_color = "#6c757d"
    secondary_range_text = "- → -"
    if innings == 1:
        secondary_value_text, secondary_value_color = _playbyplay_secondary_delta_style(
            over.get("projected_score_delta"),
            favorable_when_positive = True,
            threshold = 1,
            digits = 0,
        )
        prev_text = _playbyplay_value_text(over.get("previous_projected_score"), digits = 0)
        new_text = _playbyplay_value_text(over.get("projected_score"), digits = 0)
        secondary_label = "Proj Runs Δ"
        secondary_range_text = f"{prev_text} → {new_text}"
    elif innings == 2:
        secondary_value_text, secondary_value_color = _playbyplay_secondary_delta_style(
            over.get("required_rr_delta"),
            favorable_when_positive = False,
            threshold = 0.1,
            digits = 1,
        )
        prev_text = _playbyplay_value_text(over.get("previous_required_rr"), digits = 1)
        new_text = _playbyplay_value_text(over.get("required_rr"), digits = 1)
        secondary_label = "Req RR Δ"
        secondary_range_text = f"{prev_text} → {new_text}"

    metric_card_style = {
        "backgroundColor": "#ffffff",
        "border": "none",
        "borderRadius": "8px",
        "padding": "10px 12px",
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "textAlign": "center",
        "minHeight": "104px",
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        delta_text,
                        style={"fontSize": "30px", "fontWeight": 700, "lineHeight": "1.05", "color": delta_color},
                    ),
                    html.Div(
                        "Win Prob Δ",
                        style={"fontSize": "12px", "fontWeight": 700, "color": "#6c757d", "marginTop": "4px"},
                    ),
                    html.Div(
                        batting_team_name,
                        style={"fontSize": "13px", "fontWeight": 600, "color": "#495057", "marginTop": "4px"},
                    ),
                ],
                style=metric_card_style,
            ),
            html.Div(
                [
                    html.Div(
                        secondary_value_text,
                        style={"fontSize": "30px", "fontWeight": 700, "lineHeight": "1.05", "color": secondary_value_color},
                    ),
                    html.Div(
                        secondary_label,
                        style={"fontSize": "12px", "fontWeight": 700, "color": "#6c757d", "marginTop": "4px"},
                    ),
                    html.Div(
                        secondary_range_text,
                        style={"fontSize": "13px", "fontWeight": 600, "color": "#495057", "marginTop": "4px"},
                    ),
                ],
                style=metric_card_style,
            ),
        ],
        style={
            "flex": "1.5 1 calc(60% - 6px)",
            "minWidth": 0,
            "display": "grid",
            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
            "gap": "12px",
            "alignItems": "stretch",
        },
    )


def _build_play_by_play_tab(
    match: Match,
    *,
    desc_page: int = 1,
    ribbon_page: int = 1,
    desc_overs_per_page: int = PLAYBYPLAY_DESC_OVERS_PER_PAGE,
    ribbon_overs_per_page: int = PLAYBYPLAY_RIBBON_OVERS_PER_PAGE,
) -> html.Div:
    overs = match.play_by_play_feed()
    total_overs = len(overs)
    desc_total_pages = _playbyplay_total_pages(total_overs, overs_per_page = desc_overs_per_page)
    ribbon_total_pages = _playbyplay_total_pages(total_overs, overs_per_page = ribbon_overs_per_page)

    desc_page_safe = max(1, min(int(desc_page or 1), int(desc_total_pages)))
    ribbon_page_safe = max(1, min(int(ribbon_page or 1), int(ribbon_total_pages)))
    desc_chunk_size = max(1, int(desc_overs_per_page))
    ribbon_chunk_size = max(1, int(ribbon_overs_per_page))

    desc_start = (desc_page_safe - 1) * desc_chunk_size
    desc_chunk = overs[desc_start : desc_start + desc_chunk_size]

    ribbon_start = (ribbon_page_safe - 1) * ribbon_chunk_size
    ribbon_chunk = overs[ribbon_start : ribbon_start + ribbon_chunk_size]

    ribbon_segments = []
    for over in ribbon_chunk:
        deliveries = over.get("deliveries", [])
        token_row = html.Div(
            children=[
                _playbyplay_token_badge(d.get("token_text", ""), d.get("token_style", "default"))
                for d in list(deliveries)[::-1]
            ],
            style={
                "display": "flex",
                "gap": "6px",
                "flexWrap": "nowrap",
                "alignItems": "center",
            },
        )
        ribbon_segments.append(
            html.Div(
                children=[
                    html.Div(
                        f"{over.get('over_display')}",
                        style={"fontWeight": 700, "fontSize": "13px", "minWidth": "24px"},
                    ),
                    token_row,
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "4px",
                    "padding": "4px 6px",
                    "flex": "0 0 auto",
                },
            )
        )

    ribbon_row = html.Div(
        children=[
            dbc.Button(
                "◀",
                id={"type": "playbyplay-ribbon-arrow", "dir": "left"},
                n_clicks = 0,
                size = "sm",
                color = "secondary",
                outline = True,
                disabled = (ribbon_page_safe <= 1),
                style = {"height": "32px", "width": "32px", "padding": 0},
            ),
            html.Div(
                children=ribbon_segments if ribbon_segments else [html.Div("No overs available yet.", style={"padding": "4px 6px"})],
                style={
                    "display": "flex",
                    "gap": "10px",
                    "alignItems": "center",
                    "overflowX": "auto",
                    "flex": "1 1 auto",
                    "minWidth": 0,
                    "paddingBottom": "2px",
                },
            ),
            dbc.Button(
                "▶",
                id={"type": "playbyplay-ribbon-arrow", "dir": "right"},
                n_clicks = 0,
                size = "sm",
                color = "secondary",
                outline = True,
                disabled = (ribbon_page_safe >= ribbon_total_pages),
                style = {"height": "32px", "width": "32px", "padding": 0},
            ),
        ],
        style={"display": "flex", "gap": "8px", "alignItems": "center"},
    )

    desc_children = []
    team1_name = str(getattr(match, "batting_team", "") or "").strip()
    team2_name = str(getattr(match, "bowling_team", "") or "").strip()
    match_team_name = getattr(match, "_match_team_name", None)
    if callable(match_team_name):
        try:
            team1_name = str(match_team_name(team1_name)).strip()
            team2_name = str(match_team_name(team2_name)).strip()
        except Exception:
            pass

    for over in desc_chunk:
        team_name = str(over.get('team', '') or '').strip()
        status_text = str(over.get('status_text', '') or '').strip()
        if team_name and status_text.lower().startswith((team_name + ' ').lower()):
            status_text = status_text[len(team_name) + 1 :].strip()

        over_runs = int(over.get('over_runs', 0) or 0)
        over_wickets = int(over.get('over_wickets', 0) or 0)
        delta_text, delta_color, header_background = _playbyplay_delta_style(
            over.get('batting_win_prob_delta')
        )

        over_lines = []
        for delivery in over.get("deliveries", []):
            is_wicket_delivery = str(delivery.get("token_style", "") or "").strip().lower() == "wicket"
            over_lines.append(
                html.Div(
                    children=[
                        html.Span(
                            delivery.get("ball_display", ""),
                            style={
                                "fontSize": "15px",
                                "fontWeight": 700 if is_wicket_delivery else 400,
                                "minWidth": "44px",
                                "marginRight": "2px",
                            },
                        ),
                        html.Span(
                            _playbyplay_token_badge(
                                delivery.get("token_text", ""),
                                delivery.get("token_style", "default"),
                            ),
                            style={"display": "inline-flex", "marginRight": "12px"},
                        ),
                        html.Span(
                            delivery.get("line_text", ""),
                            style={
                                "fontSize": "15px",
                                "lineHeight": "1.42",
                                "fontWeight": 700 if is_wicket_delivery else 400,
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "8px",
                        "marginBottom": "6px",
                    },
                )
            )

        desc_children.append(
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Div(
                                        [
                                            html.Span(f"Over {over.get('over_display')} - "),
                                            _team_logo_img(team_name, "team-logo playbyplay-over-logo"),
                                            html.Span(f"{team_name} {over.get('score_text', '')}".strip()),
                                        ],
                                        className = "team-logo-inline playbyplay-over-title-logo-inline",
                                        style={"fontWeight": 700, "fontSize": "20px", "color": "#212529"},
                                    ),
                            html.Div(
                                children=[
                                    html.Span(f"{over_runs} runs"),
                                    html.Span("•"),
                                    html.Span(f"{over_wickets} wkts"),
                                    html.Span(
                                        f"(Dismissed: {', '.join(over.get('dismissed_batters', []))})",
                                    ) if over.get('dismissed_batters') else None,
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "8px",
                                    "flexWrap": "wrap",
                                    "fontSize": "14px",
                                    "fontWeight": 600,
                                    "color": "#495057",
                                    "marginTop": "4px",
                                },
                            ),
                            html.Div(
                                children=[
                                    html.Span(f"{team1_name}: {_playbyplay_value_text(over.get('team1_win_probability'), digits = 1)}%"),
                                    html.Span(f"{team2_name}: {_playbyplay_value_text(over.get('team2_win_probability'), digits = 1)}%"),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "12px",
                                    "flexWrap": "wrap",
                                    "fontSize": "13px",
                                    "fontWeight": 400,
                                    "color": "#6c757d",
                                    "marginTop": "3px",
                                },
                            ),
                            html.Div(
                                children=[
                                    html.Span(f"Run Rate: {over.get('run_rate_text', '-')}"),
                                    html.Span(
                                        f"Projected Score: {over.get('required_run_rate_text', '-')}",
                                    ) if int(over.get('innings', 0)) == 1 else None,
                                    html.Span(
                                        f"Required Run Rate: {over.get('required_run_rate_text', '-')}",
                                    ) if int(over.get('innings', 0)) == 2 else None,
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "fontSize": "13px",
                                    "fontWeight": 400,
                                    "color": "#6c757d",
                                    "marginTop": "3px",
                                },
                            ),
                                ],
                                style={
                                    "flex": "1 1 auto",
                                    "minWidth": 0,
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "stretch",
                            "gap": "12px",
                            "backgroundColor": header_background,
                            "borderRadius": "8px",
                            "padding": "10px 12px",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                children=over_lines,
                                style={"flex": "1 1 calc(40% - 8px)", "minWidth": 0},
                            ),
                            _playbyplay_delta_panel(over, delta_text, delta_color),
                        ],
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "alignItems": "stretch",
                            "flexWrap": "nowrap",
                        },
                    ),
                ],
                style={
                    "padding": "8px 10px",
                    "border": "none",
                    "borderRadius": "8px",
                    "backgroundColor": "#ffffff",
                    "marginBottom": "10px",
                },
            )
        )

    if not desc_children:
        desc_children = [html.Div("No ball-by-ball data available yet.", style={"padding": "8px 0"})]

    desc_pagination = dbc.Pagination(
        id = {"type": "playbyplay-desc-pagination", "name": "main"},
        max_value = int(desc_total_pages),
        active_page = int(desc_page_safe),
        fully_expanded = False,
        first_last = False,
        previous_next = False,
        size = "sm",
    )

    return html.Div(
        children=[
            _card(
                [
                    ribbon_row,
                ],
                body_class_name = "p-2",
            ),
            html.Div(style={"height": "8px"}),
            _card(
                desc_children,
                body_class_name = "p-3",
            ),
            html.Div(
                children=[desc_pagination],
                style={"display": "flex", "justifyContent": "center", "marginTop": "10px"},
            ),
        ],
        style={"marginTop": "8px"},
    )


def _build_scorecard_tab(match: Match, summary: dict) -> html.Div:
    innings_blocks = []
    team_batting_first = summary["team1"]
    team_bowling_first = summary["team2"]
    scorecard_warning_message = summary.get("scorecard_consistency_message")

    def _format_overs_label(overs_value) -> str:
        overs_text = str(overs_value)
        if overs_text.endswith(".0"):
            overs_text = overs_text[:-2]
        return overs_text

    innings_data = [
        (
            1,
            str(team_batting_first),
            str(team_bowling_first),
            match.bat1.reset_index(),
            match.bowl1.reset_index(),
            summary["innings1"],
        ),
        (
            2,
            str(team_bowling_first),
            str(team_batting_first),
            match.bat2.reset_index(),
            match.bowl2.reset_index(),
            summary["innings2"],
        ),
    ]

    def _scorecard_section_row(
        label: str,
        table_df: pd.DataFrame,
        figure,
        batting_total_text: str | None = None,
        annotate_not_out: bool = False,
    ) -> html.Div:
        section_label = ""
        team_label = label
        if ": " in str(label):
            section_label, team_label = str(label).split(": ", 1)
        heading = html.Div(
            [
                html.Span(f"{section_label}: " if section_label else ""),
                _team_logo_img(team_label, "team-logo scorecard-title-logo"),
                html.Span(team_label),
            ],
            className = "team-logo-inline scorecard-title-logo-inline",
            style={"fontWeight": 700, "fontSize": "18px", "lineHeight": "1.18"},
        )
        figure.update_layout(
            width = None,
            height = 290,
            margin = {"l": 40, "r": 15, "t": 45, "b": 35},
        )

        return html.Div(
            children=[
                html.Div(
                    children=[
                        _card(
                            [
                                html.Div(heading, style={"marginBottom": "8px"}),
                                _data_table(
                                    table_df,
                                    annotate_not_out = annotate_not_out,
                                    center_numeric_columns = True,
                                ),
                                html.Div(
                                    batting_total_text or "",
                                    style={
                                        "marginTop": "12px",
                                        "fontSize": "18px",
                                        "fontWeight": 700,
                                        "textAlign": "right",
                                        "width": "100%",
                                    },
                                ) if batting_total_text else html.Div(),
                            ],
                            body_class_name = "p-3",
                        ),
                    ],
                    style={"flex": "2 1 0", "minWidth": 0, "minHeight": "400px"},
                ),
                html.Div(
                    children=[
                        _card(
                            [
                                dcc.Graph(
                                    figure = figure,
                                    config = {"displayModeBar": False},
                                    style = {"height": "100%"},
                                ),
                            ],
                            body_class_name = "p-2",
                        ),
                    ],
                    style={
                        "flex": "1 1 0",
                        "minWidth": 0,
                        "minHeight": "400px",
                    },
                ),
            ],
            style={"display": "flex", "gap": "12px", "alignItems": "stretch", "minHeight": "400px"},
        )

    for innings_no, batting_team_name, bowling_team_name, bat_df, bowl_df, innings_summary in innings_data:
        bat_fig = match.batter_impact_graph(innings = innings_no)
        bowl_fig = match.bowler_impact_graph(innings = innings_no)
        total_text = (
            f"Total: {innings_summary['runs']}/{innings_summary['wickets']} "
            f"({_format_overs_label(innings_summary['overs'])} overs)"
        )
        innings_blocks.append(
            html.Div(
                children=[
                    html.H4(
                        f"Innings {innings_no}",
                        style={"margin": "0 0 10px", "fontWeight": 700},
                    ),
                    _scorecard_section_row(
                        f"Batting: {batting_team_name}",
                        bat_df,
                        bat_fig,
                        total_text,
                        annotate_not_out = True,
                    ),
                    html.Div(style={"height": "12px"}),
                    _scorecard_section_row(
                        f"Bowling: {bowling_team_name}",
                        bowl_df,
                        bowl_fig,
                        annotate_not_out = False,
                    ),
                ],
                style={"minWidth": 0},
            )
        )

    children = []
    if scorecard_warning_message:
        children.append(
            dbc.Alert(
                scorecard_warning_message,
                color = "warning",
                class_name = "mb-2",
                style = {"fontSize": "13px"},
            )
        )
    children.extend(innings_blocks)

    return html.Div(
        children=children,
        style={"display": "flex", "flexDirection": "column", "gap": "16px", "marginTop": "12px"},
    )


def _build_total_impact_tab(match: Match) -> html.Div:
    fig = match.total_impact_graph()
    fig.update_layout(
        width = None,
        height = 800,
        margin = {"l": 45, "r": 20, "t": 55, "b": 35},
    )

    return html.Div(
        children=[
            _card(
                [
                    dcc.Graph(
                        figure = fig,
                        config = {"displayModeBar": False},
                    ),
                ],
                body_class_name = "p-2",
            )
        ],
        style={"marginTop": "10px"},
    )


def _safe_int_text(value) -> str:
    num = pd.to_numeric(pd.Series([value]), errors = "coerce").iloc[0]
    if pd.isna(num):
        return "0"
    num_float = float(num)
    return str(int(round(num_float))) if num_float.is_integer() else f"{num_float:g}"


def _safe_float_text(value, digits: int = 2, fallback: str = "-") -> str:
    num = pd.to_numeric(pd.Series([value]), errors = "coerce").iloc[0]
    if pd.isna(num):
        return fallback
    return f"{float(num):.{digits}f}"


def _player_name_key(value: object) -> str:
    return "" if value is None else str(value).strip().lower()


def _format_player_batting_summary(match: Match, player_name: str) -> str:
    bat = getattr(match, "bat", pd.DataFrame()).copy()
    if bat.empty or "Batter" not in bat.columns:
        return "DNB"

    player_key = _player_name_key(player_name)
    rows = bat[bat["Batter"].apply(_player_name_key).eq(player_key)].copy()
    if rows.empty:
        return "DNB"

    rows["Runs_num"] = pd.to_numeric(rows.get("Runs"), errors = "coerce").fillna(0)
    rows = rows.sort_values("Runs_num", ascending = False)
    row = rows.iloc[0]
    runs = _safe_int_text(row.get("Runs"))
    balls = _safe_int_text(row.get("Balls"))
    status = str(row.get("Status", "") or "").strip().lower()
    not_out_marker = "*" if status == "not out" else ""
    return f"{runs}{not_out_marker} ({balls})"


def _format_player_bowling_summary(match: Match, player_name: str) -> str:
    bowl = getattr(match, "bowl", pd.DataFrame()).copy()
    if bowl.empty or "Bowler" not in bowl.columns or "Overs" not in bowl.columns:
        return "Did not bowl"

    player_key = _player_name_key(player_name)
    rows = bowl[bowl["Bowler"].apply(_player_name_key).eq(player_key)].copy()
    if rows.empty:
        return "Did not bowl"

    rows["valid_balls_num"] = rows.get("Overs", "").apply(
        lambda value: _overs_text_to_balls(str(value))
    )
    rows = rows.sort_values("valid_balls_num", ascending = False)
    row = rows.iloc[0]
    if int(row.get("valid_balls_num", 0) or 0) <= 0:
        return "Did not bowl"
    wickets = _safe_int_text(row.get("Wickets"))
    runs = _safe_int_text(row.get("Runs"))
    return f"{wickets}/{runs}"


def _overs_text_to_balls(value: str) -> int:
    text = "" if value is None else str(value).strip()
    if not text:
        return 0
    if "." not in text:
        try:
            return int(float(text)) * 6
        except (TypeError, ValueError):
            return 0
    overs_text, balls_text = text.split(".", 1)
    try:
        overs = int(float(overs_text or 0))
        balls = int(float((balls_text or "0")[0]))
    except (TypeError, ValueError):
        return 0
    return max(0, overs * 6 + balls)


def _most_impactful_player_summary(match: Match) -> dict[str, str] | None:
    impact = getattr(match, "impact", pd.DataFrame()).copy()
    if impact.empty or "Total Impact" not in impact.columns:
        return None

    impact["Total Impact"] = pd.to_numeric(impact["Total Impact"], errors = "coerce")
    impact = impact.dropna(subset = ["Total Impact"]).sort_values("Total Impact", ascending = False)
    if impact.empty:
        return None

    row = impact.iloc[0]
    player_name = str(row.get("Player", "") or "").strip()
    if not player_name:
        return None

    batting = _format_player_batting_summary(match, player_name)
    bowling = _format_player_bowling_summary(match, player_name)
    team_name = str(row.get("Team", "") or "").strip() or "-"
    total_impact = _safe_float_text(row.get("Total Impact"), digits = 2, fallback = "0.00")
    total_impact_value = _numeric_or_nan(row.get("Total Impact"))
    scoreline = f"{batting} & {bowling}"
    return {
        "player": player_name,
        "team": team_name,
        "batting": batting,
        "bowling": bowling,
        "scoreline": scoreline,
        "total_impact": _format_signed_impact(total_impact_value),
        "total_impact_value": total_impact_value,
        "summary": f"{batting} & {bowling}, Total Impact: {total_impact}",
    }


def _format_most_impactful_over(over: dict | None, match: Match | None = None) -> dict[str, str]:
    if not over:
        return {
            "title": "No over data available",
            "detail": "-",
            "swing": "-",
            "beneficiary": "-",
            "color": "#5D728A",
        }

    innings = _safe_int_text(over.get("innings"))
    over_no = _safe_int_text(over.get("over_display"))
    runs = int(pd.to_numeric(pd.Series([over.get("over_runs")]), errors = "coerce").fillna(0).iloc[0])
    wickets = int(pd.to_numeric(pd.Series([over.get("over_wickets")]), errors = "coerce").fillna(0).iloc[0])
    run_label = "run" if runs == 1 else "runs"
    wicket_label = "wicket" if wickets == 1 else "wickets"

    swing_value = pd.to_numeric(pd.Series([over.get("win_prob_swing_abs")]), errors = "coerce").iloc[0]
    swing_pct = "-" if pd.isna(swing_value) else f"+{int(round(float(swing_value) * 100))}%"
    beneficiary = str(over.get("beneficiary_team", "") or "").strip() or "-"
    innings_num = pd.to_numeric(pd.Series([over.get("innings")]), errors = "coerce").iloc[0]
    color_key = "innings_1" if pd.isna(innings_num) or int(innings_num) == 1 else "innings_2"
    if match is not None and beneficiary not in {"-", "Even"}:
        match_team_name = getattr(match, "_match_team_name", None)

        def _display_team(value) -> str:
            try:
                return str(match_team_name(value) if callable(match_team_name) else value).strip()
            except Exception:
                return str(value or "").strip()

        beneficiary_key = beneficiary.lower()
        innings_1_team = _display_team(getattr(match, "batting_team", ""))
        innings_2_team = _display_team(getattr(match, "bowling_team", ""))
        if innings_1_team and beneficiary_key == innings_1_team.lower():
            color_key = "innings_1"
        elif innings_2_team and beneficiary_key == innings_2_team.lower():
            color_key = "innings_2"
    return {
        "title": f"Innings {innings}, Over {over_no}",
        "detail": f"({runs} {run_label}, {wickets} {wicket_label})",
        "swing": swing_pct,
        "beneficiary": beneficiary,
        "color": PLOTLY_COLORS.get(color_key, "#3498db"),
    }


def _summary_highlight_card(title: str, children) -> html.Div:
    return html.Div(
        _card(
            [children],
            body_class_name = "p-3",
        ),
        style = {"flex": "1 1 360px", "minWidth": 0},
    )


def _numeric_or_nan(value) -> float:
    num = pd.to_numeric(pd.Series([value]), errors = "coerce").iloc[0]
    return np.nan if pd.isna(num) else float(num)


def _format_signed_impact(value) -> str:
    num = _numeric_or_nan(value)
    if pd.isna(num):
        return "-"
    sign = "+" if num > 0 else ""
    return f"{sign}{num:.2f}"


def _phase_row_range(row: pd.Series) -> tuple[float, float]:
    overs = str(row.get("Overs", "") or "").strip()
    if "-" in overs:
        start_text, end_text = overs.split("-", 1)
        start = _numeric_or_nan(start_text.strip())
        end = _numeric_or_nan(end_text.strip())
        if pd.notna(start) and pd.notna(end):
            return start, end

    phase_name = str(row.get("Phase", "") or "").strip().lower()
    if "power" in phase_name:
        return 1, 6
    if "death" in phase_name:
        return 16, 20
    return 7, 15


def _phase_batting_team(match: Match, innings: int, phase_df: pd.DataFrame) -> str:
    if "Team" in phase_df.columns:
        teams = phase_df["Team"].dropna().astype(str).str.strip()
        teams = teams[teams != ""]
        if not teams.empty:
            return str(teams.iloc[0])

    summary_over_table = getattr(match, "summary_over_table", None)
    if callable(summary_over_table):
        try:
            over_df = summary_over_table()
        except Exception:
            over_df = pd.DataFrame()
        if isinstance(over_df, pd.DataFrame) and not over_df.empty and {"innings", "team"}.issubset(over_df.columns):
            innings_no = pd.to_numeric(over_df["innings"], errors = "coerce")
            teams = over_df.loc[innings_no == int(innings), "team"].dropna().astype(str).str.strip()
            teams = teams[teams != ""]
            if not teams.empty:
                return str(teams.iloc[0])

    attr_name = "batting_team" if int(innings) == 1 else "bowling_team"
    team_name = getattr(match, attr_name, "")
    match_team_name = getattr(match, "_match_team_name", None)
    if callable(match_team_name):
        try:
            team_name = match_team_name(team_name)
        except Exception:
            pass
    return str(team_name or "").strip()


def _phase_batting_impact_from_over_table(
    match: Match,
    innings: int,
    phase_df: pd.DataFrame,
    batting_team: str,
) -> list[float]:
    summary_over_table = getattr(match, "summary_over_table", None)
    if not callable(summary_over_table):
        return [np.nan] * len(phase_df)

    try:
        over_df = summary_over_table()
    except Exception:
        return [np.nan] * len(phase_df)

    if not isinstance(over_df, pd.DataFrame) or over_df.empty or "innings" not in over_df.columns:
        return [np.nan] * len(phase_df)

    over_df = over_df.copy()
    over_df["innings_num"] = pd.to_numeric(over_df["innings"], errors = "coerce")
    over_df = over_df[over_df["innings_num"] == int(innings)].copy()
    if over_df.empty:
        return [np.nan] * len(phase_df)

    if "over_display" in over_df.columns:
        over_numbers = pd.to_numeric(over_df["over_display"], errors = "coerce")
    elif "over" in over_df.columns:
        over_numbers = pd.to_numeric(over_df["over"], errors = "coerce") + 1
    else:
        return [np.nan] * len(phase_df)
    over_df["phase_over_no"] = over_numbers

    impact_values: list[float] = []
    batting_team_key = batting_team.strip().lower()
    for _, phase_row in phase_df.iterrows():
        start_over, end_over = _phase_row_range(phase_row)
        phase_overs = over_df[
            (over_df["phase_over_no"] >= start_over)
            & (over_df["phase_over_no"] <= end_over)
        ].copy()
        if phase_overs.empty:
            impact_values.append(np.nan)
            continue

        if "win_prob_swing" in phase_overs.columns:
            swings = pd.to_numeric(phase_overs["win_prob_swing"], errors = "coerce").dropna()
            if not swings.empty:
                batting_direction = 1 if int(innings) == 2 else -1
                impact_values.append(float(swings.sum()) * batting_direction * 100)
                continue

        if {"win_prob_swing_abs", "beneficiary_team"}.issubset(phase_overs.columns) and batting_team_key:
            swing_abs = pd.to_numeric(phase_overs["win_prob_swing_abs"], errors = "coerce").fillna(0)
            beneficiary = phase_overs["beneficiary_team"].fillna("").astype(str).str.strip().str.lower()
            signed = np.where(
                beneficiary == batting_team_key,
                swing_abs,
                np.where(beneficiary.isin(["", "even"]), 0, -swing_abs),
            )
            impact_values.append(float(np.sum(signed)) * 100)
            continue

        impact_values.append(np.nan)

    return impact_values


def _phase_batting_impact_values(match: Match, innings: int, phase_df: pd.DataFrame, batting_team: str) -> list[float]:
    for col in ["Batting Team Impact", "Net Batting Impact", "Net Impact", "Total Impact"]:
        if col in phase_df.columns:
            return pd.to_numeric(phase_df[col], errors = "coerce").tolist()

    return _phase_batting_impact_from_over_table(match, innings, phase_df, batting_team)


PHASE_SUMMARY_GRID_COLUMNS = "minmax(132px, 1.45fr) minmax(72px, 0.95fr) minmax(56px, 0.72fr) minmax(76px, 1fr)"
PHASE_SUMMARY_LABEL_STYLE = {"fontSize": "11px", "fontWeight": 700, "color": "#6B7C8F", "textTransform": "uppercase"}
PHASE_SUMMARY_CENTER_LABEL_STYLE = {**PHASE_SUMMARY_LABEL_STYLE, "textAlign": "center"}


def _phase_metric_value(value: str, *, value_style: dict | None = None, value_weight: int = 700) -> html.Div:
    return html.Div(
        value,
        style = {
            "fontSize": "18px",
            "fontWeight": value_weight,
            "lineHeight": "1.2",
            "textAlign": "center",
            **(value_style or {}),
        },
    )


def _phase_summary_header_row() -> html.Div:
    return html.Div(
        [
            html.Div("Phase", style = PHASE_SUMMARY_LABEL_STYLE),
            html.Div("Runs/Wkts", style = PHASE_SUMMARY_CENTER_LABEL_STYLE),
            html.Div("RR", style = PHASE_SUMMARY_CENTER_LABEL_STYLE),
            html.Div("Net Impact", style = PHASE_SUMMARY_CENTER_LABEL_STYLE),
        ],
        style = {
            "display": "grid",
            "gridTemplateColumns": PHASE_SUMMARY_GRID_COLUMNS,
            "gap": "10px",
            "alignItems": "end",
        },
    )


def _phase_name_cell(phase_name: str, overs_text: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                phase_name,
                style = {
                    "fontSize": "17px",
                    "fontWeight": 700,
                    "color": "#1B2A38",
                    "lineHeight": "1.2",
                },
            ),
            html.Div(
                overs_text,
                style = {"fontSize": "12px", "fontWeight": 400, "color": "#5D728A", "lineHeight": "1.2", "marginTop": "2px"},
            ) if overs_text else html.Div(),
        ],
        style = {"minWidth": 0},
    )


def _phase_summary_card(match: Match, innings: int) -> html.Div:
    phase_df = match.innings_phase_summary(innings)
    batting_team = _phase_batting_team(match, innings, phase_df)
    impact_values = _phase_batting_impact_values(match, innings, phase_df, batting_team)

    rows = []
    for idx, (_, row) in enumerate(phase_df.iterrows()):
        runs = int(pd.to_numeric(pd.Series([row.get("Runs")]), errors = "coerce").fillna(0).iloc[0])
        wickets = int(pd.to_numeric(pd.Series([row.get("Wickets")]), errors = "coerce").fillna(0).iloc[0])
        impact_value = impact_values[idx] if idx < len(impact_values) else np.nan
        impact_num = _numeric_or_nan(impact_value)
        impact_color = _team_identity_delta_color(impact_num)
        phase_name = str(row.get("Phase", "") or "")
        overs = str(row.get("Overs", "") or "").strip()
        overs_text = f"Overs {overs}" if overs else ""
        rows.append(
            html.Div(
                [
                    _phase_summary_header_row(),
                    html.Div(
                        [
                            _phase_name_cell(phase_name, overs_text),
                            _phase_metric_value(f"{runs}/{wickets}", value_weight = 400),
                            _phase_metric_value(_safe_float_text(row.get("RR"), digits = 2), value_weight = 400),
                            _phase_metric_value(
                                _format_signed_impact(impact_value),
                                value_style = {"color": impact_color},
                            ),
                        ],
                        style = {
                            "display": "grid",
                            "gridTemplateColumns": PHASE_SUMMARY_GRID_COLUMNS,
                            "gap": "10px",
                            "alignItems": "center",
                        },
                    ),
                ],
                style = {
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "10px",
                    "justifyContent": "center",
                    "minHeight": "92px",
                    "padding": "14px 12px",
                    "border": "1px solid #E4EAF1",
                    "borderRadius": "8px",
                    "backgroundColor": "#F8FAFC",
                },
            )
        )

    batting_team_header = (
        _team_text_with_logo(
            batting_team,
            html.Div(
                batting_team,
                style = {"fontSize": "15px", "fontWeight": 600, "color": "#5D728A", "lineHeight": "1.2"},
            ),
            "team-logo-inline phase-summary-team-logo-inline",
        )
        if batting_team
        else html.Div(
            "Batting team",
            style = {"fontSize": "15px", "fontWeight": 600, "color": "#5D728A", "lineHeight": "1.2"},
        )
    )

    return _card(
        [
            html.Div(
                [
                    html.Div(f"Phase Summary (Innings {innings})", style = {"fontWeight": 700, "fontSize": "18px"}),
                    html.Div(batting_team_header, style = {"marginTop": "2px"}),
                ],
                style = {"marginBottom": "10px"},
            ),
            html.Div(
                rows,
                style = {"display": "flex", "flexDirection": "column", "gap": "8px"},
            ),
        ],
        body_class_name = "p-3",
    )


def _summary_graph_card(figure) -> html.Div:
    figure.update_layout(
        width = None,
        height = 400,
        margin = {"l": 44, "r": 16, "t": 48, "b": 34},
    )
    return _card(
        [
            dcc.Graph(
                figure = figure,
                config = {"displayModeBar": False},
                style = {"height": "100%"},
            )
        ],
        body_class_name = "p-2",
    )


def _build_innings_summary_card(match: Match, innings: int) -> html.Div:
    runs_fig = match.summary_runs_by_over_graph(innings = innings)
    rate_fig = match.projected_score_by_over_graph() if innings == 1 else match.chase_run_rate_by_over_graph()

    return html.Div(
        children = [
            html.Div(
                [
                    html.Div(_phase_summary_card(match, innings), style = {"flex": "1 1 0", "minWidth": "280px"}),
                    html.Div(_summary_graph_card(runs_fig), style = {"flex": "1 1 0", "minWidth": "280px"}),
                    html.Div(_summary_graph_card(rate_fig), style = {"flex": "1 1 0", "minWidth": "280px"}),
                ],
                style = {
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "stretch",
                    "flexWrap": "wrap",
                },
            ),
        ],
        style = {
            "backgroundColor": "transparent",
            "border": "none",
            "borderRadius": 0,
            "padding": "0",
        },
    )


def _build_summary_tab(match: Match) -> html.Div:
    player = _most_impactful_player_summary(match)
    over = _format_most_impactful_over(match.most_impactful_over(), match)

    if player is None:
        player_children = html.Div("No impact values available.", style = {"fontSize": "15px"})
    else:
        player_children = html.Div(
            [
                html.Div(
                    [
                        html.Div("Most Impactful Player", style = {"fontSize": "12px", "fontWeight": 700, "color": "#5D728A", "marginBottom": "6px"}),
                        html.Div(player["player"], style = {"fontSize": "18px", "fontWeight": 700, "lineHeight": "1.2"}),
                        html.Div(player["team"], style = {"fontSize": "14px", "color": "#5D728A", "marginTop": "2px"}),
                    ],
                    style = {"flex": "0 0 calc(50% - 6px)", "minWidth": 0},
                ),
                html.Div(
                    [
                        html.Div(
                            player["total_impact"],
                            style = {
                                "fontWeight": 700,
                                "fontSize": "26px",
                                "lineHeight": "1.05",
                                "color": _team_identity_delta_color(player["total_impact_value"]),
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Div(player["scoreline"], style = {"fontSize": "15px", "marginTop": "4px"}),
                    ],
                    style = {
                        "display": "flex",
                        "flex": "0 0 calc(50% - 6px)",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "minWidth": 0,
                        "textAlign": "center",
                    },
                ),
            ],
            style = {"display": "flex", "gap": "12px", "alignItems": "center", "justifyContent": "space-between"},
        )

    over_children = html.Div(
        [
            html.Div(
                [
                    html.Div("Most Impactful Over", style = {"fontSize": "12px", "fontWeight": 700, "color": "#5D728A", "marginBottom": "6px"}),
                    html.Div(over["title"], style = {"fontSize": "18px", "fontWeight": 700, "lineHeight": "1.2"}),
                    html.Div(over["detail"], style = {"fontSize": "15px", "color": "#5D728A", "marginTop": "6px"}),
                ],
                style = {"flex": "0 0 calc(50% - 6px)", "minWidth": 0},
            ),
            html.Div(
                [
                    html.Div(
                        over["swing"],
                        style = {
                            "fontWeight": 700,
                            "fontSize": "26px",
                            "lineHeight": "1.05",
                            "color": over["color"],
                            "whiteSpace": "nowrap",
                        },
                    ),
                    html.Div(over["beneficiary"], style = {"fontSize": "14px", "color": "#5D728A", "marginTop": "4px"}),
                ],
                style = {
                    "display": "flex",
                    "flex": "0 0 calc(50% - 6px)",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "minWidth": 0,
                    "textAlign": "center",
                },
            ),
        ],
        style = {"display": "flex", "gap": "12px", "alignItems": "center", "justifyContent": "space-between"},
    )

    return html.Div(
        [
            html.Div(
                [
                    _summary_highlight_card("Most Impactful Player", player_children),
                    _summary_highlight_card("Most Impactful Over", over_children),
                ],
                style = {
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "stretch",
                    "flexWrap": "wrap",
                    "marginTop": "10px",
                },
            ),
            html.Div(
                [
                    _build_innings_summary_card(match, innings = 1),
                    _build_innings_summary_card(match, innings = 2),
                ],
                style = {"display": "flex", "flexDirection": "column", "gap": "12px", "marginTop": "12px"},
            ),
        ]
    )


def _analysis_tabs_component() -> dcc.Tabs:
    return dcc.Tabs(
        id = "analysis-tabs",
        value = "summary",
        children = [
            dcc.Tab(label = "Summary", value = "summary"),
            dcc.Tab(label = "Play-by-Play", value = "play-by-play"),
            dcc.Tab(label = "Scorecard", value = "scorecard"),
            dcc.Tab(label = "Total Impact", value = "total-impact"),
        ],
    )


def _analysis_tab_labels(tabs_component: dcc.Tabs | None = None) -> list[str]:
    tabs = tabs_component if tabs_component is not None else _analysis_tabs_component()
    return [str(child.label) for child in tabs.children]


def _analysis_tab_default_value(tabs_component: dcc.Tabs | None = None) -> str:
    tabs = tabs_component if tabs_component is not None else _analysis_tabs_component()
    return str(tabs.value)


def _build_dashboard_shell(match: Match) -> html.Div:
    summary = match.match_summary()

    result_value = summary["result_text"]
    status_clean = str(getattr(match, "status", "")).strip().lower()
    current_innings = int(summary.get("current_innings") or 1)

    score_status_text = str(result_value)
    if status_clean == "innings_break":
        score_status_text = "Innings Break"
    elif status_clean == "delayed":
        delayed_detail = "" if getattr(match, "status_detail", None) is None else str(getattr(match, "status_detail")).strip()
        score_status_text = delayed_detail if delayed_detail else "Match delayed"
    elif status_clean not in {"complete", "abandoned", "no_result", "delayed"} and current_innings <= 1:
        toss_winner = None
        toss_decision = None
        match_info_row = getattr(match, "match_info_row", None)
        if match_info_row is not None:
            winner_raw = match_info_row.get("toss_winner")
            decision_raw = match_info_row.get("toss_decision")
            if winner_raw is not None:
                winner_text = str(winner_raw).strip()
                toss_winner = winner_text if winner_text else None
            if decision_raw is not None:
                decision_text = str(decision_raw).strip().lower()
                toss_decision = decision_text if decision_text else None
        if toss_winner and toss_decision:
            decision_display = "field" if toss_decision in {"bowl", "field"} else "bat"
            score_status_text = f"{toss_winner} chose to {decision_display}"

    score_card_color = "primary"
    status_detail_clean = "" if getattr(match, "status_detail", None) is None else str(getattr(match, "status_detail")).strip().lower()
    stop_signal_text = f"{status_detail_clean} {str(result_value).strip().lower()}"
    if status_clean in {"abandoned", "no_result"}:
        score_card_color = "danger"
    elif status_clean == "delayed" or any(token in stop_signal_text for token in ["stopped", "suspended", "interrupted", "rain delay", "delay"]):
        score_card_color = "warning"
    elif status_clean in {"live", "innings_break"}:
        score_card_color = "info"

    team1 = str(summary.get("team1", "") or "").strip()
    team2 = str(summary.get("team2", "") or "").strip()
    winner = "" if getattr(match, "match_won_by", None) is None else str(getattr(match, "match_won_by")).strip()
    winner_lower = winner.lower()
    result_decided = status_clean == "complete" and winner and winner_lower not in {"", "unknown"}

    muted_team: str | None = None
    if result_decided:
        if winner == team1:
            muted_team = team2
        elif winner == team2:
            muted_team = team1
    elif status_clean in {"abandoned", "no_result"}:
        runs1 = int(summary.get("innings1", {}).get("runs", 0) or 0)
        runs2 = int(summary.get("innings2", {}).get("runs", 0) or 0)
        if runs1 > runs2:
            muted_team = team2
        elif runs2 > runs1:
            muted_team = team1
        else:
            muted_team = team2 if team2 else None

    win_fig = match.predict_smooth()
    worm_fig = match.worm()

    win_fig.update_layout(
        width = None,
        height = 450,
        margin = {"l": 45, "r": 20, "t": 55, "b": 35},
    )
    worm_fig.update_layout(
        width = None,
        height = 450,
        margin = {"l": 35, "r": 20, "t": 55, "b": 35},
    )

    balls_sorted = match.balls.copy()
    if not balls_sorted.empty:
        sort_cols = [col for col in ['innings', 'team_balls', 'over', 'ball', 'id'] if col in balls_sorted.columns]
        if sort_cols:
            balls_sorted = balls_sorted.sort_values(sort_cols)
    if current_innings >= 2:
        prob_source = balls_sorted.loc[balls_sorted['innings'] == 2].copy()
    else:
        prob_source = balls_sorted.loc[balls_sorted['innings'] == 1].copy()

    terminal_second = None
    terminal_probability = getattr(match, "_second_innings_terminal_probability", None)
    if callable(terminal_probability):
        try:
            terminal_second = terminal_probability()
        except Exception:
            terminal_second = None

    if terminal_second is not None:
        _, terminal_prob = terminal_second
        team2_prob_pct = int(round(float(np.clip(terminal_prob, 0, 1)) * 100))
    elif prob_source.empty or 'y_prob' not in prob_source.columns:
        team2_prob_pct = 50
    else:
        y_series = pd.to_numeric(prob_source['y_prob'], errors = 'coerce').dropna()
        if y_series.empty:
            team2_prob_pct = 50
        else:
            y_smooth_last = y_series.rolling(window = 3, min_periods = 1).mean().iloc[-1]
            team2_prob_pct = int(round(float(np.clip(y_smooth_last, 0, 1)) * 100))
    team1_prob_pct = max(0, min(100, 100 - int(team2_prob_pct)))
    team2_prob_pct = max(0, min(100, int(team2_prob_pct)))

    if current_innings >= 2 or status_clean == "innings_break":
        first = summary.get("innings1", {})
        second = summary.get("innings2", {})
        target = match._target(int(first.get("runs", 0) or 0))
        second_runs = int(second.get("runs", 0) or 0)
        second_balls = int(second.get("balls", 0) or 0)
        balls_limit = match._innings_ball_limit(innings = 2)
        runs_needed = max(0, int(target) - second_runs)
        balls_left = max(0, int(balls_limit) - second_balls)
        if balls_left > 0:
            required_rr_text = f"{runs_needed / (balls_left / 6):.2f}"
        elif runs_needed == 0:
            required_rr_text = "0.00"
        else:
            required_rr_text = "-"
        second_crr = "-" if second_balls <= 0 else f"{second_runs / (second_balls / 6):.2f}"
        secondary_rate_label = "Required Run Rate:"
        secondary_rate_value = f"{required_rr_text} (Current: {second_crr})"
    else:
        secondary_rate_label = "Projected Score:"
        secondary_rate_value = ""
        first_inn = match.balls[match.balls['innings'] == 1].copy()
        if not first_inn.empty and all(
            c in first_inn.columns for c in ('wickets_remaining', 'balls_remaining', 'team_runs')
        ):
            sort_cols = [c for c in ['team_balls', 'over', 'ball', 'id'] if c in first_inn.columns]
            if sort_cols:
                first_inn = first_inn.sort_values(sort_cols)
            last_row = first_inn.iloc[[-1]].copy()
            last_row['wickets_remaining'] = (
                pd.to_numeric(last_row['wickets_remaining'], errors='coerce').fillna(10).astype(int).clip(0, 10)
            )
            last_row['balls_remaining'] = (
                pd.to_numeric(last_row['balls_remaining'], errors='coerce').fillna(0).astype(float)
            )
            team_runs_val = float(pd.to_numeric(last_row['team_runs'], errors='coerce').fillna(0).iloc[0])
            resource_val = float(ipl.resource_function(last_row, resource_params)[0])
            secondary_rate_value = str(int(round(team_runs_val + resource_val)))

    return html.Div(
        children=[
            html.Div(
                children=[
                    _score_summary_card(
                        summary,
                        score_status_text,
                        card_color=score_card_color,
                        muted_team=muted_team,
                    ),
                    _info_summary_card(summary),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "10px",
                    "marginBottom": "14px",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            _card(
                                [
                                    dcc.Graph(
                                        figure = win_fig,
                                        config = {"displayModeBar": False},
                                    ),
                                ],
                                body_class_name = "p-2",
                            ),
                        ],
                        style={
                            "flex": "2 1 0",
                            "minWidth": 0,
                        },
                    ),
                    html.Div(
                        children=[
                            _card(
                                [
                                    dcc.Graph(
                                        figure = worm_fig,
                                        config = {"displayModeBar": False},
                                    ),
                                ],
                                body_class_name = "p-2",
                            ),
                        ],
                        style={
                            "flex": "1 1 0",
                            "minWidth": 0,
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "stretch",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            _card(
                                [
                                    html.Div(
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div(
                                                        [
                                                            _team_logo_img(summary["team1"]),
                                                            html.Div(
                                                                [
                                                                    html.Div(f"{summary['team1']}:", style={"fontSize": "24px", "fontWeight": 400, "lineHeight": "1.1"}),
                                                                    html.Div(
                                                                        f"{team1_prob_pct}%",
                                                                        style={
                                                                            "fontSize": "26px",
                                                                            "fontWeight": 700,
                                                                            "lineHeight": "1.1",
                                                                            "marginTop": "2px",
                                                                            "color": PLOTLY_COLORS["innings_1"],
                                                                        },
                                                                    ),
                                                                ],
                                                                style={"minWidth": 0, "textAlign": "center"},
                                                            ),
                                                        ],
                                                        className = "team-logo-inline win-prob-team-logo-inline",
                                                    ),
                                                ],
                                                style={
                                                    "width": "50%",
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                    "textAlign": "center",
                                                },
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div(
                                                        [
                                                            _team_logo_img(summary["team2"]),
                                                            html.Div(
                                                                [
                                                                    html.Div(f"{summary['team2']}:", style={"fontSize": "24px", "fontWeight": 400, "lineHeight": "1.1"}),
                                                                    html.Div(
                                                                        f"{team2_prob_pct}%",
                                                                        style={
                                                                            "fontSize": "26px",
                                                                            "fontWeight": 700,
                                                                            "lineHeight": "1.1",
                                                                            "marginTop": "2px",
                                                                            "color": PLOTLY_COLORS["innings_2"],
                                                                        },
                                                                    ),
                                                                ],
                                                                style={"minWidth": 0, "textAlign": "center"},
                                                            ),
                                                        ],
                                                        className = "team-logo-inline win-prob-team-logo-inline",
                                                    ),
                                                ],
                                                style={
                                                    "width": "50%",
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                    "textAlign": "center",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "justifyContent": "space-between",
                                            "alignItems": "center",
                                            "height": "100%",
                                        },
                                    ),
                                ],
                                body_class_name = "p-3",
                            ),
                        ],
                        style={
                            "flex": "2 1 0",
                            "minWidth": 0,
                            "height": "100px",
                        },
                    ),
                    html.Div(
                        children=[
                            _card(
                                [
                                    html.Div(
                                        children=[
                                            html.Div(
                                                children=(
                                                    [
                                                        html.Span(f"{secondary_rate_label} "),
                                                        html.Span(secondary_rate_value, style={"fontWeight": 700}),
                                                    ]
                                                    if current_innings < 2 and status_clean != "innings_break"
                                                    else f"{secondary_rate_label} {secondary_rate_value}"
                                                ),
                                                style={"fontSize": "24px", "fontWeight": 400, "lineHeight": "1.1", "textAlign": "center"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "justifyContent": "center",
                                            "alignItems": "center",
                                            "height": "100%",
                                        },
                                    ),
                                ],
                                body_class_name = "p-3",
                            ),
                        ],
                        style={
                            "flex": "1 1 0",
                            "minWidth": 0,
                            "height": "100px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "stretch",
                    "marginTop": "10px",
                },
            ),
            html.Div(
                children=[
                    _analysis_tabs_component(),
                    html.Div(id = "analysis-tab-content", style = {"paddingTop": "6px", "minHeight": "400px"}),
                ],
                style={"marginTop": "10px"},
            ),
        ]
    )


@app.callback(
    Output("nav-container", "children"),
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("url", "search"),
)
def render_page(pathname: str | None, url_search: str | None = None):
    nav = build_navbar(pathname)

    if pathname in (None, "", "/"):
        return nav, _build_landing_page()

    if pathname == "/match-analysis":
        year_options = get_year_options()
        team_options = get_team_options_for_year(year=None)
        return nav, build_match_analysis_page(year_options, team_options)

    if pathname == "/season-leaderboard":
        season_options = get_finalized_season_options()
        default_season = get_default_leaderboard_season()
        return nav, build_season_leaderboard_page(season_options, default_season)

    if pathname == "/team-analysis":
        season_options = get_finalized_season_options()
        default_season = get_default_leaderboard_season()
        requested_team = None
        if url_search:
            params = parse_qs(url_search.lstrip("?"))
            season_values = params.get("season", [])
            team_values = params.get("team", [])
            if season_values:
                try:
                    requested_season = int(season_values[0])
                    valid_seasons = {option["value"] for option in season_options}
                    if requested_season in valid_seasons:
                        default_season = requested_season
                except (TypeError, ValueError):
                    pass
            if team_values:
                requested_team = str(team_values[0])
        team_options, default_team = _team_analysis_options_and_default(default_season, requested_team)
        return nav, build_team_analysis_page(season_options, default_season, team_options, default_team)

    if pathname == "/about":
        return nav, build_about_page()

    not_found = html.Div(
        children=[
            html.H2("Page not found", style={"margin": "0 0 8px"}),
            html.P("Use the top navigation to choose a page.", style={"margin": 0}),
        ],
        style={"padding": "24px"},
    )
    return nav, not_found


@app.callback(
    Output("team-filter-dropdown", "options"),
    Output("team-filter-dropdown", "value"),
    Input("year-dropdown", "value"),
    State("team-filter-dropdown", "value"),
)
def update_team_options(year: int | None, current_team: str | None):
    options = get_team_options_for_year(year=year)
    valid_teams = {option["value"] for option in options}
    if current_team in valid_teams:
        return options, current_team
    return options, None


@app.callback(
    Output("match-dropdown", "options"),
    Output("match-dropdown", "value"),
    Input("year-dropdown", "value"),
    Input("team-filter-dropdown", "value"),
    Input("url", "search"),
    State("match-dropdown", "value"),
)
def update_match_options(
    year: int | None,
    team: str | None,
    url_search: str | None,
    current_match_id: int | None,
):
    options = get_match_options(year=year, team=team)
    if not options:
        return [], None

    valid_ids = {option["value"] for option in options}
    if url_search:
        params = parse_qs(url_search.lstrip("?"))
        match_id_values = params.get("match_id", [])
        if match_id_values:
            try:
                requested_match_id = int(match_id_values[0])
                if requested_match_id in valid_ids:
                    return options, requested_match_id
            except (TypeError, ValueError):
                pass

    if current_match_id in valid_ids:
        return options, current_match_id

    return options, options[0]["value"]


@app.callback(
    Output("match-id-inline", "children"),
    Input("match-dropdown", "value"),
)
def update_match_id_inline(match_id: int | None):
    if match_id is None:
        return "Match ID: -"
    try:
        return f"Match ID: {int(match_id)}"
    except (TypeError, ValueError):
        return f"Match ID: {match_id}"


@app.callback(
    Output("selected-match-store", "data"),
    Output("auto-refresh-checkbox", "value"),
    Input("match-dropdown", "value"),
    Input("auto-refresh-checkbox", "value"),
    State("selected-match-store", "data"),
)
def persist_selected_match(
    match_id: int | None,
    auto_refresh_checkbox_value: list[int] | None,
    selected_store: dict | None,
):
    triggered_id = ctx.triggered_id
    if match_id is None:
        return {"match_id": None, "auto_refresh": False}, []

    match_id_int = int(match_id)
    existing_match_id = None
    existing_auto_refresh = False
    if isinstance(selected_store, dict):
        raw_match_id = selected_store.get("match_id")
        if raw_match_id is not None:
            try:
                existing_match_id = int(raw_match_id)
            except (TypeError, ValueError):
                existing_match_id = None
        existing_auto_refresh = bool(selected_store.get("auto_refresh", False))

    checkbox_enabled = bool(auto_refresh_checkbox_value)
    if triggered_id == "auto-refresh-checkbox" and existing_match_id == match_id_int:
        auto_refresh_enabled = checkbox_enabled
    else:
        auto_refresh_enabled = _match_needs_live_reload(match_id_int)

    return {"match_id": match_id_int, "auto_refresh": auto_refresh_enabled}, ([1] if auto_refresh_enabled else [])


@app.callback(
    Output("match-dashboard", "children"),
    Input("selected-match-store", "data"),
)
def render_match_dashboard(selected_match: dict | None):
    if not selected_match or selected_match.get("match_id") is None:
        return html.Div("Select a match to load the dashboard.", style={"padding": "8px 0"})

    match_id = int(selected_match["match_id"])
    try:
        cache_token = _match_cache_token(match_id)
        match = load_match(match_id, cache_token)
    except Exception as exc:
        return html.Div(f"Could not load match {match_id}: {exc}", style={"padding": "8px 0"})

    return _build_dashboard_shell(match)


@app.callback(
    Output("analysis-tab-content", "children"),
    Input("analysis-tabs", "value"),
    Input("selected-match-store", "data"),
    Input("playbyplay-desc-page-store", "data"),
    Input("playbyplay-ribbon-page-store", "data"),
)
def render_dashboard_tab(
    tab_value: str | None,
    selected_match: dict | None,
    playbyplay_desc_page: int | None,
    playbyplay_ribbon_page: int | None,
):
    if not selected_match or selected_match.get("match_id") is None:
        return html.Div()

    match_id = int(selected_match["match_id"])
    try:
        cache_token = _match_cache_token(match_id)
        match = load_match(match_id, cache_token)
    except Exception as exc:
        return html.Div(f"Could not load match {match_id}: {exc}", style={"padding": "8px 0"})

    if tab_value in {None, "summary"}:
        return _build_summary_tab(match)

    if tab_value == "play-by-play":
        return _build_play_by_play_tab(
            match,
            desc_page = int(playbyplay_desc_page or 1),
            ribbon_page = int(playbyplay_ribbon_page or 1),
            desc_overs_per_page = PLAYBYPLAY_DESC_OVERS_PER_PAGE,
            ribbon_overs_per_page = PLAYBYPLAY_RIBBON_OVERS_PER_PAGE,
        )

    if tab_value == "total-impact":
        return _build_total_impact_tab(match)

    summary = match.match_summary()
    return _build_scorecard_tab(match, summary)


@app.callback(
    Output("playbyplay-desc-page-store", "data"),
    Output("playbyplay-ribbon-page-store", "data"),
    Input("selected-match-store", "data"),
)
def reset_playbyplay_stores(_selected_match: dict | None):
    return 1, 1


@app.callback(
    Output("playbyplay-desc-page-store", "data", allow_duplicate = True),
    Input({"type": "playbyplay-desc-pagination", "name": ALL}, "active_page"),
    State("playbyplay-desc-page-store", "data"),
    prevent_initial_call = True,
)
def set_playbyplay_desc_page(active_pages: list[int | None], current_page: int | None):
    for page in active_pages:
        if page is not None:
            return int(page)
    return int(current_page or 1)


@app.callback(
    Output("playbyplay-ribbon-page-store", "data", allow_duplicate = True),
    Input({"type": "playbyplay-ribbon-arrow", "dir": ALL}, "n_clicks"),
    State("playbyplay-ribbon-page-store", "data"),
    State("selected-match-store", "data"),
    prevent_initial_call = True,
)
def set_playbyplay_ribbon_page(
    _arrow_clicks: list[int | None],
    current_page: int | None,
    selected_match: dict | None,
):
    if not selected_match or selected_match.get("match_id") is None:
        return int(current_page or 1)

    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        return int(current_page or 1)

    direction = str(triggered.get("dir") or "").strip().lower()
    if direction not in {"left", "right"}:
        return int(current_page or 1)

    match_id = int(selected_match["match_id"])
    try:
        cache_token = _match_cache_token(match_id)
        match = load_match(match_id, cache_token)
        total_pages = _playbyplay_total_pages(
            len(match.play_by_play_feed()),
            overs_per_page = PLAYBYPLAY_RIBBON_OVERS_PER_PAGE,
        )
    except Exception:
        total_pages = 1

    page = int(current_page or 1)
    if direction == "left":
        page -= 1
    else:
        page += 1

    page = max(1, min(int(page), int(total_pages)))
    return page


app.clientside_callback(
    """
    function(_nIntervals, selectedMatch, pathname) {
        if (pathname !== "/match-analysis") {
            return "";
        }
        if (!selectedMatch || !selectedMatch.match_id) {
            return "";
        }
        if (Boolean(selectedMatch.auto_refresh)) {
            window.location.reload();
        }
        return "";
    }
    """,
    Output("refresh-client-hook", "children"),
    Input("live-refresh-interval", "n_intervals"),
    State("selected-match-store", "data"),
    State("url", "pathname"),
    prevent_initial_call=True,
)


@app.callback(
    Output("season-leaderboard-content", "children"),
    Input("leaderboard-season-dropdown", "value"),
)
def render_season_leaderboard_dashboard(season_value: int | None):
    season = season_value if season_value is not None else get_default_leaderboard_season()
    if season is None:
        return html.Div("No finalized season data available.", style={"padding": "8px 0"})

    try:
        cache_token = _leaderboard_cache_token(int(season))
        leaderboard = load_leaderboard_overview(int(season), cache_token)
    except Exception as exc:
        return html.Div(f"Could not load season leaderboard for {season}: {exc}", style={"padding": "8px 0"})

    return _build_season_leaderboard_dashboard(leaderboard)


@app.callback(
    Output("season-leaderboard-tab-content", "children"),
    Input("season-leaderboard-tabs", "value"),
    Input("leaderboard-season-dropdown", "value"),
)
def render_season_leaderboard_tab(tab_value: str | None, season_value: int | None):
    season = season_value if season_value is not None else get_default_leaderboard_season()
    if season is None:
        return html.Div()

    if tab_value in {None, "overall-strength"}:
        try:
            cache_token = _leaderboard_cache_token(int(season))
            leaderboard = load_leaderboard_overview(int(season), cache_token)
        except Exception as exc:
            return html.Div(f"Could not load season leaderboard for {season}: {exc}", style={"padding": "8px 0"})
        return _build_leaderboard_overall_strength_tab(leaderboard)

    try:
        cache_token = _leaderboard_cache_token(int(season))
        leaderboard = load_leaderboard(int(season), cache_token)
    except Exception as exc:
        return html.Div(f"Could not load season leaderboard for {season}: {exc}", style={"padding": "8px 0"})

    if tab_value == "bowling":
        return _build_leaderboard_bowling_tab(leaderboard)

    return _build_leaderboard_batting_tab(leaderboard)


@app.callback(
    Output("overall-strength-chart", "figure"),
    Output("overall-strength-metric-description", "children"),
    Input("overall-strength-metric-dropdown", "value"),
    Input("leaderboard-season-dropdown", "value"),
)
def update_overall_strength_chart(selected_metric: str | None, season_value: int | None):
    season = season_value if season_value is not None else get_default_leaderboard_season()
    if season is None:
        empty_df = pd.DataFrame()
        return _overall_strength_figure(empty_df, selected_metric), _overall_strength_side_panel(empty_df, selected_metric, None)

    try:
        cache_token = _leaderboard_cache_token(int(season))
        leaderboard = load_leaderboard_overview(int(season), cache_token)
        strength_df = _team_strength_percentile_rows(leaderboard)
    except Exception:
        strength_df = pd.DataFrame()
    return _overall_strength_figure(strength_df, selected_metric), _overall_strength_side_panel(strength_df, selected_metric, int(season))


@app.callback(
    Output("team-analysis-team-dropdown", "options"),
    Output("team-analysis-team-dropdown", "value"),
    Input("team-analysis-season-dropdown", "value"),
    State("team-analysis-team-dropdown", "value"),
)
def update_team_analysis_team_options(season_value: int | None, current_team: str | None):
    season = season_value if season_value is not None else get_default_leaderboard_season()
    return _team_analysis_options_and_default(season, current_team)


@app.callback(
    Output("team-analysis-header", "children"),
    Output("team-analysis-content", "children"),
    Input("team-analysis-season-dropdown", "value"),
    Input("team-analysis-team-dropdown", "value"),
)
def render_team_analysis_dashboard(season_value: int | None, team_value: str | None):
    season = season_value if season_value is not None else get_default_leaderboard_season()
    if season is None:
        return (
            html.Div(
                [
                    html.H2("Team Analysis", style={"margin": "0 0 8px"}),
                    html.P("No finalized season data available.", style={"margin": 0}),
                ]
            ),
            html.Div("No finalized season data available.", style={"padding": "8px 0"}),
        )

    try:
        leaderboard = _load_team_analysis_leaderboard(int(season))
    except Exception as exc:
        return (
            html.Div(
                [
                    html.H2("Team Analysis", style={"margin": "0 0 8px"}),
                    html.P(f"Season {int(season)}", style={"margin": 0}),
                ]
            ),
            html.Div(f"Could not load team analysis for {season}: {exc}", style={"padding": "8px 0"}),
        )

    options, default_team = _team_analysis_options_and_default(int(season), team_value)
    selected_team = team_value if team_value in {option["value"] for option in options} else default_team
    if not selected_team:
        return (
            html.Div(
                [
                    html.H2("Team Analysis", style={"margin": "0 0 8px"}),
                    html.P(f"Season {int(season)}", style={"margin": 0}),
                ]
            ),
            html.Div("No teams available for this season.", style={"padding": "8px 0"}),
        )

    selected_team = str(selected_team)
    return _team_analysis_header(leaderboard, selected_team, int(season)), _build_team_analysis_dashboard(leaderboard, selected_team)


if __name__ == "__main__":
    app.run(debug=True)
