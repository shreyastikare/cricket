from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


PHASE_COLUMNS = [
    "PP Batting",
    "Middle Batting",
    "Death Batting",
    "PP Bowling",
    "Middle Bowling",
    "Death Bowling",
]

BATTING_ROLE_ORDER = ["Top order", "Middle order", "Finishers"]
BOWLING_PHASE_ORDER = ["Powerplay", "Middle overs", "Death overs"]

IDENTITY_CATEGORY_ORDER = [
    "Top-order batting impact",
    "Middle-order batting impact",
    "Finisher batting impact",
    "Powerplay bowling impact",
    "Middle-overs bowling impact",
    "Death-overs bowling impact",
]

BOWLING_PRESSURE_HELP = (
    "Bowling Pressure is non-wicket bowling impact from balls where the team did not take a wicket."
)

PROFILE_METRICS = [
    {
        "metric_key": "pp_batting",
        "metric_name": "Powerplay batting",
        "profile": "batting",
        "category": "Batting - Phase",
        "display_order": 1,
        "description": "",
    },
    {
        "metric_key": "middle_batting",
        "metric_name": "Middle-overs batting",
        "profile": "batting",
        "category": "Batting - Phase",
        "display_order": 2,
        "description": "",
    },
    {
        "metric_key": "death_batting",
        "metric_name": "Death-overs batting",
        "profile": "batting",
        "category": "Batting - Phase",
        "display_order": 3,
        "description": "",
    },
    {
        "metric_key": "top_order_batting",
        "metric_name": "Top-order batting",
        "profile": "batting",
        "category": "Batting - Role",
        "display_order": 4,
        "description": "",
    },
    {
        "metric_key": "middle_order_batting",
        "metric_name": "Middle-order batting",
        "profile": "batting",
        "category": "Batting - Role",
        "display_order": 5,
        "description": "",
    },
    {
        "metric_key": "finisher_batting",
        "metric_name": "Finisher batting",
        "profile": "batting",
        "category": "Batting - Role",
        "display_order": 6,
        "description": "",
    },
    {
        "metric_key": "pp_bowling",
        "metric_name": "Powerplay bowling",
        "profile": "bowling",
        "category": "Bowling - Phase",
        "display_order": 7,
        "description": "",
    },
    {
        "metric_key": "middle_bowling",
        "metric_name": "Middle-overs bowling",
        "profile": "bowling",
        "category": "Bowling - Phase",
        "display_order": 8,
        "description": "",
    },
    {
        "metric_key": "death_bowling",
        "metric_name": "Death-overs bowling",
        "profile": "bowling",
        "category": "Bowling - Phase",
        "display_order": 9,
        "description": "",
    },
    {
        "metric_key": "wicket_taking_bowling",
        "metric_name": "Wicket-taking",
        "profile": "bowling",
        "category": "Bowling - Wickets",
        "display_order": 10,
        "description": "",
    },
    {
        "metric_key": "bowling_pressure",
        "metric_name": "Bowling Pressure",
        "profile": "bowling",
        "category": "Bowling - Pressure",
        "display_order": 11,
        "description": BOWLING_PRESSURE_HELP,
    },
]

PROFILE_METRIC_SUBTITLES = {
    "pp_batting": "Overs 1-6",
    "middle_batting": "Overs 7-15",
    "death_batting": "Overs 16-20",
    "top_order_batting": "Positions 1-3",
    "middle_order_batting": "Positions 4-6",
    "finisher_batting": "Positions 7+",
    "pp_bowling": "Overs 1-6",
    "middle_bowling": "Overs 7-15",
    "death_bowling": "Overs 16-20",
    "wicket_taking_bowling": "Impact from wickets",
    "bowling_pressure": "Non-wicket impact",
}

TEAM_PHASE_HEATMAP_MIN_HEIGHT = 360
TEAM_PHASE_HEATMAP_HEADER_HEIGHT = 170
TEAM_PHASE_HEATMAP_ROW_HEIGHT = 34
TEAM_PHASE_HEATMAP_MAX_HEIGHT = 760

BENCHMARK_START_SEASON = 2008
BENCHMARK_END_SEASON = 2025

TEAM_ABBREVIATIONS = {
    "Chennai Super Kings": "CSK",
    "Deccan Chargers": "DCG",
    "Delhi Capitals": "DC",
    "Delhi Daredevils": "DD",
    "Gujarat Lions": "GL",
    "Gujarat Titans": "GT",
    "Kings XI Punjab": "KXIP",
    "Kochi Tuskers Kerala": "KTK",
    "Kolkata Knight Riders": "KKR",
    "Lucknow Super Giants": "LSG",
    "Mumbai Indians": "MI",
    "Pune Warriors": "PWI",
    "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR",
    "Rising Pune Supergiant": "RPS",
    "Rising Pune Supergiants": "RPS",
    "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB",
    "Sunrisers Hyderabad": "SRH",
}


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_key(value: Any) -> str:
    return _norm_text(value).lower()


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return float(default)
    return float(parsed)


def _safe_int(value: Any, default: int = 0) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return int(default)
    return int(parsed)


def _match_count(matches: pd.DataFrame) -> int:
    if matches.empty or "match_id" not in matches.columns:
        return 0
    return int(pd.to_numeric(matches["match_id"], errors="coerce").dropna().nunique())


def _team_mask(frame: pd.DataFrame, column: str, team: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna("").astype(str).str.strip().str.lower().eq(_norm_key(team))


def filter_team_season_matches(season_matches: pd.DataFrame, team: str | None) -> pd.DataFrame:
    if season_matches.empty or not team:
        return season_matches.iloc[0:0].copy()

    mask = _team_mask(season_matches, "bat_first", team) | _team_mask(season_matches, "bowl_first", team)
    out = season_matches.loc[mask].copy()
    sort_cols = [col for col in ["date", "match_id"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort")
    return out.reset_index(drop=True)


def phase_label(over: Any) -> str | None:
    over_num = _safe_float(over, default=math.nan)
    if math.isnan(over_num):
        return None
    if 0 <= over_num < 6:
        return "Powerplay"
    if 6 <= over_num < 15:
        return "Middle overs"
    if 15 <= over_num < 20:
        return "Death overs"
    return None


def role_label(bat_pos: Any) -> str | None:
    pos = _safe_float(bat_pos, default=math.nan)
    if math.isnan(pos):
        return None
    if 1 <= pos <= 3:
        return "Top order"
    if 4 <= pos <= 6:
        return "Middle order"
    if pos >= 7:
        return "Finishers"
    return None


def opponent_for_match(row: pd.Series | dict[str, Any], team: str) -> str:
    bat_first = _norm_text(row.get("bat_first"))
    bowl_first = _norm_text(row.get("bowl_first"))
    team_key = _norm_key(team)
    if bat_first.lower() == team_key:
        return bowl_first
    if bowl_first.lower() == team_key:
        return bat_first
    return bowl_first or bat_first


def team_abbreviation(team: str | None) -> str:
    team_name = _norm_text(team)
    if not team_name:
        return "-"
    if team_name in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[team_name]
    words = [word for word in team_name.replace("&", " ").split() if word]
    if len(words) == 1:
        return words[0][:3].upper()
    return "".join(word[0].upper() for word in words[:4])


def profile_metric_subtitle(metric_key: Any, category: Any = None) -> str:
    key = _norm_text(metric_key)
    if key in PROFILE_METRIC_SUBTITLES:
        return PROFILE_METRIC_SUBTITLES[key]
    return _norm_text(category) or "-"


def team_standings_rank(team_ranking: pd.DataFrame, team: str | None) -> int | None:
    if team_ranking.empty or not team or "Team" not in team_ranking.columns or "Rank" not in team_ranking.columns:
        return None

    ranking = team_ranking.copy()
    ranking["team_key"] = ranking["Team"].fillna("").astype(str).str.strip().str.lower()
    rows = ranking.loc[ranking["team_key"].eq(_norm_key(team))]
    if rows.empty:
        return None

    rank = pd.to_numeric(pd.Series([rows.iloc[0]["Rank"]]), errors="coerce").iloc[0]
    if pd.isna(rank):
        return None
    return int(rank)


def team_phase_heatmap_height(match_count: int) -> int:
    count = max(0, int(match_count or 0))
    height = TEAM_PHASE_HEATMAP_HEADER_HEIGHT + (count * TEAM_PHASE_HEATMAP_ROW_HEIGHT)
    return min(TEAM_PHASE_HEATMAP_MAX_HEIGHT, max(TEAM_PHASE_HEATMAP_MIN_HEIGHT, height))


def match_descriptor(row: pd.Series | dict[str, Any]) -> str:
    stage = _norm_text(row.get("stage"))
    playoff_match = _safe_int(row.get("playoff_match"), default=0)
    event_match_no = _norm_text(row.get("event_match_no"))
    if playoff_match == 1 and stage and stage.lower() != "unknown":
        return stage
    if event_match_no and event_match_no.lower() != "unknown":
        return f"Match {event_match_no}"
    return "Match"


def match_label(row: pd.Series | dict[str, Any], team: str) -> str:
    opponent = opponent_for_match(row, team)
    return f"vs {team_abbreviation(opponent)}, {match_descriptor(row)}"


def result_for_team(row: pd.Series | dict[str, Any], team: str) -> str:
    result_type = _norm_key(row.get("result_type"))
    status = _norm_key(row.get("status"))
    winner = _norm_text(row.get("match_won_by"))
    outcome = _norm_text(row.get("win_outcome"))

    if result_type in {"no result", "no_result", "abandoned"} or status in {"abandoned", "no_result"}:
        return "No result"
    if not winner or winner.lower() in {"unknown", "none", "nan", "null"}:
        return "No result"
    suffix = f" by {outcome}" if outcome else ""
    return f"Won{suffix}" if winner.lower() == _norm_key(team) else f"Lost{suffix}"


def format_match_date(value: Any) -> str:
    text = _norm_text(value)
    if not text:
        return "-"
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").strftime("%m/%d/%Y")
    except ValueError:
        return text


def _selected_team_balls(impact_balls: pd.DataFrame, matches: pd.DataFrame, team: str) -> pd.DataFrame:
    if impact_balls.empty or matches.empty or not team or "match_id" not in impact_balls.columns:
        return impact_balls.iloc[0:0].copy()

    match_ids = pd.to_numeric(matches["match_id"], errors="coerce").dropna().astype(int).tolist()
    out = impact_balls.copy()
    out["match_id"] = pd.to_numeric(out["match_id"], errors="coerce")
    out = out[out["match_id"].isin(match_ids)].copy()
    if "innings" in out.columns:
        out = out[pd.to_numeric(out["innings"], errors="coerce").fillna(0).isin([1, 2])].copy()
    return out


def aggregate_team_phase_impact(
    impact_balls: pd.DataFrame,
    matches: pd.DataFrame,
    team: str,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    team_matches = filter_team_season_matches(matches, team)
    match_count = _match_count(team_matches)
    balls = _selected_team_balls(impact_balls, team_matches, team)

    heatmap = team_matches.copy()
    if "match_id" not in heatmap.columns:
        heatmap["match_id"] = pd.Series(dtype=int)
    for column in PHASE_COLUMNS:
        heatmap[column] = 0.0

    if not balls.empty:
        balls = balls.copy()
        balls["phase"] = balls.get("over", pd.Series(np.nan, index=balls.index)).apply(phase_label)
        balls["innings_num"] = pd.to_numeric(balls.get("innings"), errors="coerce")
        balls["prob_bowling_diff_num"] = _numeric_series(balls, "prob_bowling_diff")

        batting = balls[_team_mask(balls, "batting_team", team) & balls["phase"].notna()].copy()
        if not batting.empty:
            batting["value"] = np.where(
                batting["innings_num"].eq(1),
                -batting["prob_bowling_diff_num"],
                batting["prob_bowling_diff_num"],
            )
            batting["column"] = batting["phase"].map(
                {
                    "Powerplay": "PP Batting",
                    "Middle overs": "Middle Batting",
                    "Death overs": "Death Batting",
                }
            )
            heatmap = _merge_phase_values(heatmap, batting)

        bowling = balls[_team_mask(balls, "bowling_team", team) & balls["phase"].notna()].copy()
        if not bowling.empty:
            bowling["value"] = np.where(
                bowling["innings_num"].eq(1),
                bowling["prob_bowling_diff_num"],
                -bowling["prob_bowling_diff_num"],
            )
            bowling["column"] = bowling["phase"].map(
                {
                    "Powerplay": "PP Bowling",
                    "Middle overs": "Middle Bowling",
                    "Death overs": "Death Bowling",
                }
            )
            heatmap = _merge_phase_values(heatmap, bowling)

    if not heatmap.empty:
        heatmap["Match"] = heatmap.apply(lambda row: match_label(row, team), axis=1)
        heatmap["match_id"] = pd.to_numeric(heatmap["match_id"], errors="coerce").astype("Int64")
        heatmap = heatmap[["match_id", "Match", *PHASE_COLUMNS]].copy()
    else:
        heatmap = pd.DataFrame(columns=["match_id", "Match", *PHASE_COLUMNS])

    averages = {
        column: float(pd.to_numeric(heatmap[column], errors="coerce").fillna(0).sum() / match_count)
        if match_count > 0
        else 0.0
        for column in PHASE_COLUMNS
    }
    bowling_hover = _bowling_phase_hover(balls, team)
    return {"heatmap": heatmap, "averages": averages, "bowling_hover": bowling_hover}


def _merge_phase_values(heatmap: pd.DataFrame, phase_balls: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        phase_balls.groupby(["match_id", "column"], as_index=False)["value"]
        .sum()
        .pivot(index="match_id", columns="column", values="value")
        .reset_index()
    )
    grouped["match_id"] = pd.to_numeric(grouped["match_id"], errors="coerce")
    out = heatmap.copy()
    out["match_id"] = pd.to_numeric(out["match_id"], errors="coerce")
    out = out.merge(grouped, on="match_id", how="left", suffixes=("", "_new"))
    for column in PHASE_COLUMNS:
        new_col = f"{column}_new"
        if new_col in out.columns:
            out[column] = pd.to_numeric(out[new_col], errors="coerce").fillna(out[column])
            out = out.drop(columns=[new_col])
    return out


def _bowling_phase_hover(balls: pd.DataFrame, team: str) -> pd.DataFrame:
    columns = ["Phase", "Economy", "Wickets", "Dot Ball %"]
    if balls.empty:
        return pd.DataFrame(columns=columns)

    bowling = balls[_team_mask(balls, "bowling_team", team)].copy()
    if bowling.empty:
        return pd.DataFrame(columns=columns)

    bowling["phase"] = bowling.get("over", pd.Series(np.nan, index=bowling.index)).apply(phase_label)
    bowling = bowling[bowling["phase"].notna()].copy()
    if bowling.empty:
        return pd.DataFrame(columns=columns)

    bowling["runs_bowler_num"] = _numeric_series(bowling, "runs_bowler")
    bowling["bowler_wicket_num"] = _numeric_series(bowling, "bowler_wicket")
    bowling["valid_ball_num"] = _numeric_series(bowling, "valid_ball")
    bowling["runs_batter_num"] = _numeric_series(bowling, "runs_batter")
    bowling["dot_ball_num"] = ((bowling["valid_ball_num"] == 1) & (bowling["runs_batter_num"] == 0)).astype(int)

    grouped = (
        bowling.groupby("phase", as_index=False)
        .agg(
            {
                "runs_bowler_num": "sum",
                "bowler_wicket_num": "sum",
                "valid_ball_num": "sum",
                "dot_ball_num": "sum",
            }
        )
        .rename(columns={"phase": "Phase"})
    )
    grouped["Economy"] = np.where(
        grouped["valid_ball_num"] > 0,
        grouped["runs_bowler_num"] / (grouped["valid_ball_num"] / 6.0),
        np.nan,
    )
    grouped["Wickets"] = grouped["bowler_wicket_num"]
    grouped["Dot Ball %"] = np.where(
        grouped["valid_ball_num"] > 0,
        grouped["dot_ball_num"] * 100 / grouped["valid_ball_num"],
        np.nan,
    )
    return grouped[columns]


def profile_metric_metadata() -> pd.DataFrame:
    return pd.DataFrame(PROFILE_METRICS).copy()


def benchmark_seasons_2008_2025(seasons: list[Any] | pd.Series) -> list[int]:
    parsed = pd.to_numeric(pd.Series(list(seasons)), errors="coerce").dropna().astype(int)
    values = sorted(
        {
            int(season)
            for season in parsed.tolist()
            if BENCHMARK_START_SEASON <= int(season) <= BENCHMARK_END_SEASON
        }
    )
    return values


def _wicket_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)

    wicket_col = None
    if "bowler_wicket" in frame.columns:
        bowler_wickets = _numeric_series(frame, "bowler_wicket")
        if float(bowler_wickets.abs().sum()) > 0:
            wicket_col = bowler_wickets
    if wicket_col is None and "wicket_taken" in frame.columns:
        wicket_col = _numeric_series(frame, "wicket_taken")
    if wicket_col is None:
        wicket_col = pd.Series(0, index=frame.index)
    return wicket_col.fillna(0).astype(float) > 0


def compute_team_profile_metrics(
    impact_balls: pd.DataFrame,
    matches: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    team_matches = filter_team_season_matches(matches, team)
    match_count = _match_count(team_matches)
    balls = _selected_team_balls(impact_balls, team_matches, team)

    values = {metric["metric_key"]: 0.0 for metric in PROFILE_METRICS}
    if match_count > 0 and not balls.empty:
        balls = balls.copy()
        balls["phase"] = balls.get("over", pd.Series(np.nan, index=balls.index)).apply(phase_label)
        balls["innings_num"] = pd.to_numeric(balls.get("innings"), errors="coerce")
        balls["prob_bowling_diff_num"] = _numeric_series(balls, "prob_bowling_diff")

        batting = balls[_team_mask(balls, "batting_team", team)].copy()
        if not batting.empty:
            batting["impact"] = np.where(
                batting["innings_num"].eq(1),
                -batting["prob_bowling_diff_num"],
                batting["prob_bowling_diff_num"],
            )
            batting_phase_map = {
                "Powerplay": "pp_batting",
                "Middle overs": "middle_batting",
                "Death overs": "death_batting",
            }
            for phase, metric_key in batting_phase_map.items():
                values[metric_key] = float(batting.loc[batting["phase"].eq(phase), "impact"].sum() / match_count)

            batting["Role"] = batting.get("bat_pos", pd.Series(np.nan, index=batting.index)).apply(role_label)
            batting_role_map = {
                "Top order": "top_order_batting",
                "Middle order": "middle_order_batting",
                "Finishers": "finisher_batting",
            }
            for role, metric_key in batting_role_map.items():
                values[metric_key] = float(batting.loc[batting["Role"].eq(role), "impact"].sum() / match_count)

        bowling = balls[_team_mask(balls, "bowling_team", team)].copy()
        if not bowling.empty:
            bowling["impact"] = np.where(
                bowling["innings_num"].eq(1),
                bowling["prob_bowling_diff_num"],
                -bowling["prob_bowling_diff_num"],
            )
            bowling_phase_map = {
                "Powerplay": "pp_bowling",
                "Middle overs": "middle_bowling",
                "Death overs": "death_bowling",
            }
            for phase, metric_key in bowling_phase_map.items():
                values[metric_key] = float(bowling.loc[bowling["phase"].eq(phase), "impact"].sum() / match_count)

            wickets = _wicket_mask(bowling)
            values["wicket_taking_bowling"] = float(bowling.loc[wickets, "impact"].sum() / match_count)
            values["bowling_pressure"] = float(bowling.loc[~wickets, "impact"].sum() / match_count)

    rows = []
    for metric in PROFILE_METRICS:
        row = dict(metric)
        row["raw_value"] = float(values.get(metric["metric_key"], 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_profile_percentiles(selected_metrics: pd.DataFrame, benchmark_metrics: pd.DataFrame) -> pd.DataFrame:
    out = selected_metrics.copy()
    if out.empty:
        out["percentile"] = pd.Series(dtype=float)
        return out

    if benchmark_metrics.empty or "metric_key" not in benchmark_metrics.columns:
        out["percentile"] = 0.0
        return out

    benchmark = benchmark_metrics.copy()
    benchmark["raw_value"] = pd.to_numeric(benchmark.get("raw_value"), errors="coerce")
    percentiles = []
    for _, row in out.iterrows():
        metric_key = row.get("metric_key")
        selected_value = _safe_float(row.get("raw_value"))
        values = benchmark.loc[benchmark["metric_key"].eq(metric_key), "raw_value"].dropna()
        if values.empty:
            percentiles.append(0.0)
            continue
        percentiles.append(float((values <= selected_value).sum() * 100.0 / len(values)))
    out["percentile"] = percentiles
    return out


def compute_profile_season_ranks(
    selected_metrics: pd.DataFrame,
    season_metrics: pd.DataFrame,
    team: str,
    season: int | None,
) -> pd.DataFrame:
    out = selected_metrics.copy()
    if out.empty:
        out["season_rank"] = pd.Series(dtype="Int64")
        out["season_rank_total"] = pd.Series(dtype="Int64")
        out["season"] = season
        return out

    out["season_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["season_rank_total"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["season"] = season
    if season_metrics.empty or "metric_key" not in season_metrics.columns or "team" not in season_metrics.columns:
        return out

    metrics = season_metrics.copy()
    metrics["raw_value"] = pd.to_numeric(metrics.get("raw_value"), errors="coerce")
    metrics["team_key"] = metrics["team"].fillna("").astype(str).str.strip().str.lower()
    selected_team_key = _norm_key(team)
    for idx, row in out.iterrows():
        metric_key = row.get("metric_key")
        metric_rows = metrics.loc[metrics["metric_key"].eq(metric_key)].copy()
        metric_rows = metric_rows.dropna(subset=["raw_value"])
        if metric_rows.empty:
            continue
        metric_rows["rank"] = metric_rows["raw_value"].rank(method="min", ascending=False)
        team_rows = metric_rows.loc[metric_rows["team_key"].eq(selected_team_key)]
        if team_rows.empty:
            continue
        out.at[idx, "season_rank"] = int(team_rows.iloc[0]["rank"])
        out.at[idx, "season_rank_total"] = int(metric_rows["team_key"].nunique())
    return out


def select_profile_strength_cards(profile_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "card_label",
        "metric_key",
        "metric_name",
        "profile",
        "category",
        "raw_value",
        "percentile",
        "season_rank",
        "season_rank_total",
        "season",
        "display_order",
        "description",
    ]
    if profile_metrics.empty:
        return pd.DataFrame(columns=columns)

    metrics = profile_metrics.copy()
    metrics["raw_value"] = pd.to_numeric(metrics.get("raw_value"), errors="coerce").fillna(0.0)
    metrics["percentile"] = pd.to_numeric(metrics.get("percentile"), errors="coerce").fillna(0.0)
    metrics["display_order"] = pd.to_numeric(metrics.get("display_order"), errors="coerce").fillna(9999)
    if "season_rank" not in metrics.columns:
        metrics["season_rank"] = pd.NA
    if "season_rank_total" not in metrics.columns:
        metrics["season_rank_total"] = pd.NA
    if "season" not in metrics.columns:
        metrics["season"] = pd.NA

    strongest = metrics.sort_values(["raw_value", "display_order"], ascending=[False, True], kind="mergesort")
    weakest = metrics.sort_values(["raw_value", "display_order"], ascending=[True, True], kind="mergesort")
    rows = []
    labels_and_rows = [
        ("Primary Strength", strongest.iloc[0] if len(strongest) >= 1 else None),
        ("Secondary Strength", strongest.iloc[1] if len(strongest) >= 2 else None),
        ("Weakest Area", weakest.iloc[0] if len(weakest) >= 1 else None),
    ]
    for label, row in labels_and_rows:
        if row is None:
            continue
        data = row.to_dict()
        data["card_label"] = label
        rows.append(data)
    return pd.DataFrame(rows, columns=columns)


def aggregate_team_role_impact(
    impact_balls: pd.DataFrame,
    matches: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    team_matches = filter_team_season_matches(matches, team)
    match_count = _match_count(team_matches)
    balls = _selected_team_balls(impact_balls, team_matches, team)

    rows = [{"Role": role, "Avg Impact / Match": 0.0, "Total Impact": 0.0} for role in BATTING_ROLE_ORDER]
    out = pd.DataFrame(rows)
    if balls.empty or match_count <= 0:
        return out

    batting = balls[_team_mask(balls, "batting_team", team)].copy()
    if batting.empty:
        return out

    batting["Role"] = batting.get("bat_pos", pd.Series(np.nan, index=batting.index)).apply(role_label)
    batting = batting[batting["Role"].notna()].copy()
    if batting.empty:
        return out

    batting["impact"] = _numeric_series(batting, "batter_delta")
    grouped = batting.groupby("Role", as_index=False)["impact"].sum().rename(columns={"impact": "Total Impact"})
    grouped["Avg Impact / Match"] = grouped["Total Impact"] / float(match_count)

    out = out.drop(columns=["Avg Impact / Match", "Total Impact"]).merge(grouped, on="Role", how="left")
    out[["Avg Impact / Match", "Total Impact"]] = out[["Avg Impact / Match", "Total Impact"]].fillna(0.0)
    return out


def team_total_impact(impact_balls: pd.DataFrame, matches: pd.DataFrame, team: str) -> float:
    team_matches = filter_team_season_matches(matches, team)
    balls = _selected_team_balls(impact_balls, team_matches, team)
    if balls.empty:
        return 0.0
    batting_total = _numeric_series(balls[_team_mask(balls, "batting_team", team)], "batter_delta").sum()
    bowling_total = _numeric_series(balls[_team_mask(balls, "bowling_team", team)], "bowler_delta").sum()
    return float(batting_total + bowling_total)


def team_record(matches: pd.DataFrame, team: str) -> dict[str, int]:
    team_matches = filter_team_season_matches(matches, team)
    wins = 0
    losses = 0
    no_results = 0
    for _, row in team_matches.iterrows():
        result = result_for_team(row, team)
        if result.startswith("Won"):
            wins += 1
        elif result.startswith("Lost"):
            losses += 1
        else:
            no_results += 1
    return {"matches": _match_count(team_matches), "wins": wins, "losses": losses, "no_results": no_results}


def top_impact_player(impact_balls: pd.DataFrame, matches: pd.DataFrame, team: str) -> dict[str, Any] | None:
    team_matches = filter_team_season_matches(matches, team)
    balls = _selected_team_balls(impact_balls, team_matches, team)
    player_totals = _player_impact_totals(balls, team)
    if player_totals.empty:
        return None
    row = player_totals.sort_values(["Total Impact", "Player"], ascending=[False, True]).iloc[0]
    return {"Player": row["Player"], "Total Impact": float(row["Total Impact"])}


def top_impact_player_by_match(impact_balls: pd.DataFrame, matches: pd.DataFrame, team: str) -> pd.DataFrame:
    team_matches = filter_team_season_matches(matches, team)
    balls = _selected_team_balls(impact_balls, team_matches, team)
    if balls.empty:
        return pd.DataFrame(columns=["match_id", "Top Impact Player", "Top Player Impact"])

    rows = []
    for match_id, match_balls in balls.groupby("match_id", sort=False):
        player_totals = _player_impact_totals(match_balls, team)
        if player_totals.empty:
            continue
        top = player_totals.sort_values(["Total Impact", "Player"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "match_id": int(match_id),
                "Top Impact Player": top["Player"],
                "Top Player Impact": float(top["Total Impact"]),
            }
        )
    return pd.DataFrame(rows, columns=["match_id", "Top Impact Player", "Top Player Impact"])


def _player_impact_totals(balls: pd.DataFrame, team: str) -> pd.DataFrame:
    if balls.empty:
        return pd.DataFrame(columns=["player_key", "Player", "Batting Impact", "Bowling Impact", "Total Impact"])

    batting = balls[_team_mask(balls, "batting_team", team)].copy()
    batting_totals = _role_player_totals(batting, "batter", "batter_delta", "Batting Impact")

    bowling = balls[_team_mask(balls, "bowling_team", team)].copy()
    bowling_totals = _role_player_totals(bowling, "bowler", "bowler_delta", "Bowling Impact")

    out = batting_totals.merge(bowling_totals, on="player_key", how="outer", suffixes=("_bat", "_bowl"))
    if out.empty:
        return pd.DataFrame(columns=["player_key", "Player", "Batting Impact", "Bowling Impact", "Total Impact"])

    out["Player"] = np.where(
        out.get("Player_bat", pd.Series("", index=out.index)).fillna("").astype(str).str.strip() != "",
        out.get("Player_bat", pd.Series("", index=out.index)),
        out.get("Player_bowl", pd.Series("", index=out.index)),
    )
    out["Batting Impact"] = pd.to_numeric(out.get("Batting Impact"), errors="coerce").fillna(0)
    out["Bowling Impact"] = pd.to_numeric(out.get("Bowling Impact"), errors="coerce").fillna(0)
    out["Total Impact"] = out["Batting Impact"] + out["Bowling Impact"]
    out = out[["player_key", "Player", "Batting Impact", "Bowling Impact", "Total Impact"]]
    out = out[out["Player"].fillna("").astype(str).str.strip() != ""].copy()
    return out


def _role_player_totals(frame: pd.DataFrame, role: str, impact_col: str, output_col: str) -> pd.DataFrame:
    columns = ["player_key", "Player", output_col]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    key_col = f"{role}_key"
    id_col = f"{role}_player_id"
    canonical_col = f"{role}_canonical_name"

    out = frame.copy()
    raw_name = out.get(role, pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    if key_col in out.columns:
        key = out[key_col].fillna("").astype(str).str.strip()
    elif id_col in out.columns:
        player_id = pd.to_numeric(out[id_col], errors="coerce")
        key = np.where(player_id.notna(), "id:" + player_id.astype("Int64").astype(str), "name:" + raw_name)
    else:
        key = "name:" + raw_name
    canonical = out.get(canonical_col, pd.Series("", index=out.index)).fillna("").astype(str).str.strip()

    out["player_key"] = key
    out["Player"] = np.where(canonical != "", canonical, raw_name)
    out[output_col] = _numeric_series(out, impact_col)
    out = out[(out["player_key"].astype(str).str.strip() != "") & (out["Player"].astype(str).str.strip() != "")]
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out.groupby(["player_key", "Player"], as_index=False)[output_col].sum()


def identity_category_averages(
    role_impact: pd.DataFrame,
    phase_averages: dict[str, float],
) -> dict[str, float]:
    role_lookup = {
        str(row.get("Role")): _safe_float(row.get("Avg Impact / Match"))
        for _, row in role_impact.iterrows()
    }
    return {
        "Top-order batting impact": role_lookup.get("Top order", 0.0),
        "Middle-order batting impact": role_lookup.get("Middle order", 0.0),
        "Finisher batting impact": role_lookup.get("Finishers", 0.0),
        "Powerplay bowling impact": float(phase_averages.get("PP Bowling", 0.0)),
        "Middle-overs bowling impact": float(phase_averages.get("Middle Bowling", 0.0)),
        "Death-overs bowling impact": float(phase_averages.get("Death Bowling", 0.0)),
    }


def generate_team_identity_summary(team: str, category_averages: dict[str, float]) -> dict[str, Any]:
    rows = [
        {"category": category, "value": float(category_averages.get(category, 0.0))}
        for category in IDENTITY_CATEGORY_ORDER
    ]
    ranked = sorted(
        rows,
        key=lambda row: (-row["value"], IDENTITY_CATEGORY_ORDER.index(row["category"])),
    )
    weakest_ranked = sorted(
        rows,
        key=lambda row: (row["value"], IDENTITY_CATEGORY_ORDER.index(row["category"])),
    )
    primary = ranked[0]["category"] if ranked else "-"
    secondary = ranked[1]["category"] if len(ranked) > 1 else "-"
    weakest = weakest_ranked[0]["category"] if weakest_ranked else "-"
    sentence = (
        f"{team} has most often gained its advantage through {primary} and {secondary}, "
        f"while {weakest} has been its weakest area."
    )
    return {
        "primary_strength": primary,
        "secondary_strength": secondary,
        "weakest_area": weakest,
        "sentence": sentence,
        "ranked": ranked,
    }


def best_phase_name(values: dict[str, Any] | pd.Series) -> str:
    pairs = []
    for column in PHASE_COLUMNS:
        value = values.get(column, 0.0)
        pairs.append((column, _safe_float(value)))
    if not pairs:
        return "-"
    return sorted(pairs, key=lambda item: (-item[1], PHASE_COLUMNS.index(item[0])))[0][0]


def build_match_table_rows(
    matches: pd.DataFrame,
    team: str,
    heatmap: pd.DataFrame,
    top_players: pd.DataFrame,
) -> pd.DataFrame:
    team_matches = filter_team_season_matches(matches, team)
    columns = [
        "Date",
        "Match No.",
        "Opponent",
        "Result",
        "Team Total Impact",
        "Top Impact Player",
        "Match",
    ]
    if team_matches.empty:
        return pd.DataFrame(columns=columns)

    phase_values = heatmap.copy()
    if not phase_values.empty:
        phase_values["match_id"] = pd.to_numeric(phase_values["match_id"], errors="coerce")
        phase_values["Team Total Impact"] = phase_values[PHASE_COLUMNS].sum(axis=1)
        phase_values["Best Phase"] = phase_values.apply(best_phase_name, axis=1)
        phase_values = phase_values[["match_id", "Team Total Impact", "Best Phase"]]
    else:
        phase_values = pd.DataFrame(columns=["match_id", "Team Total Impact", "Best Phase"])

    top = top_players.copy()
    if not top.empty:
        top["match_id"] = pd.to_numeric(top["match_id"], errors="coerce")

    out = team_matches.copy()
    out["match_id"] = pd.to_numeric(out["match_id"], errors="coerce")
    out = out.merge(phase_values, on="match_id", how="left")
    out = out.merge(top[["match_id", "Top Impact Player"]] if not top.empty else top, on="match_id", how="left")
    out["Date"] = out.get("date", pd.Series("", index=out.index)).apply(format_match_date)
    out["Match No."] = out.apply(match_descriptor, axis=1)
    out["Opponent"] = out.apply(lambda row: opponent_for_match(row, team), axis=1)
    out["Result"] = out.apply(lambda row: result_for_team(row, team), axis=1)
    out["Team Total Impact"] = pd.to_numeric(out["Team Total Impact"], errors="coerce").fillna(0).round(2)
    out["Top Impact Player"] = out["Top Impact Player"].fillna("-")
    out["Best Phase"] = out["Best Phase"].fillna("-")
    out["Match"] = out["match_id"].apply(lambda match_id: f"[Open](/match-analysis?match_id={int(match_id)})")
    sort_cols = [col for col in ["date", "match_id"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
    return out[columns]
