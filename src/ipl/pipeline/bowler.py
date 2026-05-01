import numpy as np
import pandas as pd


def _safe_divide(numerator, denominator):
    denom = pd.to_numeric(denominator, errors="coerce").fillna(0)
    denom = np.where(denom == 0, 1, denom)
    num = pd.to_numeric(numerator, errors="coerce").fillna(0)
    return num / denom


def _attach_identity(df: pd.DataFrame, role: str) -> pd.DataFrame:
    out = df.copy()
    raw_col = role
    player_id_col = f"{role}_player_id"
    canonical_col = f"{role}_canonical_name"

    raw_name = out.get(raw_col, pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    if player_id_col in out.columns:
        player_id = pd.to_numeric(out[player_id_col], errors="coerce")
    else:
        player_id = pd.Series(np.nan, index=out.index)

    canonical = out.get(canonical_col, pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    resolved = player_id.notna()
    player_id_text = player_id.astype("Int64").astype(str)
    out[f"{role}_key"] = np.where(resolved, "id:" + player_id_text, "name:" + raw_name)
    out[role] = np.where(resolved & (canonical != ""), canonical, raw_name)
    return out


def _ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        inferred = pd.to_datetime(out.get("date"), errors="coerce").dt.year
        out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(inferred)
    else:
        out["year"] = pd.to_datetime(out.get("date"), errors="coerce").dt.year
    out["year"] = out["year"].fillna(0).astype(int)
    return out


def _ensure_bowler_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["innings"] = pd.to_numeric(out.get("innings"), errors="coerce").fillna(0).astype(int)
    out["runs_bowler"] = pd.to_numeric(out.get("runs_bowler"), errors="coerce").fillna(0)
    out["bowler_wicket"] = pd.to_numeric(out.get("bowler_wicket"), errors="coerce").fillna(0)
    out["valid_ball"] = pd.to_numeric(out.get("valid_ball"), errors="coerce").fillna(0)
    out["over"] = pd.to_numeric(out.get("over"), errors="coerce").fillna(-1).astype(int)

    runs_batter = pd.to_numeric(out.get("runs_batter", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    runs_not_boundary = out.get("runs_not_boundary", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["four_conceded"] = ((runs_batter == 4) & (~runs_not_boundary)).astype(int)
    out["six_conceded"] = ((runs_batter == 6) & (~runs_not_boundary)).astype(int)
    out["dot_ball"] = ((runs_batter == 0) & (out["valid_ball"] == 1)).astype(int)
    out["valid_ball"] = out["valid_ball"].astype(int)
    return out


def bowler_info(df):
    X = _attach_identity(df, "bowler")
    X = X[X["bowler"].astype(str).str.strip() != ""].copy()
    X = _ensure_year_column(X)
    X = _ensure_bowler_metric_columns(X)

    # Per-match figures for best bowling and match counts.
    match_figures = (
        X[X["innings"] <= 2]
        .groupby(["year", "match_id", "bowler_key"], as_index=False)
        .agg(
            {
                "bowler": "first",
                "runs_bowler": "sum",
                "bowler_wicket": "sum",
                "valid_ball": "sum",
                "dot_ball": "sum",
                "four_conceded": "sum",
                "six_conceded": "sum",
            }
        )
        .rename(columns={"runs_bowler": "runs", "bowler_wicket": "wickets", "valid_ball": "balls"})
    )

    # Maiden overs: complete legal overs with 0 runs conceded.
    over_summary = (
        X[(X["innings"] <= 2) & (X["over"] >= 0)]
        .groupby(["year", "match_id", "bowler_key", "innings", "over"], as_index=False)
        .agg(
            {
                "bowler": "first",
                "runs_bowler": "sum",
                "valid_ball": "sum",
            }
        )
        .rename(columns={"runs_bowler": "runs_over", "valid_ball": "balls_over"})
    )
    maidens = (
        over_summary[(over_summary["balls_over"] == 6) & (over_summary["runs_over"] == 0)]
        .groupby(["year", "bowler_key"], as_index=False)
        .size()
        .rename(columns={"size": "maidens"})
    )

    best_figures = match_figures.sort_values(
        ["year", "bowler_key", "wickets", "runs", "match_id"],
        ascending=[True, True, False, True, True],
    ).drop_duplicates(["year", "bowler_key"], keep="first")
    best_figures["best_bowling"] = (
        best_figures["wickets"].astype(int).astype(str) + "/" + best_figures["runs"].round().astype(int).astype(str)
    )
    best_figures = best_figures[["year", "bowler_key", "best_bowling"]]

    bowlers_by_season = (
        match_figures.groupby(["year", "bowler_key"], as_index=False)
        .agg(
            {
                "bowler": "first",
                "match_id": "count",
                "runs": "sum",
                "wickets": "sum",
                "balls": "sum",
                "dot_ball": "sum",
                "four_conceded": "sum",
                "six_conceded": "sum",
            }
        )
        .rename(columns={"match_id": "matches"})
    )
    bowlers_by_season = bowlers_by_season.merge(maidens, on=["year", "bowler_key"], how="left")
    bowlers_by_season = bowlers_by_season.merge(best_figures, on=["year", "bowler_key"], how="left")
    bowlers_by_season["maidens"] = bowlers_by_season["maidens"].fillna(0).astype(int)
    bowlers_by_season["overs"] = bowlers_by_season["balls"] / 6.0
    bowlers_by_season["economy"] = _safe_divide(bowlers_by_season["runs"], bowlers_by_season["overs"])
    bowlers_by_season["bowling_average"] = _safe_divide(bowlers_by_season["runs"], bowlers_by_season["wickets"])
    bowlers_by_season["dot_ball_pct"] = _safe_divide(bowlers_by_season["dot_ball"], bowlers_by_season["balls"]) * 100

    bowlers_by_season = bowlers_by_season.rename(
        columns={
            "runs": "runs_conceded",
            "wickets": "wickets",
            "four_conceded": "fours_conceded",
            "six_conceded": "sixes_conceded",
            "dot_ball": "dot_balls",
        }
    )
    bowlers_by_season = bowlers_by_season.sort_values(by=["year", "wickets", "runs_conceded"], ascending=[False, False, True])
    bowlers_by_season = bowlers_by_season.reset_index(drop=True)

    bowlers = match_figures.copy()
    bowlers["seasons"] = 1
    bowlers = (
        bowlers.groupby("bowler_key", as_index=False)
        .agg(
            {
                "bowler": "first",
                "year": "nunique",
                "match_id": "nunique",
                "runs": "sum",
                "wickets": "sum",
                "balls": "sum",
                "dot_ball": "sum",
                "four_conceded": "sum",
                "six_conceded": "sum",
            }
        )
        .rename(columns={"year": "seasons", "match_id": "matches", "runs": "runs_conceded", "dot_ball": "dot_balls"})
    )
    bowlers["overs"] = bowlers["balls"] / 6.0
    bowlers["economy"] = _safe_divide(bowlers["runs_conceded"], bowlers["overs"])
    bowlers["bowling_average"] = _safe_divide(bowlers["runs_conceded"], bowlers["wickets"])
    bowlers["dot_ball_pct"] = _safe_divide(bowlers["dot_balls"], bowlers["balls"]) * 100
    bowlers = bowlers.sort_values(["wickets", "runs_conceded"], ascending=[False, True]).set_index("bowler")

    return bowlers, bowlers_by_season
