import numpy as np
import pandas as pd


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


def _attach_batter_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = _attach_identity(df, "batter")
    return out[out["batter"] != ""]


def _ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        inferred = pd.to_datetime(out.get("date"), errors="coerce").dt.year
        out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(inferred)
    else:
        out["year"] = pd.to_datetime(out.get("date"), errors="coerce").dt.year
    out["year"] = out["year"].fillna(0).astype(int)
    return out


def _ensure_balls_faced(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    extra_type = out.get("extra_type", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    wides_mask = extra_type.str.contains("wides", regex=False)
    inferred_balls = pd.Series(np.where(wides_mask, 0, 1), index=out.index)

    if "balls_faced" in out.columns:
        out["balls_faced"] = pd.to_numeric(out["balls_faced"], errors="coerce").fillna(inferred_balls)
    else:
        out["balls_faced"] = inferred_balls

    out["balls_faced"] = out["balls_faced"].fillna(0).astype(int)
    return out


def _ensure_batter_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["innings"] = pd.to_numeric(out.get("innings"), errors="coerce").fillna(0).astype(int)
    out["runs_batter"] = pd.to_numeric(out.get("runs_batter"), errors="coerce").fillna(0)
    out["bat_pos"] = pd.to_numeric(out.get("bat_pos"), errors="coerce").fillna(0).astype(int)
    out = _ensure_balls_faced(out)

    runs_not_boundary = out.get("runs_not_boundary", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    if "four" in out.columns:
        out["four"] = pd.to_numeric(out["four"], errors="coerce").fillna(0)
    else:
        out["four"] = ((out["runs_batter"] == 4) & (~runs_not_boundary)).astype(int)

    if "six" in out.columns:
        out["six"] = pd.to_numeric(out["six"], errors="coerce").fillna(0)
    else:
        out["six"] = ((out["runs_batter"] == 6) & (~runs_not_boundary)).astype(int)

    valid_ball = pd.to_numeric(out.get("valid_ball", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    if "dot" in out.columns:
        out["dot"] = pd.to_numeric(out["dot"], errors="coerce").fillna(0)
    else:
        out["dot"] = ((out["runs_batter"] == 0) & (valid_ball == 1)).astype(int)

    out["four"] = pd.to_numeric(out["four"], errors="coerce").fillna(0).astype(int)
    out["six"] = pd.to_numeric(out["six"], errors="coerce").fillna(0).astype(int)
    out["dot"] = pd.to_numeric(out["dot"], errors="coerce").fillna(0).astype(int)
    return out


def _ensure_batter_out_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "batter_out_in_innings" in out.columns:
        out["batter_out_in_innings"] = out["batter_out_in_innings"].fillna(False).astype(bool)
        return out

    out = _attach_identity(out, "player_out")
    dismissal_mask = (
        (out["innings"] <= 2)
        & out["player_out"].notna()
        & (out["player_out"].astype(str).str.strip() != "")
        & (out.get("wicket_kind", pd.Series("", index=out.index)).fillna("").astype(str).str.lower() != "retired hurt")
    )
    dismissals = (
        out.loc[dismissal_mask, ["match_id", "player_out_key"]]
        .drop_duplicates()
        .rename(columns={"player_out_key": "batter_key"})
        .assign(batter_out_in_innings=True)
    )
    out = out.merge(dismissals, on=["match_id", "batter_key"], how="left")
    out["batter_out_in_innings"] = out["batter_out_in_innings"].fillna(False).astype(bool)
    return out


def batter_info(df):
    X = _attach_batter_identity(df)
    X = _ensure_year_column(X)
    X = _ensure_batter_metric_columns(X)
    X = _ensure_batter_out_flag(X)

    batter_matches = X[X["innings"] <= 2][
        [
            "match_id",
            "year",
            "date",
            "innings",
            "bat_pos",
            "batter_key",
            "batter",
            "runs_batter",
            "balls_faced",
            "four",
            "six",
            "dot",
            "batting_team",
            "batter_out_in_innings",
        ]
    ]
    batter_matches = batter_matches.groupby(["match_id", "batter_key"], as_index=False).agg(
        {
            "batter": "first",
            "date": "first",
            "innings": "first",
            "year": "first",
            "bat_pos": "first",
            "runs_batter": "sum",
            "balls_faced": "sum",
            "four": "sum",
            "six": "sum",
            "dot": "sum",
            "batting_team": "first",
            "batter_out_in_innings": "max",
        }
    )
    batter_matches = batter_matches.rename({"runs_batter": "runs"}, axis=1)
    batter_matches["100"] = batter_matches["runs"] >= 100
    batter_matches["50"] = (batter_matches["runs"] >= 50) & (batter_matches["runs"] < 100)

    batters_by_season = (
        batter_matches[["year", "match_id", "batter_key", "batter", "runs", "balls_faced", "batter_out_in_innings"]]
        .groupby(["year", "batter_key"], as_index=False)
        .agg(
            {
                "batter": "first",
                "match_id": "count",
                "runs": "sum",
                "balls_faced": "sum",
                "batter_out_in_innings": "sum",
            }
        )
        .rename({"match_id": "matches", "batter_out_in_innings": "dismissals"}, axis=1)
    )
    batters_by_season["strike_rate"] = np.where(
        batters_by_season["balls_faced"] > 0,
        batters_by_season["runs"] / batters_by_season["balls_faced"] * 100,
        0,
    )
    batters_by_season["dismissals"] = batters_by_season["dismissals"].astype(int)
    batters_by_season["average"] = np.where(
        batters_by_season["dismissals"] == 0,
        np.nan,
        batters_by_season["runs"] / batters_by_season["dismissals"],
    )
    batters_by_season = batters_by_season.sort_values(by="runs", ascending=False)
    batters_by_season = batters_by_season.reset_index(drop=True)

    batters = batter_matches.copy()[
        ["batter_key", "batter", "match_id", "year", "runs", "balls_faced", "four", "six", "batter_out_in_innings", "50", "100"]
    ]
    batters["highest_score"] = batters["runs"]
    batters = (
        batters.groupby("batter_key", as_index=False)
        .agg(
            {
                "batter": "first",
                "match_id": "nunique",
                "year": "nunique",
                "runs": "sum",
                "balls_faced": "sum",
                "four": "sum",
                "six": "sum",
                "batter_out_in_innings": "sum",
                "50": "sum",
                "100": "sum",
                "highest_score": "max",
            }
        )
        .rename({"match_id": "innings", "year": "seasons", "batter_out_in_innings": "dismissals"}, axis=1)
    )
    batters["strike_rate"] = np.where(batters["balls_faced"] > 0, batters["runs"] / batters["balls_faced"] * 100, 0)
    batters["dismissals"] = batters["dismissals"].astype(int)
    batters["average"] = np.where(batters["dismissals"] == 0, np.nan, batters["runs"] / batters["dismissals"])
    batters = batters[
        ["batter", "seasons", "innings", "runs", "strike_rate", "average", "highest_score", "100", "50", "four", "six"]
    ]
    batters = batters.sort_values(by="runs", ascending=False).set_index("batter")

    return batters, batters_by_season


def batter_match_stats(df):
    X = _attach_batter_identity(df)
    X = _ensure_batter_metric_columns(X)
    X = _ensure_batter_out_flag(X)
    X = X.groupby(["batter_key", "batter"], as_index=False).agg(
        {
            "innings": "first",
            "bat_pos": "first",
            "runs_batter": "sum",
            "balls_faced": "sum",
            "four": "sum",
            "six": "sum",
            "batter_out_in_innings": "max",
        }
    )
    X["Not Out"] = ~X["batter_out_in_innings"].fillna(False).astype(bool)
    X = X.rename(
        columns={
            "bat_pos": "Position",
            "runs_batter": "Runs",
            "balls_faced": "Balls",
            "four": "4",
            "six": "6",
            "batter": "Batter",
        }
    )
    X["Strike Rate"] = np.where(X["Balls"] > 0, X["Runs"] / X["Balls"] * 100, 0)
    return X[["Batter", "Position", "Runs", "Balls", "Strike Rate", "4", "6", "Not Out"]]
