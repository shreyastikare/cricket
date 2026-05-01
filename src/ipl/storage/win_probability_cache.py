"""Per-ball win probability and impact cache helpers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ipl

from .config import DB_PATH, PROJECT_ROOT
from .schema import create_schema
from .sqlite import transaction, upsert_dataframe
from .utils import utc_now_iso


WIN_PROBABILITY_MODEL_TYPE = "win_probability"
WIN_PROBABILITY_CACHE_COLUMNS = [
    "prediction_id",
    "match_id",
    "ball_id",
    "model_type",
    "model_version",
    "prediction_ts",
    "batting_team_win_prob",
    "bowling_team_win_prob",
    "prob_bowling_diff",
    "prob_batting_diff",
    "metadata_json",
    "created_at",
    "updated_at",
]


def _model_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@lru_cache(maxsize=1)
def current_model_version() -> str:
    from ipl.pipeline.first_innings_glm import FIRST_INNINGS_GLM_PATH
    from ipl.pipeline.second_innings_glm import SECOND_INNINGS_GLM_PATH

    digest = hashlib.sha256()
    for path_value in [FIRST_INNINGS_GLM_PATH, SECOND_INNINGS_GLM_PATH]:
        path = _model_path(path_value)
        digest.update(str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prediction_id(ball_id: str) -> str:
    return f"{WIN_PROBABILITY_MODEL_TYPE}:{ball_id}"


def _sort_match_balls(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sort_cols: list[str] = []
    for col in ["match_id", "innings", "team_balls", "over", "ball", "id"]:
        if col in out.columns:
            helper_col = f"_{col}_sort"
            out[helper_col] = pd.to_numeric(out[col], errors="coerce")
            sort_cols.append(helper_col)
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").drop(columns=sort_cols)
    return out


def _second_innings_model_mask(frame: pd.DataFrame, base_mask: pd.Series) -> pd.Series:
    match_ids = pd.to_numeric(frame.get("match_id"), errors="coerce")
    team_balls = pd.to_numeric(frame.get("team_balls"), errors="coerce")
    pairs = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "match_id": match_ids.loc[frame.get("innings").eq(1) & team_balls.notna()].astype("Int64"),
                "team_balls": team_balls.loc[frame.get("innings").eq(1) & team_balls.notna()].round().astype("Int64"),
            }
        ).dropna()
    )
    row_pairs = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "match_id": match_ids.astype("Int64"),
                "team_balls": team_balls.round().astype("Int64"),
            },
            index=frame.index,
        )
    )
    return base_mask & pd.Series(row_pairs.isin(pairs), index=frame.index)


def _assign_predicted_probabilities(
    frame: pd.DataFrame,
    mask: pd.Series,
    feature_frame: pd.DataFrame,
    y_prob: np.ndarray,
) -> None:
    if "ball_id" in frame.columns and "ball_id" in feature_frame.columns:
        predicted = pd.Series(
            y_prob,
            index=feature_frame["ball_id"].fillna("").astype(str).str.strip(),
        )
        ball_ids = frame.loc[mask, "ball_id"].fillna("").astype(str).str.strip()
        aligned = ball_ids.map(predicted)
        valid = aligned.notna()
        if valid.any():
            frame.loc[aligned.index[valid], "y_prob"] = aligned.loc[valid].to_numpy()
        return

    idx = frame.index[mask]
    n = min(len(idx), len(y_prob))
    if n > 0:
        frame.loc[idx[:n], "y_prob"] = y_prob[:n]


def _target_from_balls(frame: pd.DataFrame, first_innings_runs: int) -> int:
    innings_two = frame.loc[pd.to_numeric(frame.get("innings"), errors="coerce") == 2]
    if not innings_two.empty and "runs_target" in innings_two.columns:
        target_values = pd.to_numeric(innings_two["runs_target"], errors="coerce").dropna()
        target_values = target_values[target_values > 0]
        if not target_values.empty:
            return int(target_values.max())
    return int(first_innings_runs) + 1


def _innings_ball_limit(frame: pd.DataFrame, innings: int) -> int:
    innings_num = pd.to_numeric(frame.get("innings"), errors="coerce")
    Xi = frame.loc[innings_num == int(innings)].copy()
    if Xi.empty:
        return 120
    if "team_balls" in Xi.columns:
        Xi = Xi.sort_values("team_balls")
    last = Xi.iloc[-1]
    overs_limit = pd.to_numeric(pd.Series([last.get("overs")]), errors="coerce").iloc[0]
    balls_per_over = pd.to_numeric(pd.Series([last.get("balls_per_over")]), errors="coerce").iloc[0]
    if pd.isna(overs_limit) or float(overs_limit) <= 0:
        return 120
    bpo = float(balls_per_over) if pd.notna(balls_per_over) and float(balls_per_over) > 0 else 6.0
    ball_limit = int(round(float(overs_limit) * bpo))
    return ball_limit if ball_limit > 0 else 120


def _second_innings_terminal_probability(frame: pd.DataFrame, match_row: pd.Series) -> tuple[Any, float] | None:
    innings_num = pd.to_numeric(frame.get("innings"), errors="coerce")
    Xi = frame.loc[innings_num == 2].copy()
    if Xi.empty:
        return None

    sort_cols = [col for col in ["team_balls", "over", "ball", "id"] if col in Xi.columns]
    if sort_cols:
        Xi = Xi.sort_values(sort_cols)
    last_idx = Xi.index[-1]
    last = Xi.iloc[-1]

    second_runs = pd.to_numeric(pd.Series([last.get("team_runs")]), errors="coerce").iloc[0]
    second_wickets = pd.to_numeric(pd.Series([last.get("team_wicket")]), errors="coerce").iloc[0]
    second_balls = pd.to_numeric(pd.Series([last.get("team_balls")]), errors="coerce").iloc[0]
    required_runs_last = pd.to_numeric(pd.Series([last.get("required_runs")]), errors="coerce").iloc[0]
    if pd.isna(second_runs):
        return None

    first_innings_runs = pd.to_numeric(
        frame.loc[innings_num == 1, "team_runs"],
        errors="coerce",
    ).dropna()
    first_end_runs = int(first_innings_runs.max()) if not first_innings_runs.empty else 0
    target = _target_from_balls(frame, first_end_runs)
    second_ball_limit = _innings_ball_limit(frame, innings=2)

    innings_over = False
    if pd.notna(required_runs_last) and float(required_runs_last) <= 0:
        innings_over = True
    if pd.notna(second_wickets) and int(second_wickets) >= 10:
        innings_over = True
    if pd.notna(second_balls) and int(second_balls) >= int(second_ball_limit):
        innings_over = True

    if str(match_row.get("status") or "").strip().lower() != "complete":
        return None
    if not innings_over:
        return None

    second_runs_int = int(second_runs)
    target_int = int(target)
    if second_runs_int == target_int - 1:
        return last_idx, 0.5

    batting_team = str(match_row.get("bat_first") or "").strip().lower()
    bowling_team = str(match_row.get("bowl_first") or "").strip().lower()
    winner = str(match_row.get("match_won_by") or "").strip().lower()
    if winner == bowling_team:
        return last_idx, 1.0
    if winner == batting_team:
        return last_idx, 0.0

    if second_runs_int >= target_int:
        return last_idx, 1.0
    return last_idx, 0.0


def score_match_probabilities(
    match_balls: pd.DataFrame,
    matches_with_prior: pd.DataFrame,
    match_row: pd.Series,
    first_innings_model,
    second_innings_model,
) -> pd.DataFrame:
    Xi = _sort_match_balls(match_balls.copy())
    Xi["y_prob"] = np.nan

    X_first = ipl.preprocess_first_innings(Xi, matches_with_prior).copy()
    X_second = ipl.preprocess_second_innings(Xi, matches_with_prior).copy()

    mask_1 = Xi["innings"] == 1
    required_runs = pd.to_numeric(Xi.get("required_runs"), errors="coerce")
    mask_2 = (Xi["innings"] == 2) & (required_runs > 0)
    mask_2_model = _second_innings_model_mask(Xi, mask_2)
    mask_2_chased = (Xi["innings"] == 2) & (required_runs <= 0)

    if not X_first.empty:
        _assign_predicted_probabilities(
            Xi,
            mask_1,
            X_first,
            first_innings_model.predict_proba(X_first)[:, 1],
        )
    if not X_second.empty:
        _assign_predicted_probabilities(
            Xi,
            mask_2_model,
            X_second,
            second_innings_model.predict_proba(X_second)[:, 1],
        )
    Xi.loc[mask_2_chased, "y_prob"] = 1.0

    terminal_second = _second_innings_terminal_probability(Xi, match_row)
    if terminal_second is not None:
        terminal_idx, terminal_prob = terminal_second
        Xi.loc[terminal_idx, "y_prob"] = float(terminal_prob)

    Xi["y_prob"] = Xi.groupby(["match_id", "innings"], sort=False)["y_prob"].ffill()
    Xi["y_prob"] = Xi["y_prob"].fillna(0.5)

    innings_num = pd.to_numeric(Xi.get("innings"), errors="coerce")
    regular_mask = innings_num.isin([1, 2])
    Xi_regular = Xi.loc[regular_mask].copy()
    Xi_extra = Xi.loc[~regular_mask].copy()

    if X_first.empty:
        scored_regular = ipl.calculate_impact(Xi_regular) if not Xi_regular.empty else Xi_regular
    else:
        baseline_by_match = {
            match_id: ipl.get_baseline(match_frame, first_innings_model)
            for match_id, match_frame in X_first.groupby("match_id", sort=False)
        }

        class _CachedBaselineModel:
            def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
                match_id = frame["match_id"].iloc[0]
                baseline = float(baseline_by_match.get(match_id, 0.5))
                return np.array([[1 - baseline, baseline]])

        scored_regular = (
            ipl.calculate_impact(Xi_regular, model=_CachedBaselineModel())
            if not Xi_regular.empty
            else Xi_regular
        )

    if not Xi_extra.empty:
        Xi_extra["y_prob"] = pd.to_numeric(Xi_extra.get("y_prob"), errors="coerce").fillna(0.5)
        for col in ["prob_bowling_diff", "prob_batting_diff", "bowler_delta", "batter_delta"]:
            Xi_extra[col] = 0.0
        return pd.concat([scored_regular, Xi_extra], ignore_index=True, sort=False)

    return scored_regular


def score_matches_probabilities_bulk(
    match_balls: pd.DataFrame,
    matches_with_prior: pd.DataFrame,
    match_rows: pd.DataFrame,
    first_innings_model,
    second_innings_model,
) -> pd.DataFrame:
    Xi = _sort_match_balls(match_balls.copy())
    Xi["y_prob"] = np.nan

    X_first = ipl.preprocess_first_innings(Xi, matches_with_prior).copy()
    X_second = ipl.preprocess_second_innings(Xi, matches_with_prior).copy()

    mask_1 = Xi["innings"] == 1
    required_runs = pd.to_numeric(Xi.get("required_runs"), errors="coerce")
    mask_2 = (Xi["innings"] == 2) & (required_runs > 0)
    mask_2_model = _second_innings_model_mask(Xi, mask_2)
    mask_2_chased = (Xi["innings"] == 2) & (required_runs <= 0)

    if not X_first.empty:
        _assign_predicted_probabilities(
            Xi,
            mask_1,
            X_first,
            first_innings_model.predict_proba(X_first)[:, 1],
        )
    if not X_second.empty:
        _assign_predicted_probabilities(
            Xi,
            mask_2_model,
            X_second,
            second_innings_model.predict_proba(X_second)[:, 1],
        )
    Xi.loc[mask_2_chased, "y_prob"] = 1.0

    matches_by_id = match_rows.copy()
    matches_by_id["match_id"] = pd.to_numeric(matches_by_id["match_id"], errors="coerce").astype("Int64")
    matches_by_id = matches_by_id.dropna(subset=["match_id"]).drop_duplicates("match_id").set_index("match_id")
    for match_id, group_index in Xi.groupby("match_id", sort=False).groups.items():
        match_key = pd.to_numeric(pd.Series([match_id]), errors="coerce").iloc[0]
        if pd.isna(match_key) or int(match_key) not in matches_by_id.index:
            continue
        terminal_second = _second_innings_terminal_probability(
            Xi.loc[group_index],
            matches_by_id.loc[int(match_key)],
        )
        if terminal_second is not None:
            terminal_idx, terminal_prob = terminal_second
            Xi.loc[terminal_idx, "y_prob"] = float(terminal_prob)

    Xi["y_prob"] = Xi.groupby(["match_id", "innings"], sort=False)["y_prob"].ffill()
    Xi["y_prob"] = Xi["y_prob"].fillna(0.5)

    innings_num = pd.to_numeric(Xi.get("innings"), errors="coerce")
    regular_mask = innings_num.isin([1, 2])
    Xi_regular = Xi.loc[regular_mask].copy()
    Xi_extra = Xi.loc[~regular_mask].copy()

    if X_first.empty:
        scored_regular = ipl.calculate_impact(Xi_regular) if not Xi_regular.empty else Xi_regular
    else:
        baseline_by_match = {
            match_id: ipl.get_baseline(match_frame, first_innings_model)
            for match_id, match_frame in X_first.groupby("match_id", sort=False)
        }

        class _CachedBaselineModel:
            def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
                match_id = frame["match_id"].iloc[0]
                baseline = float(baseline_by_match.get(match_id, 0.5))
                return np.array([[1 - baseline, baseline]])

        scored_regular = (
            ipl.calculate_impact(Xi_regular, model=_CachedBaselineModel())
            if not Xi_regular.empty
            else Xi_regular
        )

    if not Xi_extra.empty:
        Xi_extra["y_prob"] = pd.to_numeric(Xi_extra.get("y_prob"), errors="coerce").fillna(0.5)
        for col in ["prob_bowling_diff", "prob_batting_diff", "bowler_delta", "batter_delta"]:
            Xi_extra[col] = 0.0
        return pd.concat([scored_regular, Xi_extra], ignore_index=True, sort=False)

    return scored_regular


def reconstruct_cached_impact_columns(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.copy()
    innings = pd.to_numeric(X.get("innings"), errors="coerce")
    prob_bowling = pd.to_numeric(X.get("prob_bowling_diff"), errors="coerce")
    prob_batting = pd.to_numeric(X.get("prob_batting_diff"), errors="coerce")
    batter_runs = pd.to_numeric(X.get("batter_runs"), errors="coerce").fillna(0)
    extra_type = X.get("extra_type", pd.Series(np.nan, index=X.index))
    wicket_taken = pd.to_numeric(X.get("wicket_taken"), errors="coerce").fillna(0)

    X["prob_bowling_diff"] = prob_bowling
    X["prob_batting_diff"] = prob_batting
    X["bowler_delta"] = np.where(innings == 1, prob_bowling, -prob_bowling)
    X["batter_delta"] = np.where(
        extra_type.isna() | (batter_runs > 0),
        np.where(innings == 1, prob_batting, prob_bowling),
        0,
    )
    X.loc[wicket_taken == 1, "batter_delta"] = 0
    return _sort_match_balls(X)


def write_cached_match_predictions(
    scored_balls: pd.DataFrame,
    *,
    db_path: Path | str = DB_PATH,
    model_version: str | None = None,
    prediction_ts: str | None = None,
) -> int:
    create_schema(db_path)
    model_version = model_version or current_model_version()
    now_ts = prediction_ts or utc_now_iso()
    required = ["match_id", "ball_id", "y_prob", "prob_bowling_diff", "prob_batting_diff"]
    missing = [col for col in required if col not in scored_balls.columns]
    if missing:
        raise ValueError(f"Cannot cache win probabilities; missing columns: {missing}")

    out = scored_balls.copy()
    out["ball_id"] = out["ball_id"].fillna("").astype(str).str.strip()
    if out["ball_id"].eq("").any() or out["ball_id"].duplicated().any():
        raise ValueError("Cannot cache win probabilities without unique non-empty ball_id values.")
    match_ids = pd.to_numeric(out["match_id"], errors="coerce").dropna().astype(int).unique().tolist()
    if len(match_ids) != 1:
        raise ValueError("write_cached_match_predictions expects exactly one match_id.")
    match_id = int(match_ids[0])

    y_prob = pd.to_numeric(out["y_prob"], errors="coerce")
    prob_bowling = pd.to_numeric(out["prob_bowling_diff"], errors="coerce")
    prob_batting = pd.to_numeric(out["prob_batting_diff"], errors="coerce")
    if y_prob.isna().any() or prob_bowling.isna().any() or prob_batting.isna().any():
        raise ValueError("Cannot cache win probabilities with null probability or impact delta values.")

    rows = pd.DataFrame(
        {
            "prediction_id": [_prediction_id(ball_id) for ball_id in out["ball_id"]],
            "match_id": str(match_id),
            "ball_id": out["ball_id"].tolist(),
            "model_type": WIN_PROBABILITY_MODEL_TYPE,
            "model_version": str(model_version),
            "prediction_ts": now_ts,
            "batting_team_win_prob": (1 - y_prob).astype(float).tolist(),
            "bowling_team_win_prob": y_prob.astype(float).tolist(),
            "prob_bowling_diff": prob_bowling.astype(float).tolist(),
            "prob_batting_diff": prob_batting.astype(float).tolist(),
            "metadata_json": json.dumps({"source": "win_probability_cache"}),
            "created_at": now_ts,
            "updated_at": now_ts,
        },
        columns=WIN_PROBABILITY_CACHE_COLUMNS,
    )

    with transaction(db_path) as conn:
        conn.execute(
            "DELETE FROM prediction_history WHERE model_type = ? AND CAST(match_id AS INTEGER) = ?;",
            (WIN_PROBABILITY_MODEL_TYPE, match_id),
        )
        return upsert_dataframe(
            conn=conn,
            table_name="prediction_history",
            df=rows,
            conflict_columns=["prediction_id"],
            update_columns=[col for col in WIN_PROBABILITY_CACHE_COLUMNS if col != "prediction_id"],
        )


def _cached_prediction_rows(
    scored_balls: pd.DataFrame,
    *,
    model_version: str,
    prediction_ts: str,
) -> pd.DataFrame:
    required = ["match_id", "ball_id", "y_prob", "prob_bowling_diff", "prob_batting_diff"]
    missing = [col for col in required if col not in scored_balls.columns]
    if missing:
        raise ValueError(f"Cannot cache win probabilities; missing columns: {missing}")

    out = scored_balls.copy()
    out["ball_id"] = out["ball_id"].fillna("").astype(str).str.strip()
    if out["ball_id"].eq("").any() or out["ball_id"].duplicated().any():
        raise ValueError("Cannot cache win probabilities without unique non-empty ball_id values.")

    y_prob = pd.to_numeric(out["y_prob"], errors="coerce")
    prob_bowling = pd.to_numeric(out["prob_bowling_diff"], errors="coerce")
    prob_batting = pd.to_numeric(out["prob_batting_diff"], errors="coerce")
    if y_prob.isna().any() or prob_bowling.isna().any() or prob_batting.isna().any():
        raise ValueError("Cannot cache win probabilities with null probability or impact delta values.")

    return pd.DataFrame(
        {
            "prediction_id": [_prediction_id(ball_id) for ball_id in out["ball_id"]],
            "match_id": pd.to_numeric(out["match_id"], errors="raise").astype(int).astype(str).tolist(),
            "ball_id": out["ball_id"].tolist(),
            "model_type": WIN_PROBABILITY_MODEL_TYPE,
            "model_version": str(model_version),
            "prediction_ts": prediction_ts,
            "batting_team_win_prob": (1 - y_prob).astype(float).tolist(),
            "bowling_team_win_prob": y_prob.astype(float).tolist(),
            "prob_bowling_diff": prob_bowling.astype(float).tolist(),
            "prob_batting_diff": prob_batting.astype(float).tolist(),
            "metadata_json": json.dumps({"source": "win_probability_cache"}),
            "created_at": prediction_ts,
            "updated_at": prediction_ts,
        },
        columns=WIN_PROBABILITY_CACHE_COLUMNS,
    )


def _delete_win_probability_rows_for_matches(conn: sqlite3.Connection, match_ids: list[int]) -> None:
    if not match_ids:
        return
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _wp_refresh_match_ids (match_id INTEGER PRIMARY KEY);")
    conn.execute("DELETE FROM _wp_refresh_match_ids;")
    conn.executemany(
        "INSERT OR IGNORE INTO _wp_refresh_match_ids (match_id) VALUES (?);",
        [(int(match_id),) for match_id in match_ids],
    )
    conn.execute(
        """
        DELETE FROM prediction_history
        WHERE model_type = ?
          AND CAST(match_id AS INTEGER) IN (SELECT match_id FROM _wp_refresh_match_ids);
        """,
        (WIN_PROBABILITY_MODEL_TYPE,),
    )
    conn.execute("DROP TABLE _wp_refresh_match_ids;")


def write_cached_predictions_bulk(
    scored_balls: pd.DataFrame,
    *,
    db_path: Path | str = DB_PATH,
    match_ids: list[int] | None = None,
    model_version: str | None = None,
    prediction_ts: str | None = None,
) -> int:
    create_schema(db_path)
    model_version = model_version or current_model_version()
    now_ts = prediction_ts or utc_now_iso()
    rows = _cached_prediction_rows(
        scored_balls,
        model_version=model_version,
        prediction_ts=now_ts,
    )
    if match_ids is None:
        match_ids = pd.to_numeric(scored_balls["match_id"], errors="raise").astype(int).unique().tolist()

    with transaction(db_path) as conn:
        _delete_win_probability_rows_for_matches(conn, [int(match_id) for match_id in match_ids])
        return upsert_dataframe(
            conn=conn,
            table_name="prediction_history",
            df=rows,
            conflict_columns=["prediction_id"],
            update_columns=[col for col in WIN_PROBABILITY_CACHE_COLUMNS if col != "prediction_id"],
        )


def load_cached_prediction_impact(
    match_balls: pd.DataFrame,
    *,
    db_path: Path | str = DB_PATH,
    match_id: int,
    model_version: str | None = None,
    allow_stale_model_version: bool = False,
) -> pd.DataFrame | None:
    if match_balls.empty or "ball_id" not in match_balls.columns:
        return None
    try:
        create_schema(db_path)
    except Exception:
        return None

    if model_version is None:
        try:
            model_version = current_model_version()
        except Exception:
            if not allow_stale_model_version:
                return None
            model_version = None

    ball_ids = match_balls["ball_id"].fillna("").astype(str).str.strip()
    if ball_ids.eq("").any() or ball_ids.duplicated().any():
        return None

    def _load_version(conn: sqlite3.Connection, version: str) -> pd.DataFrame:
        return pd.read_sql_query(
            """
            SELECT
                ball_id,
                bowling_team_win_prob,
                prob_bowling_diff,
                prob_batting_diff
            FROM prediction_history
            WHERE model_type = ?
              AND model_version = ?
              AND CAST(match_id AS INTEGER) = ?
            """,
            conn,
            params=[WIN_PROBABILITY_MODEL_TYPE, str(version), int(match_id)],
        )

    def _align_cached(cached: pd.DataFrame) -> pd.DataFrame | None:
        if cached.empty:
            return None

        cached = cached.copy()
        cached["ball_id"] = cached["ball_id"].fillna("").astype(str).str.strip()
        cached = cached.drop_duplicates("ball_id", keep="last").set_index("ball_id")
        expected = ball_ids.tolist()
        if not set(expected).issubset(set(cached.index)):
            return None

        aligned = cached.reindex(expected)
        required_cols = ["bowling_team_win_prob", "prob_bowling_diff", "prob_batting_diff"]
        for col in required_cols:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        if aligned[required_cols].isna().any().any():
            return None

        out = match_balls.copy()
        out["y_prob"] = aligned["bowling_team_win_prob"].to_numpy()
        out["prob_bowling_diff"] = aligned["prob_bowling_diff"].to_numpy()
        out["prob_batting_diff"] = aligned["prob_batting_diff"].to_numpy()
        return reconstruct_cached_impact_columns(out)

    with sqlite3.connect(db_path) as conn:
        if model_version is not None:
            cached = _load_version(conn, str(model_version))
            aligned = _align_cached(cached)
            if aligned is not None:
                return aligned

        if not allow_stale_model_version:
            return None

        versions = pd.read_sql_query(
            """
            SELECT
                model_version,
                MAX(COALESCE(updated_at, created_at, prediction_ts)) AS latest_prediction_ts
            FROM prediction_history
            WHERE model_type = ?
              AND CAST(match_id AS INTEGER) = ?
              AND (? IS NULL OR model_version != ?)
            GROUP BY model_version
            ORDER BY latest_prediction_ts DESC
            """,
            conn,
            params=[
                WIN_PROBABILITY_MODEL_TYPE,
                int(match_id),
                None if model_version is None else str(model_version),
                None if model_version is None else str(model_version),
            ],
        )

        for stale_version in versions.get("model_version", pd.Series(dtype=str)).dropna().astype(str):
            cached = _load_version(conn, stale_version)
            aligned = _align_cached(cached)
            if aligned is not None:
                return aligned

        return None


def load_cached_prediction_impacts_bulk(
    match_balls: pd.DataFrame,
    *,
    db_path: Path | str = DB_PATH,
    match_ids: list[int],
    model_version: str | None = None,
    allow_stale_model_version: bool = False,
) -> dict[int, pd.DataFrame]:
    if match_balls.empty or "ball_id" not in match_balls.columns or not match_ids:
        return {}
    try:
        create_schema(db_path)
    except Exception:
        return {}

    if model_version is None:
        try:
            model_version = current_model_version()
        except Exception:
            if not allow_stale_model_version:
                return {}
            model_version = None

    balls = match_balls.copy()
    balls["match_id"] = pd.to_numeric(balls["match_id"], errors="coerce").astype("Int64")
    balls["ball_id"] = balls["ball_id"].fillna("").astype(str).str.strip()
    balls = balls.loc[balls["match_id"].notna() & balls["ball_id"].ne("")].copy()
    if balls.empty:
        return {}

    requested_ids = sorted({int(match_id) for match_id in match_ids})
    placeholders = ",".join(["?"] * len(requested_ids))
    with sqlite3.connect(db_path) as conn:
        cached = pd.read_sql_query(
            f"""
            SELECT
                CAST(match_id AS INTEGER) AS match_id,
                ball_id,
                model_version,
                bowling_team_win_prob,
                prob_bowling_diff,
                prob_batting_diff,
                COALESCE(updated_at, created_at, prediction_ts) AS prediction_updated_at
            FROM prediction_history
            WHERE model_type = ?
              AND CAST(match_id AS INTEGER) IN ({placeholders})
            """,
            conn,
            params=[WIN_PROBABILITY_MODEL_TYPE, *requested_ids],
        )
    if cached.empty:
        return {}

    cached["match_id"] = pd.to_numeric(cached["match_id"], errors="coerce").astype("Int64")
    cached["ball_id"] = cached["ball_id"].fillna("").astype(str).str.strip()
    cached["model_version"] = cached["model_version"].fillna("").astype(str)

    out_by_match: dict[int, pd.DataFrame] = {}
    required_cols = ["bowling_team_win_prob", "prob_bowling_diff", "prob_batting_diff"]

    for match_id, match_frame in balls.groupby("match_id", sort=False):
        match_id_int = int(match_id)
        if match_frame["ball_id"].duplicated().any():
            continue

        cached_match = cached.loc[cached["match_id"].eq(match_id_int)].copy()
        if cached_match.empty:
            continue

        version_order: list[str] = []
        if model_version is not None:
            version_order.append(str(model_version))
        if allow_stale_model_version:
            stale_versions = (
                cached_match.loc[
                    cached_match["model_version"].ne(str(model_version)) if model_version is not None else cached_match["model_version"].ne(""),
                    ["model_version", "prediction_updated_at"],
                ]
                .groupby("model_version", as_index=False)["prediction_updated_at"]
                .max()
                .sort_values("prediction_updated_at", ascending=False, kind="mergesort")["model_version"]
                .dropna()
                .astype(str)
                .tolist()
            )
            version_order.extend([version for version in stale_versions if version not in version_order])

        expected = match_frame["ball_id"].tolist()
        for version in version_order:
            version_rows = cached_match.loc[cached_match["model_version"].eq(version)].copy()
            if version_rows.empty:
                continue
            version_rows = version_rows.drop_duplicates("ball_id", keep="last").set_index("ball_id")
            if not set(expected).issubset(set(version_rows.index)):
                continue
            aligned = version_rows.reindex(expected)
            for col in required_cols:
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            if aligned[required_cols].isna().any().any():
                continue
            scored = match_frame.copy()
            scored["y_prob"] = aligned["bowling_team_win_prob"].to_numpy()
            scored["prob_bowling_diff"] = aligned["prob_bowling_diff"].to_numpy()
            scored["prob_batting_diff"] = aligned["prob_batting_diff"].to_numpy()
            out_by_match[match_id_int] = reconstruct_cached_impact_columns(scored)
            break

    return out_by_match


def _load_match_and_balls(db_path: Path | str, match_id: int) -> tuple[pd.Series | None, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        match = pd.read_sql_query(
            "SELECT * FROM match_list WHERE CAST(match_id AS INTEGER) = ? LIMIT 1",
            conn,
            params=[int(match_id)],
        )
        balls = pd.read_sql_query(
            "SELECT * FROM ball_by_ball WHERE CAST(match_id AS INTEGER) = ?",
            conn,
            params=[int(match_id)],
        )
    match_row = None if match.empty else match.iloc[0]
    return match_row, balls


def _load_matches_and_balls(db_path: Path | str, match_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not match_ids:
        return pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TEMP TABLE _wp_refresh_match_ids (match_id INTEGER PRIMARY KEY);")
        conn.executemany(
            "INSERT OR IGNORE INTO _wp_refresh_match_ids (match_id) VALUES (?);",
            [(int(match_id),) for match_id in match_ids],
        )
        match_rows = pd.read_sql_query(
            """
            SELECT ml.*
            FROM match_list ml
            JOIN _wp_refresh_match_ids ids
              ON CAST(ml.match_id AS INTEGER) = ids.match_id
            ORDER BY CAST(ml.year AS INTEGER), ml.date, CAST(ml.match_id AS INTEGER);
            """,
            conn,
        )
        balls = pd.read_sql_query(
            """
            SELECT b.*
            FROM ball_by_ball b
            JOIN _wp_refresh_match_ids ids
              ON CAST(b.match_id AS INTEGER) = ids.match_id
            ORDER BY
                CAST(b.match_id AS INTEGER),
                CAST(b.innings AS INTEGER),
                CAST(b.team_balls AS INTEGER),
                CAST(b.over AS REAL),
                CAST(b.ball AS REAL),
                CAST(b.id AS INTEGER);
            """,
            conn,
        )
        conn.execute("DROP TABLE _wp_refresh_match_ids;")
    return match_rows, balls


def _refresh_match_with_context(
    *,
    db_path: Path | str,
    match_id: int,
    model_version: str,
    matches_with_prior: pd.DataFrame,
    first_innings_model,
    second_innings_model,
    force: bool,
) -> dict[str, Any]:
    match_row, balls = _load_match_and_balls(db_path, match_id)
    if match_row is None:
        return {"match_id": int(match_id), "status": "skipped", "skipped": True, "skip_reason": "match_not_found", "rows_written": 0}
    if str(match_row.get("status") or "").strip().lower() != "complete":
        return {"match_id": int(match_id), "status": "skipped", "skipped": True, "skip_reason": "not_complete", "rows_written": 0}
    if balls.empty:
        return {"match_id": int(match_id), "status": "skipped", "skipped": True, "skip_reason": "no_ball_rows", "rows_written": 0}
    if (not force) and load_cached_prediction_impact(
        balls,
        db_path=db_path,
        match_id=int(match_id),
        model_version=model_version,
    ) is not None:
        return {"match_id": int(match_id), "status": "skipped", "skipped": True, "skip_reason": "cache_current", "rows_written": len(balls)}

    scored = score_match_probabilities(
        balls,
        matches_with_prior,
        match_row,
        first_innings_model,
        second_innings_model,
    )
    rows_written = write_cached_match_predictions(
        scored,
        db_path=db_path,
        model_version=model_version,
    )
    return {
        "match_id": int(match_id),
        "status": "success",
        "skipped": False,
        "model_version": model_version,
        "rows_written": int(rows_written),
    }


def refresh_win_probabilities_for_match(
    *,
    match_id: int,
    db_path: Path | str = DB_PATH,
    force: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    try:
        create_schema(db_path)
        with sqlite3.connect(db_path) as conn:
            all_matches = pd.read_sql_query("SELECT * FROM match_list", conn)
        matches_with_prior = ipl.prior_match_stats(all_matches)
        first_innings_model = ipl.first_innings_glm_load()
        second_innings_model = ipl.second_innings_glm_load()
        model_version = current_model_version()
        return _refresh_match_with_context(
            db_path=db_path,
            match_id=int(match_id),
            model_version=model_version,
            matches_with_prior=matches_with_prior,
            first_innings_model=first_innings_model,
            second_innings_model=second_innings_model,
            force=force,
        )
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "match_id": int(match_id),
            "status": "failed",
            "skipped": False,
            "rows_written": 0,
            "error_message": f"{type(exc).__name__}: {exc}",
        }


def _complete_match_ids(db_path: Path | str, *, season: int | None, all_seasons: bool) -> pd.DataFrame:
    clauses = ["LOWER(TRIM(COALESCE(status, ''))) = 'complete'"]
    params: list[Any] = []
    if not all_seasons:
        if season is None:
            raise ValueError("Provide --season or --all-seasons.")
        clauses.append("CAST(year AS INTEGER) = ?")
        params.append(int(season))
    query = f"""
        SELECT CAST(match_id AS INTEGER) AS match_id, CAST(year AS INTEGER) AS season
        FROM match_list
        WHERE {' AND '.join(clauses)}
        ORDER BY season, date, match_id
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def refresh_win_probabilities(
    *,
    db_path: Path | str = DB_PATH,
    season: int | None = None,
    all_seasons: bool = False,
    force: bool = False,
    refresh_team_metrics: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    create_schema(db_path)
    matches = _complete_match_ids(db_path, season=season, all_seasons=all_seasons)
    if matches.empty:
        return {
            "artifact": "win_probabilities",
            "status": "skipped",
            "skipped": True,
            "matches_total": 0,
            "matches_succeeded": 0,
            "matches_failed": 0,
            "rows_written": 0,
            "model_version": None,
        }

    try:
        with sqlite3.connect(db_path) as conn:
            all_matches = pd.read_sql_query("SELECT * FROM match_list", conn)
        matches_with_prior = ipl.prior_match_stats(all_matches)
        first_innings_model = ipl.first_innings_glm_load()
        second_innings_model = ipl.second_innings_glm_load()
        model_version = current_model_version()

        match_ids = matches["match_id"].dropna().astype(int).tolist()
        if force:
            match_rows, balls = _load_matches_and_balls(db_path, match_ids)
            if balls.empty:
                return {
                    "artifact": "win_probabilities",
                    "status": "skipped",
                    "skipped": True,
                    "refresh_mode": "bulk",
                    "matches_total": int(len(matches)),
                    "matches_succeeded": 0,
                    "matches_failed": 0,
                    "rows_written": 0,
                    "model_version": model_version,
                    "team_profile_metrics_refresh": [],
                    "match_results": [
                        {
                            "match_id": int(match_id),
                            "status": "skipped",
                            "skipped": True,
                            "skip_reason": "no_ball_rows",
                            "rows_written": 0,
                        }
                        for match_id in match_ids
                    ],
                }

            scored = score_matches_probabilities_bulk(
                balls,
                matches_with_prior,
                match_rows,
                first_innings_model,
                second_innings_model,
            )
            rows_written = write_cached_predictions_bulk(
                scored,
                db_path=db_path,
                match_ids=match_ids,
                model_version=model_version,
            )

            rows_by_match = (
                scored.assign(match_id=pd.to_numeric(scored["match_id"], errors="coerce"))
                .dropna(subset=["match_id"])
                .groupby("match_id")
                .size()
                .astype(int)
                .to_dict()
            )
            results = [
                {
                    "match_id": int(match_id),
                    "status": "success" if int(rows_by_match.get(match_id, 0)) else "skipped",
                    "skipped": not bool(int(rows_by_match.get(match_id, 0))),
                    "skip_reason": None if int(rows_by_match.get(match_id, 0)) else "no_ball_rows",
                    "model_version": model_version if int(rows_by_match.get(match_id, 0)) else None,
                    "rows_written": int(rows_by_match.get(match_id, 0)),
                }
                for match_id in match_ids
            ]

            team_metric_results = []
            if refresh_team_metrics:
                from .team_profile_metrics import refresh_team_profile_metrics_for_season

                seasons = sorted(matches["season"].dropna().astype(int).unique().tolist())
                for season_value in seasons:
                    team_metric_results.append(
                        refresh_team_profile_metrics_for_season(
                            season=int(season_value),
                            db_path=db_path,
                            force=force,
                            raise_on_error=raise_on_error,
                        )
                    )

            return {
                "artifact": "win_probabilities",
                "status": "success",
                "skipped": False,
                "refresh_mode": "bulk",
                "matches_total": int(len(matches)),
                "matches_succeeded": int(sum(1 for result in results if str(result.get("status") or "").lower() in {"success", "skipped"})),
                "matches_failed": 0,
                "rows_written": int(rows_written),
                "model_version": model_version,
                "team_profile_metrics_refresh": team_metric_results,
                "match_results": results,
            }

        results = []
        rows_written = 0
        failed = 0
        for match_id in match_ids:
            result = _refresh_match_with_context(
                db_path=db_path,
                match_id=int(match_id),
                model_version=model_version,
                matches_with_prior=matches_with_prior,
                first_innings_model=first_innings_model,
                second_innings_model=second_innings_model,
                force=force,
            )
            results.append(result)
            rows_written += int(result.get("rows_written") or 0) if not result.get("skipped") else 0
            if str(result.get("status") or "").lower() == "failed":
                failed += 1

        team_metric_results = []
        if refresh_team_metrics:
            from .team_profile_metrics import refresh_team_profile_metrics_for_season

            seasons = sorted(matches["season"].dropna().astype(int).unique().tolist())
            for season_value in seasons:
                team_metric_results.append(
                    refresh_team_profile_metrics_for_season(
                        season=int(season_value),
                        db_path=db_path,
                        force=force,
                        raise_on_error=raise_on_error,
                    )
                )

        return {
            "artifact": "win_probabilities",
            "status": "failed" if failed else "success",
            "skipped": False,
            "matches_total": int(len(matches)),
            "matches_succeeded": int(sum(1 for result in results if str(result.get("status") or "").lower() in {"success", "skipped"})),
            "matches_failed": int(failed),
            "rows_written": int(rows_written),
            "model_version": model_version,
            "team_profile_metrics_refresh": team_metric_results,
            "match_results": results,
        }
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "artifact": "win_probabilities",
            "status": "failed",
            "skipped": False,
            "matches_total": int(len(matches)),
            "matches_succeeded": 0,
            "matches_failed": int(len(matches)),
            "rows_written": 0,
            "error_message": f"{type(exc).__name__}: {exc}",
        }
