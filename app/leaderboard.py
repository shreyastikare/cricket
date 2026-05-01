from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sqlite3
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ipl

try:
    from ipl.storage.win_probability_cache import load_cached_prediction_impact, load_cached_prediction_impacts_bulk
except ImportError:
    load_cached_prediction_impact = None
    load_cached_prediction_impacts_bulk = None

try:
    from plot_theme import apply_plot_theme
    from theme_config import (
        PLOTLY_AXIS_TICK_FONT_SIZE,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_COLORS,
        PLOTLY_PALETTE,
        PLOTLY_FONT_FAMILY,
        PLOTLY_HEADER_PLOT_TOP,
        PLOTLY_HEADER_MARGIN_LEFT,
        PLOTLY_HEADER_MARGIN_RIGHT,
        PLOTLY_HEADER_MARGIN_TOP,
        PLOTLY_HEADER_TITLE_Y,
        PLOTLY_LABEL_FONT_SIZE,
        PLOTLY_LEGEND_FONT_SIZE,
    )
except ModuleNotFoundError:
    from app.plot_theme import apply_plot_theme
    from app.theme_config import (
        PLOTLY_AXIS_TICK_FONT_SIZE,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_COLORS,
        PLOTLY_PALETTE,
        PLOTLY_FONT_FAMILY,
        PLOTLY_HEADER_PLOT_TOP,
        PLOTLY_HEADER_MARGIN_LEFT,
        PLOTLY_HEADER_MARGIN_RIGHT,
        PLOTLY_HEADER_MARGIN_TOP,
        PLOTLY_HEADER_TITLE_Y,
        PLOTLY_LABEL_FONT_SIZE,
        PLOTLY_LEGEND_FONT_SIZE,
    )


DB_PATH = os.getenv('DB_PATH', 'data/sqlite/ipl.db')
FINAL_MATCH_STATUSES = {"complete", "abandoned", "no_result"}
NO_RESULT_TYPES = {"no result", "no_result", "abandoned"}
ENABLE_LEADERBOARD_IMPACT_PARITY_CHECK = False


def _safe_divide(numerator, denominator):
    num = pd.to_numeric(numerator, errors="coerce").fillna(0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0)
    den = np.where(den == 0, 1, den)
    return num / den


def _norm_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def get_finalized_season_options() -> list[dict[str, int]]:
    query = """
        SELECT DISTINCT year
        FROM match_list
        WHERE year IS NOT NULL
          AND LOWER(TRIM(COALESCE(status, ''))) IN ('complete', 'abandoned', 'no_result')
        ORDER BY year DESC
    """
    with _connect() as conn:
        rows = conn.execute(query).fetchall()
    return [{"label": str(int(row[0])), "value": int(row[0])} for row in rows]


def get_default_leaderboard_season() -> int | None:
    options = get_finalized_season_options()
    if options:
        return int(options[0]["value"])
    with _connect() as conn:
        row = conn.execute("SELECT MAX(year) FROM match_list WHERE year IS NOT NULL").fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


@lru_cache(maxsize=1)
def _load_win_probability_models():
    return ipl.first_innings_glm_load(), ipl.second_innings_glm_load()


class Leaderboard:
    def __init__(
        self,
        season: int | None = None,
        db_path: Path | str | None = None,
        *,
        include_impact: bool = True,
    ):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        if season is None:
            default_season = get_default_leaderboard_season()
            if default_season is None:
                raise ValueError("No IPL seasons available in match_list")
            season = default_season
        self.season = int(season)

        self.matches = self._load_matches_for_season(self.season)
        if self.matches.empty:
            raise ValueError(f"No finalized matches found for season {self.season}")

        self.match_ids = self.matches["match_id"].astype(int).tolist()
        self.balls = self._load_balls_for_matches(self.match_ids)
        self.balls = self._prepare_ball_data(self.balls)

        self._batters_all_time, self._batters_by_season = ipl.batter_info(self.balls)
        self._bowlers_all_time, self._bowlers_by_season = ipl.bowler_info(self.balls)
        self._player_meta = self._build_player_meta()
        if include_impact:
            self._impact_balls, self._impact_table, self._impact_by_key = self._build_impact()
        else:
            self._impact_balls, self._impact_table, self._impact_by_key = self._empty_impact()

        self.batter_stats = self._build_batter_stats()
        self.bowler_stats = self._build_bowler_stats()
        self.player_impact_stats = self._build_player_impact_stats()
        self.team_ranking = self._build_team_ranking()

    def _attach_player_identity(self, frame: pd.DataFrame, role: str) -> pd.DataFrame:
        out = frame.copy()
        raw_name = out.get(role, pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
        player_id_col = f"{role}_player_id"
        canonical_col = f"{role}_canonical_name"

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

    def _load_matches_for_season(self, season: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM match_list
            WHERE year = ?
              AND LOWER(TRIM(COALESCE(status, ''))) IN ('complete', 'abandoned', 'no_result')
            ORDER BY date, match_id
        """
        with self._connect() as conn:
            matches = pd.read_sql_query(query, conn, params=[int(season)])
        matches["match_id"] = pd.to_numeric(matches["match_id"], errors="coerce").astype("Int64")
        matches = matches.dropna(subset=["match_id"]).copy()
        matches["match_id"] = matches["match_id"].astype(int)
        return matches

    def _load_balls_for_matches(self, match_ids: list[int]) -> pd.DataFrame:
        if not match_ids:
            return pd.DataFrame()
        placeholders = ",".join(["?"] * len(match_ids))
        query = f"SELECT * FROM ball_by_ball WHERE match_id IN ({placeholders})"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=match_ids)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _empty_impact(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        impact_balls = self.balls.copy()
        for col in ["y_prob", "prob_bowling_diff", "prob_batting_diff", "bowler_delta", "batter_delta"]:
            impact_balls[col] = 0.0

        impact_table = pd.DataFrame(columns=["Player", "Team", "Batting Impact", "Bowling Impact", "Total Impact"])
        impact_by_key = pd.DataFrame(columns=["player_key", "Player", "Batting Impact", "Bowling Impact", "Total Impact"])
        return impact_balls, impact_table, impact_by_key

    def _prepare_ball_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        X = frame.copy()
        X = self._attach_player_identity(X, "batter")
        X = self._attach_player_identity(X, "bowler")
        X = self._attach_player_identity(X, "player_out")

        if "year" not in X.columns:
            X["year"] = pd.to_datetime(X.get("date"), errors="coerce").dt.year
        else:
            inferred_year = pd.to_datetime(X.get("date"), errors="coerce").dt.year
            X["year"] = pd.to_numeric(X["year"], errors="coerce").fillna(inferred_year)
        X["year"] = X["year"].fillna(0).astype(int)

        X["innings"] = pd.to_numeric(X.get("innings"), errors="coerce").fillna(0).astype(int)
        X["runs_batter"] = pd.to_numeric(X.get("runs_batter"), errors="coerce").fillna(0)
        X["runs_bowler"] = pd.to_numeric(X.get("runs_bowler"), errors="coerce").fillna(0)
        X["valid_ball"] = pd.to_numeric(X.get("valid_ball"), errors="coerce").fillna(0).astype(int)
        X["over"] = pd.to_numeric(X.get("over"), errors="coerce").fillna(-1).astype(int)

        runs_not_boundary = X.get("runs_not_boundary", pd.Series(False, index=X.index)).fillna(False).astype(bool)
        if "four" not in X.columns:
            X["four"] = ((X["runs_batter"] == 4) & (~runs_not_boundary)).astype(int)
        else:
            X["four"] = pd.to_numeric(X["four"], errors="coerce").fillna(0).astype(int)

        if "six" not in X.columns:
            X["six"] = ((X["runs_batter"] == 6) & (~runs_not_boundary)).astype(int)
        else:
            X["six"] = pd.to_numeric(X["six"], errors="coerce").fillna(0).astype(int)

        if "dot" not in X.columns:
            X["dot"] = ((X["runs_batter"] == 0) & (X["valid_ball"] == 1)).astype(int)
        else:
            X["dot"] = pd.to_numeric(X["dot"], errors="coerce").fillna(0).astype(int)

        extra_type = X.get("extra_type", pd.Series("", index=X.index)).fillna("").astype(str).str.lower()
        wides_mask = extra_type.str.contains("wides", regex=False)
        inferred_balls_faced = pd.Series(np.where(wides_mask, 0, 1), index=X.index)
        if "balls_faced" in X.columns:
            X["balls_faced"] = pd.to_numeric(X["balls_faced"], errors="coerce").fillna(inferred_balls_faced).astype(int)
        else:
            X["balls_faced"] = inferred_balls_faced.astype(int)

        if "batter_out_in_innings" not in X.columns:
            dismissal_mask = (
                (X["innings"] <= 2)
                & X["player_out_key"].notna()
                & (X["player_out_key"].astype(str).str.strip() != "")
                & (X.get("wicket_kind", pd.Series("", index=X.index)).fillna("").astype(str).str.lower() != "retired hurt")
            )
            dismissals = (
                X.loc[dismissal_mask, ["match_id", "player_out_key"]]
                .drop_duplicates()
                .rename(columns={"player_out_key": "batter_key"})
                .assign(batter_out_in_innings=True)
            )
            X = X.merge(dismissals, on=["match_id", "batter_key"], how="left")
            X["batter_out_in_innings"] = X["batter_out_in_innings"].fillna(False).astype(bool)
        else:
            X["batter_out_in_innings"] = X["batter_out_in_innings"].fillna(False).astype(bool)

        return X

    def _build_impact(self):
        if self.balls.empty:
            empty = pd.DataFrame(columns=["Player", "Team", "Batting Impact", "Bowling Impact", "Total Impact"])
            empty_by_key = pd.DataFrame(columns=["player_key", "Player", "Batting Impact", "Bowling Impact", "Total Impact"])
            return self.balls.copy(), empty, empty_by_key

        impact_balls = self._sort_impact_rows(self.balls.copy())

        matches_with_prior = None
        first_innings_glm = None
        second_innings_glm = None

        def _ensure_model_context():
            nonlocal matches_with_prior, first_innings_glm, second_innings_glm
            if matches_with_prior is None:
                with self._connect() as conn:
                    all_matches = pd.read_sql_query("SELECT * FROM match_list", conn)
                matches_with_prior = ipl.prior_match_stats(all_matches)
            if first_innings_glm is None or second_innings_glm is None:
                try:
                    first_innings_glm, second_innings_glm = _load_win_probability_models()
                except Exception as exc:
                    warnings.warn(
                        f"Leaderboard impact scoring: win probability models unavailable, using heuristic fallback ({exc}).",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        scored_frames: list[pd.DataFrame] = []
        per_match_impact_frames: list[pd.DataFrame] = []
        status_by_match = {}
        if not self.matches.empty and {"match_id", "status"}.issubset(self.matches.columns):
            status_by_match = {
                int(row["match_id"]): _norm_text(row.get("status"))
                for _, row in self.matches.iterrows()
                if pd.notna(row.get("match_id"))
            }
        cached_scored_by_match = {}
        complete_match_ids = [
            int(match_id)
            for match_id in self.match_ids
            if status_by_match.get(int(match_id)) == "complete"
        ]
        if load_cached_prediction_impacts_bulk is not None and complete_match_ids:
            cached_scored_by_match = load_cached_prediction_impacts_bulk(
                impact_balls,
                db_path=self.db_path,
                match_ids=complete_match_ids,
                allow_stale_model_version=True,
            )
        for match_id, match_balls in impact_balls.groupby("match_id", sort=True):
            match_id_int = int(match_id)
            scored_match = cached_scored_by_match.get(match_id_int)
            if scored_match is None and load_cached_prediction_impact is not None and status_by_match.get(match_id_int) == "complete":
                scored_match = load_cached_prediction_impact(
                    match_balls.copy(),
                    db_path=self.db_path,
                    match_id=match_id_int,
                    allow_stale_model_version=True,
                )
            if scored_match is None:
                _ensure_model_context()
                scored_match = self._score_match_impact(
                    match_id=match_id,
                    match_balls=match_balls.copy(),
                    matches_with_prior=matches_with_prior,
                    first_innings_glm=first_innings_glm,
                    second_innings_glm=second_innings_glm,
                )
            scored_frames.append(scored_match)
            if ENABLE_LEADERBOARD_IMPACT_PARITY_CHECK:
                per_match_impact_frames.append(self._aggregate_impact_by_key(scored_match))

        if scored_frames:
            impact_balls = pd.concat(scored_frames, ignore_index=True)
        else:
            impact_balls = impact_balls.copy()
            impact_balls["y_prob"] = np.nan
            impact_balls = ipl.calculate_impact(impact_balls)

        impact_table = ipl.aggregate_impact(impact_balls)
        impact_by_key = self._aggregate_impact_by_key(impact_balls)

        if per_match_impact_frames:
            per_match_impact = pd.concat(per_match_impact_frames, ignore_index=True)
            per_match_impact = (
                per_match_impact.groupby("player_key", as_index=False)
                .agg(
                    {
                        "Player": "first",
                        "Batting Impact": "sum",
                        "Bowling Impact": "sum",
                        "Total Impact": "sum",
                    }
                )
            )
            self._warn_if_impact_parity_mismatch(impact_by_key, per_match_impact)

        return impact_balls, impact_table, impact_by_key

    def _sort_impact_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        sort_cols: list[str] = []
        for col in ["match_id", "innings", "team_balls", "over", "ball", "id"]:
            if col in out.columns:
                helper_col = f"_{col}_sort"
                out[helper_col] = pd.to_numeric(out[col], errors="coerce")
                sort_cols.append(helper_col)
        if not sort_cols:
            return out
        out = out.sort_values(sort_cols, kind="mergesort")
        return out.drop(columns=sort_cols)

    def _warn_prob_length_mismatch(self, *, match_id: int, innings: int, expected: int, predicted: int) -> None:
        warnings.warn(
            (
                f"Leaderboard impact scoring length mismatch for match_id={match_id}, innings={innings}: "
                f"{expected} rows in ball data vs {predicted} model predictions. "
                "Assigning probabilities to available rows only."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    def _second_innings_model_mask(self, frame: pd.DataFrame, base_mask: pd.Series) -> pd.Series:
        team_balls = pd.to_numeric(frame.get("team_balls"), errors="coerce")
        first_team_balls = set(
            team_balls.loc[frame.get("innings").eq(1) & team_balls.notna()]
            .round()
            .astype(int)
            .tolist()
        )
        return base_mask & team_balls.round().astype("Int64").isin(first_team_balls)

    def _score_match_impact(
        self,
        *,
        match_id: int,
        match_balls: pd.DataFrame,
        matches_with_prior: pd.DataFrame,
        first_innings_glm,
        second_innings_glm,
    ) -> pd.DataFrame:
        Xi = self._sort_impact_rows(match_balls.copy())
        Xi["y_prob"] = np.nan

        X_first = ipl.preprocess_first_innings(Xi, matches_with_prior).copy()
        X_second = ipl.preprocess_second_innings(Xi, matches_with_prior).copy()

        mask_1 = Xi["innings"] == 1
        required_runs = pd.to_numeric(Xi.get("required_runs"), errors="coerce")
        mask_2 = (Xi["innings"] == 2) & (required_runs > 0)
        mask_2_model = self._second_innings_model_mask(Xi, mask_2)
        mask_2_chased = (Xi["innings"] == 2) & (required_runs <= 0)

        def _assign_probs(mask: pd.Series, y_prob: np.ndarray, innings: int) -> None:
            idx = Xi.index[mask]
            if len(idx) != len(y_prob):
                self._warn_prob_length_mismatch(
                    match_id=int(match_id),
                    innings=innings,
                    expected=len(idx),
                    predicted=len(y_prob),
                )
            n = min(len(idx), len(y_prob))
            if n > 0:
                Xi.loc[idx[:n], "y_prob"] = y_prob[:n]

        model_scored = first_innings_glm is not None and second_innings_glm is not None
        if model_scored:
            try:
                if not X_first.empty:
                    y_p1 = first_innings_glm.predict_proba(X_first)[:, 1]
                    _assign_probs(mask_1, y_p1, innings=1)
                if not X_second.empty:
                    y_p2 = second_innings_glm.predict_proba(X_second)[:, 1]
                    _assign_probs(mask_2_model, y_p2, innings=2)
            except Exception as exc:
                model_scored = False
                warnings.warn(
                    (
                        f"Leaderboard impact scoring model inference failed for match_id={int(match_id)}; "
                        f"using heuristic fallback ({exc})."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

        if not model_scored:
            balls_remaining = pd.to_numeric(Xi.get("balls_remaining"), errors="coerce").fillna(1)
            heuristic = 1 - (required_runs.fillna(0) / np.maximum(balls_remaining, 1))
            heuristic = np.clip(heuristic, 0, 1)
            Xi.loc[mask_1, "y_prob"] = 0.5
            Xi.loc[Xi["innings"] == 2, "y_prob"] = heuristic.loc[Xi["innings"] == 2]

        Xi.loc[mask_2_chased, "y_prob"] = 1.0
        Xi["y_prob"] = Xi.groupby(["match_id", "innings"], sort=False)["y_prob"].ffill()
        Xi["y_prob"] = Xi["y_prob"].fillna(0.5)
        return ipl.calculate_impact(Xi)

    def _aggregate_impact_by_key(self, impact_balls: pd.DataFrame) -> pd.DataFrame:
        batter_imp = self._attach_player_identity(impact_balls, "batter")
        batter_imp = (
            batter_imp.groupby(["batter_key", "batter"], as_index=False)
            .agg({"batter_delta": "sum"})
            .rename(columns={"batter_key": "player_key", "batter": "Player", "batter_delta": "Batting Impact"})
        )

        bowler_imp = self._attach_player_identity(impact_balls, "bowler")
        bowler_imp = (
            bowler_imp.groupby(["bowler_key", "bowler"], as_index=False)
            .agg({"bowler_delta": "sum"})
            .rename(columns={"bowler_key": "player_key", "bowler": "Player", "bowler_delta": "Bowling Impact"})
        )

        impact_by_key = pd.merge(
            batter_imp[["player_key", "Player", "Batting Impact"]],
            bowler_imp[["player_key", "Player", "Bowling Impact"]],
            on="player_key",
            how="outer",
            suffixes=("_bat", "_bowl"),
        )
        impact_by_key["Player"] = np.where(
            impact_by_key["Player_bat"].notna() & (impact_by_key["Player_bat"] != ""),
            impact_by_key["Player_bat"],
            impact_by_key["Player_bowl"],
        )
        impact_by_key["Batting Impact"] = pd.to_numeric(impact_by_key["Batting Impact"], errors="coerce").fillna(0)
        impact_by_key["Bowling Impact"] = pd.to_numeric(impact_by_key["Bowling Impact"], errors="coerce").fillna(0)
        impact_by_key["Total Impact"] = impact_by_key["Batting Impact"] + impact_by_key["Bowling Impact"]
        impact_by_key = impact_by_key[["player_key", "Player", "Batting Impact", "Bowling Impact", "Total Impact"]]
        impact_by_key = impact_by_key[impact_by_key["Player"].astype(str).str.strip() != ""].copy()
        return impact_by_key

    def _warn_if_impact_parity_mismatch(
        self,
        season_impact: pd.DataFrame,
        per_match_impact: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> None:
        merged = season_impact.merge(
            per_match_impact,
            on="player_key",
            how="outer",
            suffixes=("_season", "_per_match"),
        )
        for col in ["Batting Impact", "Bowling Impact", "Total Impact"]:
            merged[f"{col}_season"] = pd.to_numeric(merged.get(f"{col}_season"), errors="coerce").fillna(0)
            merged[f"{col}_per_match"] = pd.to_numeric(merged.get(f"{col}_per_match"), errors="coerce").fillna(0)
        merged["diff"] = (
            (merged["Batting Impact_season"] - merged["Batting Impact_per_match"]).abs()
            + (merged["Bowling Impact_season"] - merged["Bowling Impact_per_match"]).abs()
            + (merged["Total Impact_season"] - merged["Total Impact_per_match"]).abs()
        )
        mismatch = merged[merged["diff"] > tolerance]
        if mismatch.empty:
            return
        sample = ", ".join(mismatch["player_key"].head(5).astype(str).tolist())
        warnings.warn(
            (
                "Leaderboard impact parity check mismatch: "
                f"{len(mismatch)} player(s) differ between season aggregate and per-match sum. "
                f"Sample keys: {sample}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    def _build_player_meta(self) -> pd.DataFrame:
        if self.balls.empty:
            return pd.DataFrame(columns=["player_key", "Player", "Team", "Matches Played"])

        batter_rows = self.balls[["match_id", "batter_key", "batter", "batting_team"]].rename(
            columns={"batter_key": "player_key", "batter": "Player", "batting_team": "Team"}
        )
        bowler_rows = self.balls[["match_id", "bowler_key", "bowler", "bowling_team"]].rename(
            columns={"bowler_key": "player_key", "bowler": "Player", "bowling_team": "Team"}
        )
        players = pd.concat([batter_rows, bowler_rows], ignore_index=True)
        players["player_key"] = players["player_key"].fillna("").astype(str).str.strip()
        players["Player"] = players["Player"].fillna("").astype(str).str.strip()
        players["Team"] = players["Team"].fillna("").astype(str).str.strip()
        players = players[(players["player_key"] != "") & (players["Player"] != "")]

        meta = (
            players.groupby("player_key", as_index=False)
            .agg(
                {
                    "Player": "first",
                    "match_id": "nunique",
                    "Team": lambda s: ", ".join(sorted({x for x in s if str(x).strip()})),
                }
            )
            .rename(columns={"match_id": "Matches Played"})
        )
        return meta

    def _build_batter_stats(self) -> pd.DataFrame:
        season_bat = self._batters_by_season[self._batters_by_season["year"] == self.season].copy()
        if season_bat.empty:
            return pd.DataFrame()

        boundaries = (
            self.balls[self.balls["innings"] <= 2]
            .groupby("batter_key", as_index=False)
            .agg({"four": "sum", "six": "sum"})
            .rename(columns={"four": "4", "six": "6"})
        )
        season_bat = season_bat.merge(boundaries, on="batter_key", how="left")
        season_bat[["4", "6"]] = season_bat[["4", "6"]].fillna(0).astype(int)

        batter_season_match = self.balls[(self.balls["innings"] <= 2) & (self.balls["year"] == self.season)].copy()
        batter_season_match["runs_batter"] = pd.to_numeric(batter_season_match.get("runs_batter"), errors="coerce").fillna(0)
        batter_season_match["balls_faced"] = pd.to_numeric(batter_season_match.get("balls_faced"), errors="coerce").fillna(0)
        batter_season_match["valid_ball"] = pd.to_numeric(batter_season_match.get("valid_ball"), errors="coerce").fillna(0)
        batter_season_match["batter_out_in_innings"] = batter_season_match.get("batter_out_in_innings", pd.Series(False, index=batter_season_match.index)).fillna(False).astype(bool)
        batter_match_scores = (
            batter_season_match.groupby(["batter_key", "match_id"], as_index=False)
            .agg(
                {
                    "runs_batter": "sum",
                    "balls_faced": "sum",
                    "batter_out_in_innings": "max",
                }
            )
            .rename(columns={"runs_batter": "match_runs", "balls_faced": "match_balls"})
        )
        if "batter_runs" in batter_season_match.columns:
            batter_season_match["batter_runs"] = pd.to_numeric(batter_season_match.get("batter_runs"), errors="coerce")
            runs_cum = (
                batter_season_match.groupby(["batter_key", "match_id"], as_index=False)["batter_runs"]
                .max()
                .rename(columns={"batter_runs": "match_runs_cum"})
            )
            batter_match_scores = batter_match_scores.merge(runs_cum, on=["batter_key", "match_id"], how="left")
            use_runs_cum = (
                batter_match_scores["match_runs_cum"].notna()
                & (batter_match_scores["match_runs_cum"] >= batter_match_scores["match_runs"])
            )
            batter_match_scores["match_runs"] = np.where(
                use_runs_cum,
                batter_match_scores["match_runs_cum"],
                batter_match_scores["match_runs"],
            )

        if "batter_balls" in batter_season_match.columns:
            batter_season_match["batter_balls"] = pd.to_numeric(batter_season_match.get("batter_balls"), errors="coerce")
            balls_cum = (
                batter_season_match.groupby(["batter_key", "match_id"], as_index=False)["batter_balls"]
                .max()
                .rename(columns={"batter_balls": "match_balls_cum"})
            )
            batter_match_scores = batter_match_scores.merge(balls_cum, on=["batter_key", "match_id"], how="left")
            use_balls_cum = (
                batter_match_scores["match_balls_cum"].notna()
                & (batter_match_scores["match_balls_cum"] >= batter_match_scores["match_balls"])
            )
            batter_match_scores["match_balls"] = np.where(
                use_balls_cum,
                batter_match_scores["match_balls_cum"],
                batter_match_scores["match_balls"],
            )

        batter_match_totals = (
            batter_match_scores.groupby("batter_key", as_index=False)[["match_runs", "match_balls"]]
            .sum()
            .rename(columns={"match_runs": "season_runs_calc", "match_balls": "season_balls_calc"})
        )
        season_bat = season_bat.merge(batter_match_totals, on="batter_key", how="left")
        use_calc_runs = season_bat["season_runs_calc"].notna() & (season_bat["season_runs_calc"] >= season_bat["runs"])
        season_bat["runs"] = np.where(use_calc_runs, season_bat["season_runs_calc"], season_bat["runs"])
        use_calc_balls = season_bat["season_balls_calc"].notna() & (season_bat["season_balls_calc"] >= season_bat["balls_faced"])
        season_bat["balls_faced"] = np.where(use_calc_balls, season_bat["season_balls_calc"], season_bat["balls_faced"])
        season_bat["runs"] = pd.to_numeric(season_bat["runs"], errors="coerce").fillna(0)
        season_bat["balls_faced"] = pd.to_numeric(season_bat["balls_faced"], errors="coerce").fillna(0)

        if batter_match_scores.empty:
            highest_score = pd.DataFrame(columns=["batter_key", "Highest Score"])
        else:
            top_scores = batter_match_scores.groupby("batter_key", as_index=False)["match_runs"].max().rename(columns={"match_runs": "highest_runs"})
            top_scores = top_scores.merge(batter_match_scores, on="batter_key", how="left")
            top_scores = top_scores[top_scores["match_runs"] == top_scores["highest_runs"]].copy()
            top_scores["hs_not_out"] = ~top_scores["batter_out_in_innings"].astype(bool)
            highest_score = (
                top_scores.groupby(["batter_key", "highest_runs"], as_index=False)["hs_not_out"]
                .max()
            )
            highest_score["Highest Score"] = highest_score["highest_runs"].astype(int).astype(str) + np.where(highest_score["hs_not_out"], "*", "")
            highest_score = highest_score[["batter_key", "Highest Score"]]
        season_bat = season_bat.merge(highest_score, on="batter_key", how="left")
        season_bat["Highest Score"] = season_bat["Highest Score"].fillna("0")

        extra_type = batter_season_match.get("extra_type", pd.Series("", index=batter_season_match.index)).fillna("").astype(str).str.lower()
        byes_mask = extra_type.str.contains("byes", regex=False) | extra_type.str.contains("legbyes", regex=False)
        legal_deliveries = (batter_season_match["valid_ball"] == 1)
        dot_numerator = legal_deliveries & (batter_season_match["runs_batter"] == 0) & (~byes_mask)
        dot_ball = (
            batter_season_match.assign(
                legal_ball=legal_deliveries.astype(int),
                dot_ball_for_pct=dot_numerator.astype(int),
            )
            .groupby("batter_key", as_index=False)[["legal_ball", "dot_ball_for_pct"]]
            .sum()
        )
        dot_ball["Dot Ball %"] = _safe_divide(dot_ball["dot_ball_for_pct"] * 100, dot_ball["legal_ball"])
        season_bat = season_bat.merge(dot_ball[["batter_key", "Dot Ball %"]], on="batter_key", how="left")
        season_bat["Dot Ball %"] = pd.to_numeric(season_bat["Dot Ball %"], errors="coerce").fillna(0)

        bat_impact = self._impact_by_key[["player_key", "Batting Impact"]].rename(columns={"player_key": "batter_key"})
        season_bat = season_bat.merge(bat_impact, on="batter_key", how="left")
        season_bat["Batting Impact"] = pd.to_numeric(season_bat["Batting Impact"], errors="coerce").fillna(0)
        season_bat["Avg. Batting Impact"] = _safe_divide(season_bat["Batting Impact"], season_bat["dismissals"])
        season_bat = season_bat.merge(
            self._player_meta.rename(columns={"player_key": "batter_key"}),
            on="batter_key",
            how="left",
        )
        season_bat["average"] = _safe_divide(season_bat["runs"], season_bat["dismissals"])
        season_bat["strike_rate"] = _safe_divide(season_bat["runs"] * 100, season_bat["balls_faced"])

        season_bat = season_bat.rename(
            columns={
                "batter_key": "Player Key",
                "batter": "Batter",
                "matches": "Matches",
                "runs": "Runs",
                "balls_faced": "Balls",
                "dismissals": "Dismissals",
                "strike_rate": "Strike Rate",
                "average": "Average",
            }
        )
        season_bat["Team"] = season_bat["Team"].fillna("Unknown")
        season_bat = season_bat[
            [
                "Player Key",
                "Batter",
                "Team",
                "Matches",
                "Runs",
                "Balls",
                "Dismissals",
                "Average",
                "Strike Rate",
                "Highest Score",
                "4",
                "6",
                "Dot Ball %",
                "Batting Impact",
                "Avg. Batting Impact",
            ]
        ]
        season_bat = season_bat.sort_values(["Runs", "Batting Impact"], ascending=[False, False]).reset_index(drop=True)
        return season_bat

    def _build_bowler_stats(self) -> pd.DataFrame:
        season_bowl = self._bowlers_by_season[self._bowlers_by_season["year"] == self.season].copy()
        if season_bowl.empty:
            return pd.DataFrame()

        bowl_impact = self._impact_by_key[["player_key", "Bowling Impact"]].rename(columns={"player_key": "bowler_key"})
        season_bowl = season_bowl.merge(bowl_impact, on="bowler_key", how="left")
        season_bowl["Bowling Impact"] = pd.to_numeric(season_bowl["Bowling Impact"], errors="coerce").fillna(0)
        season_bowl["Avg Bowling Impact / Match"] = _safe_divide(season_bowl["Bowling Impact"], season_bowl["matches"])
        season_bowl["Avg Bowling Impact / Over"] = _safe_divide(season_bowl["Bowling Impact"], season_bowl["overs"])

        season_bowl = season_bowl.merge(
            self._player_meta.rename(columns={"player_key": "bowler_key"}),
            on="bowler_key",
            how="left",
        )
        season_bowl = season_bowl.rename(
            columns={
                "bowler_key": "Player Key",
                "bowler": "Bowler",
                "matches": "Matches",
                "balls": "Balls",
                "overs": "Overs",
                "runs_conceded": "Runs",
                "wickets": "Wickets",
                "economy": "Economy",
                "bowling_average": "Average",
                "maidens": "Maidens",
                "fours_conceded": "4",
                "sixes_conceded": "6",
                "dot_ball_pct": "Dot Ball %",
                "best_bowling": "Best Bowling",
                "Avg Bowling Impact / Match": "Avg. Bowling Impact",
            }
        )
        season_bowl["Team"] = season_bowl["Team"].fillna("Unknown")
        season_bowl = season_bowl[
            [
                "Player Key",
                "Bowler",
                "Team",
                "Matches",
                "Balls",
                "Overs",
                "Wickets",
                "Economy",
                "Average",
                "Best Bowling",
                "Runs",
                "4",
                "6",
                "Maidens",
                "Dot Ball %",
                "Bowling Impact",
                "Avg. Bowling Impact",
            ]
        ]
        season_bowl = season_bowl.sort_values(["Wickets", "Runs"], ascending=[False, True]).reset_index(drop=True)
        return season_bowl

    def _build_player_impact_stats(self) -> pd.DataFrame:
        impact = self._impact_by_key.copy()
        if impact.empty:
            return pd.DataFrame()

        impact = impact.merge(self._player_meta, on=["player_key", "Player"], how="left")
        impact["Team"] = impact["Team"].fillna("Unknown")
        impact["Matches Played"] = pd.to_numeric(impact["Matches Played"], errors="coerce").fillna(0).astype(int)
        impact["Avg Total Impact / Match"] = _safe_divide(impact["Total Impact"], impact["Matches Played"])
        impact = impact.rename(columns={"player_key": "Player Key"})
        impact = impact[
            [
                "Player Key",
                "Player",
                "Team",
                "Matches Played",
                "Batting Impact",
                "Bowling Impact",
                "Total Impact",
                "Avg Total Impact / Match",
            ]
        ]
        impact = impact.sort_values("Total Impact", ascending=False).reset_index(drop=True)
        return impact

    def _build_team_ranking(self) -> pd.DataFrame:
        team_rows = []
        for _, row in self.matches.iterrows():
            team1 = str(row.get("bat_first", "")).strip()
            team2 = str(row.get("bowl_first", "")).strip()
            winner_raw = row.get("match_won_by")
            winner = "" if pd.isna(winner_raw) else str(winner_raw).strip()
            winner_norm = _norm_text(winner)
            result_type = _norm_text(row.get("result_type"))
            status = _norm_text(row.get("status"))

            if not team1 or not team2:
                continue

            if winner_norm in {"", "unknown", "none", "nan", "null"}:
                winner = ""

            if winner:
                team_rows.extend(
                    [
                        {"Team": winner, "GP": 1, "W": 1, "L": 0, "NR": 0, "Points": 2},
                        {"Team": team2 if winner == team1 else team1, "GP": 1, "W": 0, "L": 1, "NR": 0, "Points": 0},
                    ]
                )
            elif result_type in NO_RESULT_TYPES or status in {"abandoned", "no_result"}:
                team_rows.extend(
                    [
                        {"Team": team1, "GP": 1, "W": 0, "L": 0, "NR": 1, "Points": 1},
                        {"Team": team2, "GP": 1, "W": 0, "L": 0, "NR": 1, "Points": 1},
                    ]
                )
            else:
                team_rows.extend(
                    [
                        {"Team": team1, "GP": 1, "W": 0, "L": 0, "NR": 1, "Points": 1},
                        {"Team": team2, "GP": 1, "W": 0, "L": 0, "NR": 1, "Points": 1},
                    ]
                )

        if not team_rows:
            return pd.DataFrame(columns=["Rank", "Team", "Matches", "Wins", "Losses", "No Result", "Points"])

        ranking = pd.DataFrame(team_rows).groupby("Team", as_index=False).sum(numeric_only=True)
        ranking["GP"] = ranking["GP"].astype(int)
        ranking["W"] = ranking["W"].astype(int)
        ranking["L"] = ranking["L"].astype(int)
        ranking["NR"] = ranking["NR"].astype(int)
        ranking["Points"] = ranking["Points"].astype(int)
        ranking = ranking.sort_values(["Points", "W", "Team"], ascending=[False, False, True]).reset_index(drop=True)
        # Competition ranking on points: ties share rank and next rank skips accordingly.
        ranking["Rank"] = ranking["Points"].rank(method="min", ascending=False).astype(int)
        ranking = ranking.rename(
            columns={
                "GP": "Matches",
                "W": "Wins",
                "L": "Losses",
                "NR": "No Result",
            }
        )
        ranking = ranking[["Rank", "Team", "Matches", "Wins", "Losses", "No Result", "Points"]]
        return ranking

    def _styled_empty_figure(self, message: str):
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=PLOTLY_BASE_FONT_SIZE, family=PLOTLY_FONT_FAMILY),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=320, margin={"l": 20, "r": 20, "t": 40, "b": 20})
        apply_plot_theme(fig)
        self._apply_graph_typography(fig)
        return fig

    def _apply_graph_typography(self, fig):
        # Match match.py behavior: let Plotly/template handle title/axis-title sizing.
        fig.update_layout(
            legend_font=dict(size=PLOTLY_LEGEND_FONT_SIZE, family=PLOTLY_FONT_FAMILY),
        )
        fig.update_xaxes(tickfont=dict(size=PLOTLY_AXIS_TICK_FONT_SIZE, family=PLOTLY_FONT_FAMILY))
        fig.update_yaxes(tickfont=dict(size=PLOTLY_BASE_FONT_SIZE, family=PLOTLY_FONT_FAMILY))

    def _apply_horizontal_bar_axes(self, fig, player_order: list[str]):
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=player_order,
            autorange="reversed",
            fixedrange=True,
            domain=[0, PLOTLY_HEADER_PLOT_TOP],
            tickfont=dict(size=PLOTLY_BASE_FONT_SIZE, family=PLOTLY_FONT_FAMILY),
            ticklabelstandoff=6,
            automargin=True,
            ticks="",
            ticklen=0,
        )

    def _format_bar_labels(self, values: pd.Series) -> list[str]:
        labels = []
        numeric = pd.to_numeric(values, errors="coerce")
        for value in numeric:
            if pd.isna(value):
                labels.append("")
            elif abs(float(value) - round(float(value))) < 1e-9:
                labels.append(str(int(round(float(value)))))
            else:
                labels.append(f"{float(value):.2f}")
        return labels

    def _player_bar_chart(
        self,
        df: pd.DataFrame,
        *,
        player_col: str,
        value_col: str,
        title: str,
        xaxis_title: str,
        n: int = 5,
        ascending: bool = False,
        color: str | None = None,
    ):
        if df.empty:
            return self._styled_empty_figure("No data available")

        X = df.copy()
        X = X.dropna(subset=[player_col, value_col])
        if X.empty:
            return self._styled_empty_figure("No data available")

        n = max(1, int(n))
        X[value_col] = pd.to_numeric(X[value_col], errors="coerce")
        X = X.dropna(subset=[value_col])
        X = X.sort_values(value_col, ascending=ascending).head(n)

        customdata = np.column_stack(
            [
                X[player_col].astype(str),
                X.get("Team", X.get("Teams", pd.Series("", index=X.index))).fillna("").astype(str),
                pd.to_numeric(X.get("Matches", X.get("Matches Played", pd.Series(0, index=X.index))), errors="coerce").fillna(0).astype(int).astype(str),
            ]
        )
        fig = go.Figure(
            data=[
                go.Bar(
                    x=X[value_col],
                    y=X[player_col],
                    orientation="h",
                    marker=dict(color=color or PLOTLY_COLORS["leaderboard_primary"]),
                    text=self._format_bar_labels(X[value_col]),
                    textposition="outside",
                    textfont=dict(
                        size=PLOTLY_BASE_FONT_SIZE,
                        color="#1F2A37",
                        family=PLOTLY_FONT_FAMILY,
                    ),
                    cliponaxis=False,
                    customdata=customdata,
                    hovertemplate=(
                        "Player = %{customdata[0]}<br>"
                        "Team = %{customdata[1]}<br>"
                        "Matches = %{customdata[2]}<br>"
                        f"{xaxis_title} = %{{x:.2f}}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            ]
        )
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", y=PLOTLY_HEADER_TITLE_Y, yanchor="top"),
            xaxis_title=f"<b>{xaxis_title}</b>",
            yaxis_title=None,
            height=320,
            margin={"t": PLOTLY_HEADER_MARGIN_TOP, "l": PLOTLY_HEADER_MARGIN_LEFT, "r": PLOTLY_HEADER_MARGIN_RIGHT, "b": 40},
        )
        apply_plot_theme(fig)
        fig.update_traces(cliponaxis=False)
        xaxis_kwargs = dict(
            fixedrange=True,
            showgrid=False,
            tickfont=dict(size=PLOTLY_AXIS_TICK_FONT_SIZE, family=PLOTLY_FONT_FAMILY),
        )
        values = pd.to_numeric(X[value_col], errors="coerce").dropna()
        if not values.empty:
            x_min = min(0.0, float(values.min()))
            x_max = max(0.0, float(values.max()))
            span = x_max - x_min
            if span <= 0:
                span = max(abs(x_min), abs(x_max), 1.0)

            left_pad = span * (0.24 if x_min < 0 else 0.06)
            right_pad = span * (0.18 if x_max > 0 else 0.06)
            xaxis_kwargs["range"] = [x_min - left_pad, x_max + right_pad]

        fig.update_xaxes(**xaxis_kwargs)
        self._apply_horizontal_bar_axes(fig, X[player_col].astype(str).tolist())
        self._apply_graph_typography(fig)
        return fig

    def plot_total_runs(self, n: int = 5):
        return self._player_bar_chart(
            self.batter_stats,
            player_col="Batter",
            value_col="Runs",
            title="Total Runs",
            xaxis_title="Runs",
            n=n,
            ascending=False,
            color=PLOTLY_COLORS["innings_2"],
        )

    def plot_total_wickets(self, n: int = 5):
        return self._player_bar_chart(
            self.bowler_stats,
            player_col="Bowler",
            value_col="Wickets",
            title="Total Wickets",
            xaxis_title="Wickets",
            n=n,
            ascending=False,
            color=PLOTLY_PALETTE[0],
        )

    def plot_total_impact(self, n: int = 5):
        return self._player_bar_chart(
            self.player_impact_stats,
            player_col="Player",
            value_col="Total Impact",
            title="Total Impact",
            xaxis_title="Total Impact",
            n=n,
            ascending=False,
            color=PLOTLY_COLORS["leaderboard_impact"],
        )

    def plot_strike_rate(self, n: int = 5, min_runs: int = 100):
        X = self.batter_stats[self.batter_stats["Runs"] >= int(min_runs)].copy()
        return self._player_bar_chart(
            X,
            player_col="Batter",
            value_col="Strike Rate",
            title=f"Strike Rate (Min {int(min_runs)} Runs)",
            xaxis_title="Strike Rate",
            n=n,
            ascending=False,
            color=PLOTLY_COLORS["innings_2"],
        )

    def plot_batting_average(self, n: int = 5, min_runs: int = 100):
        X = self.batter_stats[self.batter_stats["Runs"] >= int(min_runs)].copy()
        return self._player_bar_chart(
            X,
            player_col="Batter",
            value_col="Average",
            title=f"Batting Average (Min {int(min_runs)} Runs)",
            xaxis_title="Batting Average",
            n=n,
            ascending=False,
            color=PLOTLY_COLORS["innings_2"],
        )

    def plot_total_boundaries_stacked(self, n: int = 5):
        if self.batter_stats.empty:
            return self._styled_empty_figure("No data available")

        X = self.batter_stats.copy()
        X["Total Boundaries"] = pd.to_numeric(X["4"], errors="coerce").fillna(0) + pd.to_numeric(X["6"], errors="coerce").fillna(0)
        X = X.sort_values("Total Boundaries", ascending=False).head(max(1, int(n))).copy()
        customdata = np.column_stack(
            [
                X["Batter"].astype(str),
                X["Team"].fillna("").astype(str),
                pd.to_numeric(X["Matches"], errors="coerce").fillna(0).astype(int).astype(str),
            ]
        )

        fig = go.Figure()
        four_text = X["4"].apply(lambda v: str(int(v)) if pd.notna(v) and int(v) > 0 else "")
        six_text = X["6"].apply(lambda v: str(int(v)) if pd.notna(v) and int(v) > 0 else "")
        fig.add_trace(
            go.Bar(
                x=X["4"],
                y=X["Batter"],
                name="4",
                orientation="h",
                marker=dict(color=PLOTLY_COLORS["success_card"]),
                text=four_text,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    size=PLOTLY_BASE_FONT_SIZE,
                    color="#1F2A37",
                    family=PLOTLY_FONT_FAMILY,
                ),
                customdata=customdata,
                hovertemplate=(
                    "Player = %{customdata[0]}<br>"
                    "Team = %{customdata[1]}<br>"
                    "Matches = %{customdata[2]}<br>"
                    "4 = %{x}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                x=X["6"],
                y=X["Batter"],
                name="6",
                orientation="h",
                marker=dict(color=PLOTLY_COLORS["leaderboard_impact"]),
                text=six_text,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    size=PLOTLY_BASE_FONT_SIZE,
                    color="#1F2A37",
                    family=PLOTLY_FONT_FAMILY,
                ),
                customdata=customdata,
                hovertemplate=(
                    "Player = %{customdata[0]}<br>"
                    "Team = %{customdata[1]}<br>"
                    "Matches = %{customdata[2]}<br>"
                    "6 = %{x}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            barmode="stack",
            title=dict(text="<b>Total Boundaries (4s + 6s)</b>", x=0.5, xanchor="center", y=PLOTLY_HEADER_TITLE_Y, yanchor="top"),
            xaxis_title="<b>Boundaries</b>",
            yaxis_title=None,
            height=320,
            margin={"t": PLOTLY_HEADER_MARGIN_TOP, "l": PLOTLY_HEADER_MARGIN_LEFT, "r": PLOTLY_HEADER_MARGIN_RIGHT, "b": 40},
        )
        apply_plot_theme(fig)
        fig.update_traces(cliponaxis=False)
        total_boundaries = pd.to_numeric(X["4"], errors="coerce").fillna(0) + pd.to_numeric(X["6"], errors="coerce").fillna(0)
        max_total = float(total_boundaries.max()) if not total_boundaries.empty else 0.0
        span = max(max_total, 1.0)
        fig.update_xaxes(
            range=[0, max_total + span * 0.18],
            fixedrange=True,
            showgrid=False,
            tickfont=dict(size=PLOTLY_AXIS_TICK_FONT_SIZE, family=PLOTLY_FONT_FAMILY),
        )
        self._apply_horizontal_bar_axes(fig, X["Batter"].astype(str).tolist())
        self._apply_graph_typography(fig)
        return fig

    def plot_bowling_average(self, n: int = 5, min_overs: float = 8, ascending: bool = True):
        X = self.bowler_stats[self.bowler_stats["Overs"] >= float(min_overs)].copy()
        return self._player_bar_chart(
            X,
            player_col="Bowler",
            value_col="Average",
            title=f"Bowling Average (Min {float(min_overs):g} Overs)",
            xaxis_title="Bowling Average",
            n=n,
            ascending=ascending,
        )

    def plot_economy(self, n: int = 5, min_overs: float = 8, ascending: bool = True):
        X = self.bowler_stats[self.bowler_stats["Overs"] >= float(min_overs)].copy()
        return self._player_bar_chart(
            X,
            player_col="Bowler",
            value_col="Economy",
            title=f"Economy Rate (Min {float(min_overs):g} Overs)",
            xaxis_title="Economy",
            n=n,
            ascending=ascending,
        )

    def plot_dot_ball_pct(self, n: int = 5, min_overs: float = 8, descending: bool = True):
        X = self.bowler_stats[self.bowler_stats["Overs"] >= float(min_overs)].copy()
        return self._player_bar_chart(
            X,
            player_col="Bowler",
            value_col="Dot Ball %",
            title=f"Dot Ball % (Min {float(min_overs):g} Overs)",
            xaxis_title="Dot Ball %",
            n=n,
            ascending=not descending,
        )

    def plot_maidens(self, n: int = 5):
        return self._player_bar_chart(
            self.bowler_stats,
            player_col="Bowler",
            value_col="Maidens",
            title="Maiden Overs",
            xaxis_title="Maidens",
            n=n,
            ascending=False,
        )

    def plot_avg_total_impact_per_game(self, n: int = 5, min_matches: int = 2):
        X = self.player_impact_stats[self.player_impact_stats["Matches Played"] >= int(min_matches)].copy()
        return self._player_bar_chart(
            X,
            player_col="Player",
            value_col="Avg Total Impact / Match",
            title=f"Average Total Impact Per Match (Min {int(min_matches)} Matches)",
            xaxis_title="Avg Total Impact / Match",
            n=n,
            ascending=False,
        )

    def plot_clutch_impact(self, n: int = 5):
        return self._styled_empty_figure("Clutch Impact placeholder (to be implemented)")
