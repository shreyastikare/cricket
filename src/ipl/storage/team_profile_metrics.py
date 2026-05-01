"""Derived team profile metric cache refresh helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH
from .schema import create_schema
from .sqlite import transaction, upsert_dataframe
from .utils import utc_now_iso


TEAM_PROFILE_METRICS_ARTIFACT = "team_profile_metrics"
TEAM_PROFILE_METRICS_CALCULATION_VERSION = 2
FINAL_MATCH_STATUSES = ("complete", "abandoned", "no_result")

TEAM_PROFILE_METRIC_COLUMNS = [
    "season",
    "team",
    "metric_key",
    "metric_name",
    "profile",
    "category",
    "display_order",
    "description",
    "raw_value",
    "created_at",
    "updated_at",
]


def _final_status_placeholders() -> str:
    return ",".join(["?"] * len(FINAL_MATCH_STATUSES))


def _source_state(conn, season: int) -> dict[str, Any]:
    placeholders = _final_status_placeholders()
    params: list[Any] = [int(season), *FINAL_MATCH_STATUSES]
    matches = conn.execute(
        f"""
        SELECT CAST(match_id AS INTEGER), updated_at
        FROM match_list
        WHERE CAST(year AS INTEGER) = ?
          AND LOWER(TRIM(COALESCE(status, ''))) IN ({placeholders})
        """,
        params,
    ).fetchall()
    match_ids = [int(row[0]) for row in matches if row[0] is not None]
    match_updated_values = [row[1] for row in matches if row[1] is not None]

    source_values = list(match_updated_values)
    if match_ids:
        ball_placeholders = ",".join(["?"] * len(match_ids))
        ball_row = conn.execute(
            f"""
            SELECT MAX(updated_at)
            FROM ball_by_ball
            WHERE CAST(match_id AS INTEGER) IN ({ball_placeholders})
            """,
            match_ids,
        ).fetchone()
        if ball_row and ball_row[0] is not None:
            source_values.append(ball_row[0])

    prediction_updated_at = None
    prediction_model_version = None
    prediction_count = 0
    if match_ids:
        ball_placeholders = ",".join(["?"] * len(match_ids))
        prediction_row = conn.execute(
            f"""
            SELECT
                MAX(COALESCE(updated_at, created_at)),
                MAX(model_version),
                COUNT(*)
            FROM prediction_history
            WHERE model_type = 'win_probability'
              AND CAST(match_id AS INTEGER) IN ({ball_placeholders})
            """,
            match_ids,
        ).fetchone()
        if prediction_row:
            prediction_updated_at = prediction_row[0]
            prediction_model_version = prediction_row[1]
            prediction_count = int(prediction_row[2] or 0)
            if prediction_updated_at is not None:
                source_values.append(prediction_updated_at)

    source_max_updated_at = max(source_values) if source_values else None
    source_fingerprint = (
        f"{source_max_updated_at or ''}"
        f"|prediction:{prediction_updated_at or ''}:{prediction_model_version or ''}:{prediction_count}"
    )

    return {
        "finalized_match_count": len(match_ids),
        "source_max_updated_at": source_fingerprint,
        "match_ids": match_ids,
    }


def _load_refresh_state(conn, season: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            artifact,
            season,
            calculation_version,
            finalized_match_count,
            source_max_updated_at,
            last_success_at,
            last_attempt_at,
            status,
            rows_written,
            error_message
        FROM derived_refresh_state
        WHERE artifact = ? AND season = ?
        """,
        (TEAM_PROFILE_METRICS_ARTIFACT, int(season)),
    ).fetchone()
    if row is None:
        return None
    keys = [
        "artifact",
        "season",
        "calculation_version",
        "finalized_match_count",
        "source_max_updated_at",
        "last_success_at",
        "last_attempt_at",
        "status",
        "rows_written",
        "error_message",
    ]
    return dict(zip(keys, row))


def _state_matches_success(existing: dict[str, Any] | None, source: dict[str, Any]) -> bool:
    if not existing:
        return False
    return (
        str(existing.get("status") or "").lower() == "success"
        and int(existing.get("calculation_version") or 0) == TEAM_PROFILE_METRICS_CALCULATION_VERSION
        and int(existing.get("finalized_match_count") or 0) == int(source.get("finalized_match_count") or 0)
        and (existing.get("source_max_updated_at") or None) == (source.get("source_max_updated_at") or None)
    )


def _upsert_refresh_state(
    conn,
    *,
    season: int,
    source: dict[str, Any],
    now_ts: str,
    status: str,
    rows_written: int,
    error_message: str | None,
    last_success_at: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO derived_refresh_state (
            artifact,
            season,
            calculation_version,
            finalized_match_count,
            source_max_updated_at,
            last_success_at,
            last_attempt_at,
            status,
            rows_written,
            error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (artifact, season) DO UPDATE SET
            calculation_version = excluded.calculation_version,
            finalized_match_count = excluded.finalized_match_count,
            source_max_updated_at = excluded.source_max_updated_at,
            last_success_at = excluded.last_success_at,
            last_attempt_at = excluded.last_attempt_at,
            status = excluded.status,
            rows_written = excluded.rows_written,
            error_message = excluded.error_message;
        """,
        (
            TEAM_PROFILE_METRICS_ARTIFACT,
            int(season),
            TEAM_PROFILE_METRICS_CALCULATION_VERSION,
            int(source.get("finalized_match_count") or 0),
            source.get("source_max_updated_at"),
            last_success_at,
            now_ts,
            status,
            int(rows_written),
            error_message,
        ),
    )


def _team_options_from_leaderboard(leaderboard) -> list[str]:
    matches = getattr(leaderboard, "matches", pd.DataFrame())
    teams: list[str] = []
    if isinstance(matches, pd.DataFrame) and not matches.empty:
        for col in ["bat_first", "bowl_first"]:
            if col in matches.columns:
                teams.extend(matches[col].dropna().astype(str).str.strip().tolist())
    return sorted({team for team in teams if team})


def _compute_profile_rows(season: int, db_path: Path | str, now_ts: str) -> pd.DataFrame:
    try:
        from app.leaderboard import Leaderboard
        from app.team_analysis import compute_team_profile_metrics
    except (ImportError, ModuleNotFoundError):
        from leaderboard import Leaderboard
        from team_analysis import compute_team_profile_metrics

    leaderboard = Leaderboard(season=int(season), db_path=db_path)
    matches = getattr(leaderboard, "matches", pd.DataFrame())
    impact_balls = getattr(leaderboard, "_impact_balls", pd.DataFrame())
    rows = []
    for team in _team_options_from_leaderboard(leaderboard):
        metrics = compute_team_profile_metrics(impact_balls, matches, str(team))
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["season"] = int(season)
        metrics["team"] = str(team)
        metrics["created_at"] = now_ts
        metrics["updated_at"] = now_ts
        rows.append(metrics)
    if not rows:
        return pd.DataFrame(columns=TEAM_PROFILE_METRIC_COLUMNS)
    out = pd.concat(rows, ignore_index=True)
    out["display_order"] = pd.to_numeric(out.get("display_order"), errors="coerce").fillna(9999).astype(int)
    out["raw_value"] = pd.to_numeric(out.get("raw_value"), errors="coerce").fillna(0.0).astype(float)
    for col in TEAM_PROFILE_METRIC_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[TEAM_PROFILE_METRIC_COLUMNS]


def refresh_team_profile_metrics_for_season(
    *,
    season: int,
    db_path: Path | str = DB_PATH,
    force: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    create_schema(db_path)
    season = int(season)
    now_ts = utc_now_iso()
    try:
        with transaction(db_path) as conn:
            source = _source_state(conn, season)
            existing = _load_refresh_state(conn, season)
            if (not force) and _state_matches_success(existing, source):
                _upsert_refresh_state(
                    conn,
                    season=season,
                    source=source,
                    now_ts=now_ts,
                    status="skipped",
                    rows_written=int(existing.get("rows_written") or 0),
                    error_message=None,
                    last_success_at=existing.get("last_success_at"),
                )
                return {
                    "artifact": TEAM_PROFILE_METRICS_ARTIFACT,
                    "season": season,
                    "status": "skipped",
                    "skipped": True,
                    "rows_written": int(existing.get("rows_written") or 0),
                    "finalized_match_count": int(source.get("finalized_match_count") or 0),
                    "source_max_updated_at": source.get("source_max_updated_at"),
                }

        if int(source.get("finalized_match_count") or 0) == 0:
            metrics = pd.DataFrame(columns=TEAM_PROFILE_METRIC_COLUMNS)
        else:
            metrics = _compute_profile_rows(season, db_path, now_ts)

        with transaction(db_path) as conn:
            conn.execute("DELETE FROM team_profile_metrics WHERE season = ?;", (season,))
            rows_written = upsert_dataframe(
                conn=conn,
                table_name="team_profile_metrics",
                df=metrics,
                conflict_columns=["season", "team", "metric_key"],
                update_columns=[col for col in TEAM_PROFILE_METRIC_COLUMNS if col not in {"season", "team", "metric_key", "created_at"}],
            )
            _upsert_refresh_state(
                conn,
                season=season,
                source=source,
                now_ts=now_ts,
                status="success",
                rows_written=int(rows_written),
                error_message=None,
                last_success_at=now_ts,
            )

        return {
            "artifact": TEAM_PROFILE_METRICS_ARTIFACT,
            "season": season,
            "status": "success",
            "skipped": False,
            "rows_written": int(rows_written),
            "finalized_match_count": int(source.get("finalized_match_count") or 0),
            "source_max_updated_at": source.get("source_max_updated_at"),
        }
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        try:
            with transaction(db_path) as conn:
                source = _source_state(conn, season)
                existing = _load_refresh_state(conn, season)
                _upsert_refresh_state(
                    conn,
                    season=season,
                    source=source,
                    now_ts=utc_now_iso(),
                    status="failed",
                    rows_written=int(existing.get("rows_written") or 0) if existing else 0,
                    error_message=error_message,
                    last_success_at=existing.get("last_success_at") if existing else None,
                )
        except Exception:
            pass
        if raise_on_error:
            raise
        return {
            "artifact": TEAM_PROFILE_METRICS_ARTIFACT,
            "season": season,
            "status": "failed",
            "skipped": False,
            "rows_written": 0,
            "error_message": error_message,
        }


def load_team_profile_metrics(
    *,
    db_path: Path | str = DB_PATH,
    seasons: list[int] | None = None,
    season: int | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    create_schema(db_path)
    clauses = []
    params: list[Any] = []
    if season is not None:
        clauses.append("season = ?")
        params.append(int(season))
    elif seasons is not None:
        season_values = [int(value) for value in seasons]
        if not season_values:
            return pd.DataFrame(columns=TEAM_PROFILE_METRIC_COLUMNS)
        clauses.append(f"season IN ({','.join(['?'] * len(season_values))})")
        params.extend(season_values)
    if team is not None:
        clauses.append("team = ?")
        params.append(str(team))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT {', '.join(TEAM_PROFILE_METRIC_COLUMNS)}
        FROM team_profile_metrics
        {where}
        ORDER BY season, team, display_order
    """
    with transaction(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)
