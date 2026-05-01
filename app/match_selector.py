from __future__ import annotations

import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv('DB_PATH', 'data/sqlite/ipl.db')


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def get_year_options() -> list[dict[str, int]]:
    query = """
        SELECT DISTINCT year
        FROM match_list
        WHERE year IS NOT NULL
        ORDER BY year DESC
    """
    with _connect() as conn:
        rows = conn.execute(query).fetchall()

    return [{"label": str(row[0]), "value": int(row[0])} for row in rows]


def get_team_options() -> list[dict[str, str]]:
    return get_team_options_for_year(year=None)


def get_team_options_for_year(year: int | None = None) -> list[dict[str, str]]:
    if year is None:
        query = """
            SELECT team
            FROM (
                SELECT bat_first AS team FROM match_list
                UNION
                SELECT bowl_first AS team FROM match_list
            )
            WHERE team IS NOT NULL AND TRIM(team) != ''
            ORDER BY team
        """
        params: list[object] = []
    else:
        query = """
            SELECT team
            FROM (
                SELECT bat_first AS team FROM match_list WHERE year = ?
                UNION
                SELECT bowl_first AS team FROM match_list WHERE year = ?
            )
            WHERE team IS NOT NULL AND TRIM(team) != ''
            ORDER BY team
        """
        params = [year, year]

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [{"label": row[0], "value": row[0]} for row in rows]


def _format_date(iso_date: str) -> str:
    try:
        return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%m/%d/%Y")
    except ValueError:
        return iso_date


def _format_local_date_from_utc(timestamp_text: str | None) -> str | None:
    if timestamp_text is None:
        return None
    text = str(timestamp_text).strip()
    if not text:
        return None
    try:
        dt_utc = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            return None
        return dt_utc.astimezone().strftime("%m/%d/%Y")
    except ValueError:
        return None


def _match_descriptor(event_match_no: str | None, stage: str | None, playoff_match: int | None) -> str:
    match_no = "" if event_match_no is None else str(event_match_no).strip()
    stage_clean = "" if stage is None else str(stage).strip()
    is_playoff = False
    if playoff_match is not None:
        try:
            is_playoff = int(float(str(playoff_match).strip())) == 1
        except (TypeError, ValueError):
            is_playoff = False

    if is_playoff and stage_clean and stage_clean.lower() != "unknown":
        return stage_clean
    if match_no and match_no.lower() != "unknown":
        return f"Match {match_no}"
    return "Match"


def get_match_options(
    year: int | None = None,
    team: str | None = None,
    match_type: str | None = None,
) -> list[dict[str, int | str]]:
    query = """
        SELECT
            ml.match_id,
            COALESCE(ms.match_date, ml.date) AS match_date,
            ms.scheduled_start_ts,
            ml.event_match_no,
            ml.bat_first,
            ml.bowl_first,
            ml.stage,
            ml.playoff_match
        FROM match_list ml
        LEFT JOIN match_schedule ms
          ON CAST(ms.match_id AS INTEGER) = ml.match_id
        WHERE 1 = 1
    """
    params: list[object] = []

    if year is not None:
        query += " AND ml.year = ?"
        params.append(year)

    if team:
        query += " AND (ml.bat_first = ? OR ml.bowl_first = ?)"
        params.extend([team, team])

    if match_type == "regular":
        query += " AND ml.playoff_match = 0"
    elif match_type == "knockout":
        query += " AND ml.playoff_match = 1"

    query += " ORDER BY ml.year DESC, COALESCE(ms.match_date, ml.date) DESC, ml.match_id DESC"

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    options: list[dict[str, int | str]] = []
    for match_id, match_date, scheduled_start_ts, event_match_no, batting_team, bowling_team, stage, playoff_match in rows:
        descriptor = _match_descriptor(event_match_no, stage, playoff_match)
        date_text = _format_local_date_from_utc(scheduled_start_ts) or _format_date(match_date)
        label = f"{date_text} {descriptor}: {batting_team} vs. {bowling_team}"
        options.append({"label": label, "value": int(match_id)})

    return options
