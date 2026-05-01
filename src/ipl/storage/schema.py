"""SQLite schema creation and validation for Phase 1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import (
    BALL_BY_BALL_BASE_COLUMNS,
    BALL_BY_BALL_HELPER_COLUMNS,
    DB_PATH,
    MATCH_LIST_BASE_COLUMNS,
    MATCH_LIST_HELPER_COLUMNS,
)
from .sqlite import quote_ident, transaction
from .utils import ensure_phase1_dirs


def _column_definitions(columns: list[str]) -> str:
    return ",\n    ".join(quote_ident(col) for col in columns)


def _create_ball_by_ball_table(conn: sqlite3.Connection) -> None:
    base_defs = _column_definitions(BALL_BY_BALL_BASE_COLUMNS)
    sql = f"""
    CREATE TABLE IF NOT EXISTS ball_by_ball (
        {base_defs},
        "ball_id" TEXT NOT NULL UNIQUE,
        "source" TEXT,
        "ingest_run_id" TEXT,
        "created_at" TEXT NOT NULL,
        "updated_at" TEXT NOT NULL
    );
    """
    conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ball_by_ball_match_id ON ball_by_ball(match_id);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ball_by_ball_match_id_innings ON ball_by_ball(match_id, innings);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ball_by_ball_ball_id ON ball_by_ball(ball_id);")


def _migrate_ball_by_ball_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(ball_by_ball);").fetchall()
    if not rows:
        return

    actual = [row[1] for row in rows]
    for col in BALL_BY_BALL_BASE_COLUMNS:
        if col not in actual:
            conn.execute(f'ALTER TABLE ball_by_ball ADD COLUMN {quote_ident(col)};')


def _create_match_list_table(conn: sqlite3.Connection) -> None:
    base_defs = _column_definitions(MATCH_LIST_BASE_COLUMNS[1:])
    sql = f"""
    CREATE TABLE IF NOT EXISTS match_list (
        "match_id" PRIMARY KEY,
        {base_defs},
        "source" TEXT,
        "status" TEXT,
        "status_detail" TEXT,
        "ingestion_status" TEXT,
        "completeness_status" TEXT,
        "data_quality_flag" TEXT,
        "last_successful_fetch_ts" TEXT,
        "last_consistency_check_ts" TEXT,
        "created_at" TEXT NOT NULL,
        "updated_at" TEXT NOT NULL
    );
    """
    conn.execute(sql)


def _migrate_match_list_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(match_list);").fetchall()
    if not rows:
        return

    actual = [row[1] for row in rows]

    rename_pairs = [
        ("batting_team", "bat_first"),
        ("bowling_team", "bowl_first"),
    ]
    for old_col, new_col in rename_pairs:
        if old_col in actual and new_col not in actual:
            conn.execute(
                f"ALTER TABLE match_list RENAME COLUMN {quote_ident(old_col)} TO {quote_ident(new_col)};"
            )
            actual = [new_col if col == old_col else col for col in actual]

    for col in MATCH_LIST_BASE_COLUMNS[1:]:
        if col not in actual:
            conn.execute(f'ALTER TABLE match_list ADD COLUMN {quote_ident(col)};')


def _create_match_schedule_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_schedule (
            match_id TEXT PRIMARY KEY,
            season INTEGER NULL,
            competition TEXT NULL,
            match_date TEXT NULL,
            scheduled_start_ts TEXT NULL,
            actual_start_ts TEXT NULL,
            status TEXT NULL,
            status_detail TEXT NULL,
            team1 TEXT NULL,
            team2 TEXT NULL,
            venue TEXT NULL,
            city TEXT NULL,
            event_match_no TEXT NULL,
            stage TEXT NULL,
            source TEXT NULL,
            ingestion_status TEXT NULL,
            last_successful_fetch_ts TEXT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_match_schedule_start ON match_schedule(scheduled_start_ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_match_schedule_status ON match_schedule(status);")


def _migrate_match_schedule_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(match_schedule);").fetchall()
    if not rows:
        return

    actual = [row[1] for row in rows]
    for col in ["event_match_no", "stage"]:
        if col not in actual:
            conn.execute(f"ALTER TABLE match_schedule ADD COLUMN {quote_ident(col)};")


def _create_match_poll_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_poll_state (
            match_id TEXT PRIMARY KEY,
            phase TEXT NOT NULL,
            last_polled_at TEXT NULL,
            next_poll_at TEXT NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            cooldown_until TEXT NULL,
            last_seen_status TEXT NULL,
            last_ingested_ball_id TEXT NULL,
            closed_at TEXT NULL,
            terminal_first_seen_at TEXT NULL,
            terminal_confirm_count INTEGER NOT NULL DEFAULT 0,
            terminal_candidate_hash TEXT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_match_poll_state_phase ON match_poll_state(phase);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_match_poll_state_next_poll ON match_poll_state(next_poll_at);")


def _migrate_match_poll_state_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(match_poll_state);").fetchall()
    if not rows:
        return

    actual = [row[1] for row in rows]
    add_columns = {
        "phase": "TEXT NOT NULL DEFAULT 'idle'",
        "last_polled_at": "TEXT NULL",
        "next_poll_at": "TEXT NULL",
        "failure_count": "INTEGER NOT NULL DEFAULT 0",
        "cooldown_until": "TEXT NULL",
        "last_seen_status": "TEXT NULL",
        "last_ingested_ball_id": "TEXT NULL",
        "closed_at": "TEXT NULL",
        "terminal_first_seen_at": "TEXT NULL",
        "terminal_confirm_count": "INTEGER NOT NULL DEFAULT 0",
        "terminal_candidate_hash": "TEXT NULL",
        "created_at": "TEXT NULL",
        "updated_at": "TEXT NULL",
    }
    for col, definition in add_columns.items():
        if col not in actual:
            conn.execute(f"ALTER TABLE match_poll_state ADD COLUMN {quote_ident(col)} {definition};")


def _create_prediction_history_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            prediction_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            ball_id TEXT NULL,
            model_type TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction_ts TEXT NOT NULL,
            batting_team_win_prob REAL NULL,
            bowling_team_win_prob REAL NULL,
            prob_bowling_diff REAL NULL,
            prob_batting_diff REAL NULL,
            metadata_json TEXT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_history_match_id ON prediction_history(match_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_history_ball_id ON prediction_history(ball_id);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_model_match "
        "ON prediction_history(model_type, model_version, match_id);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_model_match_int "
        "ON prediction_history(model_type, model_version, CAST(match_id AS INTEGER));"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_model_match_int_lookup "
        "ON prediction_history(model_type, CAST(match_id AS INTEGER));"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_match_int "
        "ON prediction_history(CAST(match_id AS INTEGER));"
    )


def _migrate_prediction_history_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(prediction_history);").fetchall()
    if not rows:
        return
    actual = [row[1] for row in rows]
    add_columns = {
        "prob_bowling_diff": "REAL NULL",
        "prob_batting_diff": "REAL NULL",
        "updated_at": "TEXT NULL",
    }
    for col, definition in add_columns.items():
        if col not in actual:
            conn.execute(f"ALTER TABLE prediction_history ADD COLUMN {quote_ident(col)} {definition};")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_history_ball_id ON prediction_history(ball_id);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_model_match "
        "ON prediction_history(model_type, model_version, match_id);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_model_match_int "
        "ON prediction_history(model_type, model_version, CAST(match_id AS INTEGER));"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_model_match_int_lookup "
        "ON prediction_history(model_type, CAST(match_id AS INTEGER));"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_match_int "
        "ON prediction_history(CAST(match_id AS INTEGER));"
    )


def _create_api_fetch_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_fetch_log (
            fetch_id TEXT PRIMARY KEY,
            match_id TEXT NULL,
            endpoint_type TEXT NOT NULL,
            request_url TEXT NULL,
            request_ts TEXT NOT NULL,
            response_ts TEXT NULL,
            http_status INTEGER NULL,
            success_flag INTEGER NOT NULL,
            empty_response_flag INTEGER DEFAULT 0,
            parse_success_flag INTEGER DEFAULT 0,
            rows_written INTEGER DEFAULT 0,
            retry_number INTEGER DEFAULT 0,
            error_type TEXT NULL,
            error_message TEXT NULL,
            payload_path TEXT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_fetch_log_request_ts ON api_fetch_log(request_ts);")


def _create_raw_api_responses_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_api_responses (
            response_id TEXT PRIMARY KEY,
            fetch_id TEXT NOT NULL,
            match_id TEXT NULL,
            endpoint_type TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(fetch_id) REFERENCES api_fetch_log(fetch_id)
        );
        """
    )


def _create_player_master_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_master (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            espn_athlete_id TEXT UNIQUE NULL,
            canonical_name TEXT NOT NULL,
            canonical_source TEXT NULL,
            active_flag INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_master_espn_athlete_id "
        "ON player_master(espn_athlete_id) WHERE espn_athlete_id IS NOT NULL;"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_master_canonical_name ON player_master(canonical_name);")


def _migrate_player_master_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(player_master);").fetchall()
    if not rows:
        return
    actual = [row[1] for row in rows]
    expected_defaults = {
        "espn_athlete_id": None,
        "canonical_name": None,
        "canonical_source": None,
        "active_flag": "1",
        "created_at": None,
        "updated_at": None,
    }
    for col, default_val in expected_defaults.items():
        if col in actual:
            continue
        if default_val is None:
            conn.execute(f"ALTER TABLE player_master ADD COLUMN {quote_ident(col)};")
        else:
            conn.execute(
                f"ALTER TABLE player_master ADD COLUMN {quote_ident(col)} "
                f"INTEGER NOT NULL DEFAULT {default_val};"
            )


def _create_player_alias_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_alias (
            alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias_name TEXT NOT NULL,
            alias_source TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            verified INTEGER NOT NULL DEFAULT 0,
            first_seen_match_id TEXT NULL,
            last_seen_match_id TEXT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(alias_name, alias_source),
            FOREIGN KEY(player_id) REFERENCES player_master(player_id)
        );
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_alias_name_source "
        "ON player_alias(alias_name, alias_source);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_alias_player_id ON player_alias(player_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_alias_alias_name ON player_alias(alias_name);")


def _migrate_player_alias_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(player_alias);").fetchall()
    if not rows:
        return
    actual = [row[1] for row in rows]
    add_columns = {
        "verified": "INTEGER NOT NULL DEFAULT 0",
        "first_seen_match_id": "TEXT NULL",
        "last_seen_match_id": "TEXT NULL",
        "created_at": "TEXT NULL",
        "updated_at": "TEXT NULL",
    }
    for col, definition in add_columns.items():
        if col not in actual:
            conn.execute(f"ALTER TABLE player_alias ADD COLUMN {quote_ident(col)} {definition};")


def _create_player_resolution_audit_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_resolution_audit (
            audit_id TEXT PRIMARY KEY,
            alias_name TEXT NOT NULL,
            alias_source TEXT NOT NULL,
            match_id TEXT NULL,
            reason TEXT NOT NULL,
            detail TEXT NULL,
            attempted_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_player_resolution_audit_alias "
        "ON player_resolution_audit(alias_name, alias_source);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_resolution_audit_match_id ON player_resolution_audit(match_id);")


def _create_api_request_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_request_cache (
            cache_key TEXT PRIMARY KEY,
            endpoint_type TEXT NOT NULL,
            match_id TEXT NOT NULL,
            status_code INTEGER NULL,
            success_flag INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NULL,
            payload_hash TEXT NULL,
            error_type TEXT NULL,
            error_message TEXT NULL,
            fetched_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_request_cache_endpoint_match "
        "ON api_request_cache(endpoint_type, match_id);"
    )


def _migrate_api_request_cache_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(api_request_cache);").fetchall()
    if not rows:
        return
    actual = [row[1] for row in rows]
    add_columns = {
        "status_code": "INTEGER NULL",
        "success_flag": "INTEGER NOT NULL DEFAULT 0",
        "payload_json": "TEXT NULL",
        "payload_hash": "TEXT NULL",
        "error_type": "TEXT NULL",
        "error_message": "TEXT NULL",
        "fetched_at": "TEXT NULL",
        "created_at": "TEXT NULL",
        "updated_at": "TEXT NULL",
    }
    for col, definition in add_columns.items():
        if col not in actual:
            conn.execute(f"ALTER TABLE api_request_cache ADD COLUMN {quote_ident(col)} {definition};")


def _create_team_profile_metrics_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_profile_metrics (
            season INTEGER NOT NULL,
            team TEXT NOT NULL,
            metric_key TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            profile TEXT NOT NULL,
            category TEXT NOT NULL,
            display_order INTEGER NOT NULL,
            description TEXT NULL,
            raw_value REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (season, team, metric_key)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_team_profile_metrics_season ON team_profile_metrics(season);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_team_profile_metrics_team ON team_profile_metrics(team);")


def _migrate_team_profile_metrics_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(team_profile_metrics);").fetchall()
    if not rows:
        return
    actual = [row[1] for row in rows]
    add_columns = {
        "description": "TEXT NULL",
        "created_at": "TEXT NULL",
        "updated_at": "TEXT NULL",
    }
    for col, definition in add_columns.items():
        if col not in actual:
            conn.execute(f"ALTER TABLE team_profile_metrics ADD COLUMN {quote_ident(col)} {definition};")


def _create_derived_refresh_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS derived_refresh_state (
            artifact TEXT NOT NULL,
            season INTEGER NOT NULL,
            calculation_version INTEGER NOT NULL,
            finalized_match_count INTEGER NOT NULL,
            source_max_updated_at TEXT NULL,
            last_success_at TEXT NULL,
            last_attempt_at TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_written INTEGER NOT NULL DEFAULT 0,
            error_message TEXT NULL,
            PRIMARY KEY (artifact, season)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_refresh_state_artifact ON derived_refresh_state(artifact);")


def _migrate_derived_refresh_state_table(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(derived_refresh_state);").fetchall()
    if not rows:
        return
    actual = [row[1] for row in rows]
    add_columns = {
        "calculation_version": "INTEGER NOT NULL DEFAULT 1",
        "finalized_match_count": "INTEGER NOT NULL DEFAULT 0",
        "source_max_updated_at": "TEXT NULL",
        "last_success_at": "TEXT NULL",
        "last_attempt_at": "TEXT NULL",
        "status": "TEXT NOT NULL DEFAULT 'unknown'",
        "rows_written": "INTEGER NOT NULL DEFAULT 0",
        "error_message": "TEXT NULL",
    }
    for col, definition in add_columns.items():
        if col not in actual:
            conn.execute(f"ALTER TABLE derived_refresh_state ADD COLUMN {quote_ident(col)} {definition};")


def create_schema(db_path: Path | str = DB_PATH) -> None:
    ensure_phase1_dirs()
    with transaction(db_path) as conn:
        _create_ball_by_ball_table(conn)
        _migrate_ball_by_ball_table(conn)
        _create_match_list_table(conn)
        _migrate_match_list_table(conn)
        _create_match_schedule_table(conn)
        _migrate_match_schedule_table(conn)
        _create_match_poll_state_table(conn)
        _migrate_match_poll_state_table(conn)
        _create_prediction_history_table(conn)
        _migrate_prediction_history_table(conn)
        _create_api_fetch_log_table(conn)
        _create_raw_api_responses_table(conn)
        _create_player_master_table(conn)
        _migrate_player_master_table(conn)
        _create_player_alias_table(conn)
        _migrate_player_alias_table(conn)
        _create_player_resolution_audit_table(conn)
        _create_api_request_cache_table(conn)
        _migrate_api_request_cache_table(conn)
        _create_team_profile_metrics_table(conn)
        _migrate_team_profile_metrics_table(conn)
        _create_derived_refresh_state_table(conn)
        _migrate_derived_refresh_state_table(conn)


def recreate_ball_by_ball_table(db_path: Path | str = DB_PATH) -> None:
    ensure_phase1_dirs()
    with transaction(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS ball_by_ball;")
        _create_ball_by_ball_table(conn)


def expected_table_columns() -> dict[str, list[str]]:
    return {
        "ball_by_ball": BALL_BY_BALL_BASE_COLUMNS + BALL_BY_BALL_HELPER_COLUMNS,
        "match_list": MATCH_LIST_BASE_COLUMNS + MATCH_LIST_HELPER_COLUMNS,
    }


def validate_core_table_columns(conn: sqlite3.Connection) -> None:
    expectations = expected_table_columns()
    for table_name, expected in expectations.items():
        rows = conn.execute(f"PRAGMA table_info({quote_ident(table_name)});").fetchall()
        actual = [row[1] for row in rows]
        missing = [col for col in expected if col not in actual]
        extra = [col for col in actual if col not in expected]
        if missing or extra:
            raise ValueError(
                f"Schema mismatch for {table_name}.\n"
                f"Missing: {missing}\n"
                f"Extra:   {extra}\n"
                f"Expected:{expected}\n"
                f"Actual:  {actual}"
            )
