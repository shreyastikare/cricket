import os
import sqlite3
import sys
import tempfile
import gc
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl.storage.schema import create_schema
from ipl.storage.team_profile_metrics import (  # noqa: E402
    TEAM_PROFILE_METRIC_COLUMNS,
    load_team_profile_metrics,
    refresh_team_profile_metrics_for_season,
)


class TeamProfileMetricsRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(prefix="ipl-team-profile-", suffix=".db")
        os.close(fd)
        create_schema(self.db_path)
        self.now_ts = "2026-04-12T00:00:00Z"

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            for attempt in range(5):
                try:
                    gc.collect()
                    os.remove(self.db_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.1)

    def _insert_finalized_match(self, match_id: int, season: int = 2026, updated_at: str | None = None) -> None:
        ts = updated_at or self.now_ts
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO match_list (
                    match_id, date, bat_first, bowl_first, year, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (match_id, "2026-04-12", "Team A", "Team B", season, "complete", ts, ts),
            )
            conn.execute(
                """
                INSERT INTO ball_by_ball (
                    match_id, ball_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?);
                """,
                (match_id, f"{match_id}:1", ts, ts),
            )

    def _metric_frame(self, season: int = 2026, team: str = "Team A", raw_value: float = 1.25) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "season": season,
                    "team": team,
                    "metric_key": "pp_batting",
                    "metric_name": "Powerplay batting",
                    "profile": "batting",
                    "category": "Batting - Phase",
                    "display_order": 1,
                    "description": "",
                    "raw_value": raw_value,
                    "created_at": self.now_ts,
                    "updated_at": self.now_ts,
                }
            ],
            columns=TEAM_PROFILE_METRIC_COLUMNS,
        )

    def test_schema_creates_refresh_tables_idempotently(self) -> None:
        create_schema(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table';"
                ).fetchall()
            }
        self.assertIn("team_profile_metrics", tables)
        self.assertIn("derived_refresh_state", tables)

    def test_refresh_writes_then_skips_when_source_state_unchanged(self) -> None:
        self._insert_finalized_match(1)
        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            return_value=self._metric_frame(),
        ) as mocked_compute:
            first = refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path)
            second = refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path)

        self.assertEqual("success", first["status"])
        self.assertFalse(first["skipped"])
        self.assertEqual(1, first["rows_written"])
        self.assertEqual("skipped", second["status"])
        self.assertTrue(second["skipped"])
        mocked_compute.assert_called_once()
        cached = load_team_profile_metrics(db_path=self.db_path, season=2026)
        self.assertEqual(1, len(cached))
        self.assertAlmostEqual(1.25, cached.iloc[0]["raw_value"])

    def test_force_recomputes_even_when_source_state_matches(self) -> None:
        self._insert_finalized_match(1)
        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            return_value=self._metric_frame(raw_value=1.0),
        ):
            refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path)
        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            return_value=self._metric_frame(raw_value=2.0),
        ) as mocked_compute:
            result = refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path, force=True)

        self.assertEqual("success", result["status"])
        mocked_compute.assert_called_once()
        cached = load_team_profile_metrics(db_path=self.db_path, season=2026)
        self.assertAlmostEqual(2.0, cached.iloc[0]["raw_value"])

    def test_source_updated_at_change_recomputes_only_target_season(self) -> None:
        self._insert_finalized_match(1, season=2026)
        self._insert_finalized_match(2, season=2025)
        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            return_value=self._metric_frame(season=2026, raw_value=1.0),
        ):
            refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path)
        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            return_value=self._metric_frame(season=2025, raw_value=9.0),
        ):
            refresh_team_profile_metrics_for_season(season=2025, db_path=self.db_path)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE ball_by_ball SET updated_at = ? WHERE match_id = ?;",
                ("2026-04-12T00:05:00Z", 1),
            )

        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            return_value=self._metric_frame(season=2026, raw_value=3.0),
        ) as mocked_compute:
            result = refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path)

        self.assertEqual("success", result["status"])
        mocked_compute.assert_called_once()
        cached_2026 = load_team_profile_metrics(db_path=self.db_path, season=2026)
        cached_2025 = load_team_profile_metrics(db_path=self.db_path, season=2025)
        self.assertAlmostEqual(3.0, cached_2026.iloc[0]["raw_value"])
        self.assertAlmostEqual(9.0, cached_2025.iloc[0]["raw_value"])

    def test_failure_records_state_and_preserves_existing_rows(self) -> None:
        self._insert_finalized_match(1)
        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            return_value=self._metric_frame(raw_value=4.0),
        ):
            refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE ball_by_ball SET updated_at = ? WHERE match_id = ?;",
                ("2026-04-12T00:10:00Z", 1),
            )

        with patch(
            "ipl.storage.team_profile_metrics._compute_profile_rows",
            side_effect=RuntimeError("boom"),
        ):
            result = refresh_team_profile_metrics_for_season(season=2026, db_path=self.db_path)

        self.assertEqual("failed", result["status"])
        cached = load_team_profile_metrics(db_path=self.db_path, season=2026)
        self.assertAlmostEqual(4.0, cached.iloc[0]["raw_value"])
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT status, error_message
                FROM derived_refresh_state
                WHERE artifact = 'team_profile_metrics' AND season = 2026;
                """
            ).fetchone()
        self.assertEqual("failed", row[0])
        self.assertIn("boom", row[1])


if __name__ == "__main__":
    unittest.main()
