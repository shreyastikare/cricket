import os
import sqlite3
import sys
import tempfile
import gc
import time
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ipl  # noqa: E402
from ipl.storage.schema import create_schema  # noqa: E402
from ipl.storage.win_probability_cache import (  # noqa: E402
    _assign_predicted_probabilities,
    load_cached_prediction_impact,
    write_cached_match_predictions,
)


class WinProbabilityCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(prefix="ipl-win-prob-cache-", suffix=".db")
        os.close(fd)
        create_schema(self.db_path)

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

    def _sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-04-01",
                    "match_id": 1,
                    "ball_id": "1:1",
                    "innings": 1,
                    "team_balls": 1,
                    "team_runs": 4,
                    "team_wicket": 0,
                    "y_prob": 0.50,
                    "extra_type": None,
                    "batter_runs": 4,
                    "wicket_taken": 0,
                },
                {
                    "date": "2026-04-01",
                    "match_id": 1,
                    "ball_id": "1:3",
                    "innings": 2,
                    "team_balls": 1,
                    "team_runs": 1,
                    "team_wicket": 0,
                    "y_prob": 0.60,
                    "extra_type": None,
                    "batter_runs": 1,
                    "wicket_taken": 0,
                },
                {
                    "date": "2026-04-01",
                    "match_id": 1,
                    "ball_id": "1:2",
                    "innings": 1,
                    "team_balls": 2,
                    "team_runs": 6,
                    "team_wicket": 0,
                    "y_prob": 0.40,
                    "extra_type": None,
                    "batter_runs": 2,
                    "wicket_taken": 0,
                },
            ]
        )

    def test_schema_includes_probability_delta_columns(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(prediction_history);")}

        self.assertIn("prob_bowling_diff", columns)
        self.assertIn("prob_batting_diff", columns)

    def test_cache_round_trip_preserves_calculate_impact_deltas(self) -> None:
        scored = ipl.calculate_impact(self._sample_frame())

        rows_written = write_cached_match_predictions(
            scored,
            db_path=self.db_path,
            model_version="test-model",
            prediction_ts="2026-04-01T00:00:00Z",
        )
        cached = load_cached_prediction_impact(
            self._sample_frame(),
            db_path=self.db_path,
            match_id=1,
            model_version="test-model",
        )

        self.assertEqual(3, rows_written)
        self.assertIsNotNone(cached)
        first_second_innings_ball = cached[
            (cached["innings"] == 2) & (cached["team_balls"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(20.0, first_second_innings_ball["prob_bowling_diff"])
        self.assertAlmostEqual(-20.0, first_second_innings_ball["prob_batting_diff"])
        self.assertAlmostEqual(-20.0, first_second_innings_ball["bowler_delta"])
        self.assertAlmostEqual(20.0, first_second_innings_ball["batter_delta"])

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), SUM(prob_bowling_diff), SUM(prob_batting_diff)
                FROM prediction_history
                WHERE model_type = 'win_probability';
                """
            ).fetchone()
        self.assertEqual(3, row[0])
        self.assertAlmostEqual(10.0, row[1])
        self.assertAlmostEqual(-10.0, row[2])

    def test_cache_with_old_model_version_is_ignored(self) -> None:
        scored = ipl.calculate_impact(self._sample_frame())
        write_cached_match_predictions(
            scored,
            db_path=self.db_path,
            model_version="old-model",
            prediction_ts="2026-04-01T00:00:00Z",
        )

        cached = load_cached_prediction_impact(
            self._sample_frame(),
            db_path=self.db_path,
            match_id=1,
            model_version="new-model",
        )

        self.assertIsNone(cached)

    def test_cache_can_opt_into_latest_stale_model_version(self) -> None:
        scored = ipl.calculate_impact(self._sample_frame())
        write_cached_match_predictions(
            scored,
            db_path=self.db_path,
            model_version="old-model",
            prediction_ts="2026-04-01T00:00:00Z",
        )

        cached = load_cached_prediction_impact(
            self._sample_frame(),
            db_path=self.db_path,
            match_id=1,
            model_version="new-model",
            allow_stale_model_version=True,
        )

        self.assertIsNotNone(cached)
        first_second_innings_ball = cached[
            (cached["innings"] == 2) & (cached["team_balls"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(20.0, first_second_innings_ball["prob_bowling_diff"])

    def test_probability_assignment_aligns_by_ball_id_when_preprocessing_drops_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "ball_id": ["a", "b", "c"],
                "y_prob": [None, None, None],
            }
        )
        features = pd.DataFrame({"ball_id": ["b", "c"]})

        _assign_predicted_probabilities(
            frame,
            frame["ball_id"].isin(["a", "b", "c"]),
            features,
            pd.Series([0.25, 0.75]).to_numpy(),
        )

        self.assertTrue(pd.isna(frame.loc[0, "y_prob"]))
        self.assertAlmostEqual(0.25, frame.loc[1, "y_prob"])
        self.assertAlmostEqual(0.75, frame.loc[2, "y_prob"])


if __name__ == "__main__":
    unittest.main()
