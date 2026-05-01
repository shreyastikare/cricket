import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ipl  # noqa: E402


class FakeBaselineModel:
    def __init__(self, baseline_by_match_id):
        self.baseline_by_match_id = baseline_by_match_id
        self.seen_match_ids = []

    def predict_proba(self, frame):
        match_id = int(frame.iloc[0]["match_id"])
        self.seen_match_ids.append(match_id)
        baseline = float(self.baseline_by_match_id[match_id])
        return np.array([[1 - baseline, baseline]])


class CalculateImpactTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_load_resource_params = ipl.load_resource_params
        self._old_resource_function = ipl.resource_function
        ipl.load_resource_params = lambda: {}
        ipl.resource_function = lambda frame, params: [0.0] * len(frame)

    def tearDown(self) -> None:
        ipl.load_resource_params = self._old_load_resource_params
        ipl.resource_function = self._old_resource_function

    def test_first_second_innings_ball_uses_final_first_innings_probability(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-01",
                    "match_id": 1,
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

        result = ipl.calculate_impact(frame)
        first_second_innings_ball = result[
            (result["innings"] == 2) & (result["team_balls"] == 1)
        ].iloc[0]

        self.assertAlmostEqual(20.0, first_second_innings_ball["prob_bowling_diff"])
        self.assertAlmostEqual(-20.0, first_second_innings_ball["bowler_delta"])
        self.assertAlmostEqual(20.0, first_second_innings_ball["batter_delta"])

    def test_model_baseline_adjusts_first_row_only(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-01",
                    "match_id": 1,
                    "innings": 1,
                    "team_balls": 1,
                    "team_runs": 4,
                    "team_wicket": 0,
                    "balls_remaining": 119,
                    "wickets_remaining": 10,
                    "y_prob": 0.55,
                    "extra_type": None,
                    "batter_runs": 4,
                    "wicket_taken": 0,
                },
                {
                    "date": "2026-04-01",
                    "match_id": 1,
                    "innings": 1,
                    "team_balls": 2,
                    "team_runs": 6,
                    "team_wicket": 0,
                    "balls_remaining": 118,
                    "wickets_remaining": 10,
                    "y_prob": 0.60,
                    "extra_type": None,
                    "batter_runs": 2,
                    "wicket_taken": 0,
                },
            ]
        )

        result = ipl.calculate_impact(frame, model=FakeBaselineModel({1: 0.40}))

        first_ball = result[result["team_balls"] == 1].iloc[0]
        second_ball = result[result["team_balls"] == 2].iloc[0]
        self.assertAlmostEqual(15.0, first_ball["prob_bowling_diff"])
        self.assertAlmostEqual(5.0, second_ball["prob_bowling_diff"])
        self.assertAlmostEqual(-15.0, first_ball["prob_batting_diff"])

    def test_model_baseline_is_per_match(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-01",
                    "match_id": 1,
                    "innings": 1,
                    "team_balls": 1,
                    "team_runs": 4,
                    "team_wicket": 0,
                    "balls_remaining": 119,
                    "wickets_remaining": 10,
                    "y_prob": 0.55,
                    "extra_type": None,
                    "batter_runs": 4,
                    "wicket_taken": 0,
                },
                {
                    "date": "2026-04-02",
                    "match_id": 2,
                    "innings": 1,
                    "team_balls": 1,
                    "team_runs": 1,
                    "team_wicket": 0,
                    "balls_remaining": 119,
                    "wickets_remaining": 10,
                    "y_prob": 0.45,
                    "extra_type": None,
                    "batter_runs": 1,
                    "wicket_taken": 0,
                },
                {
                    "date": "2026-04-02",
                    "match_id": 2,
                    "innings": 1,
                    "team_balls": 2,
                    "team_runs": 3,
                    "team_wicket": 0,
                    "balls_remaining": 118,
                    "wickets_remaining": 10,
                    "y_prob": 0.50,
                    "extra_type": None,
                    "batter_runs": 2,
                    "wicket_taken": 0,
                },
            ]
        )
        model = FakeBaselineModel({1: 0.40, 2: 0.30})

        result = ipl.calculate_impact(frame, model=model)

        first_match = result[result["match_id"] == 1].iloc[0]
        second_match_first = result[(result["match_id"] == 2) & (result["team_balls"] == 1)].iloc[0]
        second_match_second = result[(result["match_id"] == 2) & (result["team_balls"] == 2)].iloc[0]
        self.assertEqual([1, 2], model.seen_match_ids)
        self.assertAlmostEqual(15.0, first_match["prob_bowling_diff"])
        self.assertAlmostEqual(15.0, second_match_first["prob_bowling_diff"])
        self.assertAlmostEqual(5.0, second_match_second["prob_bowling_diff"])


if __name__ == "__main__":
    unittest.main()
