import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PREPROCESSING_PATH = SRC / "ipl" / "pipeline" / "preprocessing.py"

_ORIGINAL_IPL_MODULE = sys.modules.get("ipl")
_IPL_STUB = types.ModuleType("ipl")
_IPL_STUB.load_resource_params = lambda: {}
_IPL_STUB.resource_function = lambda X, params: np.zeros(len(X))
sys.modules["ipl"] = _IPL_STUB

_SPEC = importlib.util.spec_from_file_location("ipl_pipeline_preprocessing_under_test", PREPROCESSING_PATH)
prep = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(prep)


def tearDownModule() -> None:
    if _ORIGINAL_IPL_MODULE is None:
        sys.modules.pop("ipl", None)
    else:
        sys.modules["ipl"] = _ORIGINAL_IPL_MODULE


def _delivery_row(
    *,
    row_id: int,
    match_id: int,
    match_date: str,
    innings: int,
    batting_team: str,
    bowling_team: str,
    team_balls: int,
    team_runs: int,
    team_wicket: int,
    overs_limit: int,
    runs_target: float,
    match_won_by: str,
    event_match_no: int,
) -> dict:
    over = (team_balls - 1) // 6
    ball = ((team_balls - 1) % 6) + 1
    ts = pd.Timestamp(match_date)
    return {
        "id": row_id,
        "match_id": match_id,
        "date": match_date,
        "match_type": "T20",
        "event_name": "Indian Premier League",
        "innings": innings,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "over": over,
        "ball": ball,
        "ball_no": float(f"{over}.{ball}"),
        "batter": f"{batting_team}_batter",
        "bat_pos": 1,
        "runs_batter": 1,
        "balls_faced": 1,
        "bowler": f"{bowling_team}_bowler",
        "valid_ball": 1,
        "runs_extras": 0,
        "runs_total": 1,
        "runs_bowler": 1,
        "runs_not_boundary": False,
        "extra_type": "",
        "non_striker": f"{batting_team}_non_striker",
        "non_striker_pos": 2,
        "wicket_kind": np.nan,
        "player_out": np.nan,
        "fielders": "",
        "runs_target": runs_target,
        "review_batter": "",
        "team_reviewed": "",
        "review_decision": "",
        "umpire": "",
        "umpires_call": False,
        "player_of_match": "",
        "match_won_by": match_won_by,
        "win_outcome": "10 runs",
        "toss_winner": batting_team,
        "toss_decision": "bat",
        "venue": "Some Ground",
        "city": "Some City",
        "day": ts.day,
        "month": ts.month,
        "year": ts.year,
        "season": "2026",
        "gender": "male",
        "team_type": "club",
        "superover_winner": np.nan,
        "result_type": "NA",
        "method": "NA",
        "balls_per_over": 6,
        "overs": overs_limit,
        "event_match_no": event_match_no,
        "stage": "Unknown",
        "match_number": "Unknown",
        "team_runs": team_runs,
        "team_balls": team_balls,
        "team_wicket": team_wicket,
        "new_batter": "",
        "batter_runs": team_runs,
        "batter_balls": team_balls,
        "bowler_wicket": 0,
        "batting_partners": f"('{batting_team}_a', '{batting_team}_b')",
        "next_batter": "",
        "striker_out": False,
    }


def _build_sample_ball_data() -> pd.DataFrame:
    rows = []
    row_id = 1

    # Match 1: shortened to 11 overs (66 balls).
    m1 = 1527686
    m1_first = [(20, 50, 1), (53, 120, 2), (66, 150, 3)]
    m1_second = [(20, 40, 1), (53, 100, 5), (66, 140, 8)]
    for team_balls, team_runs, team_wicket in m1_first:
        rows.append(_delivery_row(
            row_id = row_id,
            match_id = m1,
            match_date = "2026-04-08",
            innings = 1,
            batting_team = "Rajasthan Royals",
            bowling_team = "Mumbai Indians",
            team_balls = team_balls,
            team_runs = team_runs,
            team_wicket = team_wicket,
            overs_limit = 11,
            runs_target = np.nan,
            match_won_by = "Rajasthan Royals",
            event_match_no = 20,
        ))
        row_id += 1
    for team_balls, team_runs, team_wicket in m1_second:
        rows.append(_delivery_row(
            row_id = row_id,
            match_id = m1,
            match_date = "2026-04-08",
            innings = 2,
            batting_team = "Mumbai Indians",
            bowling_team = "Rajasthan Royals",
            team_balls = team_balls,
            team_runs = team_runs,
            team_wicket = team_wicket,
            overs_limit = 11,
            runs_target = 151,
            match_won_by = "Rajasthan Royals",
            event_match_no = 20,
        ))
        row_id += 1

    # Match 2: standard 20 overs (120 balls).
    m2 = 1527999
    m2_first = [(36, 60, 1), (96, 150, 4), (120, 200, 6)]
    m2_second = [(36, 55, 2), (96, 145, 5), (120, 190, 8)]
    for team_balls, team_runs, team_wicket in m2_first:
        rows.append(_delivery_row(
            row_id = row_id,
            match_id = m2,
            match_date = "2026-04-12",
            innings = 1,
            batting_team = "Chennai Super Kings",
            bowling_team = "Delhi Capitals",
            team_balls = team_balls,
            team_runs = team_runs,
            team_wicket = team_wicket,
            overs_limit = 20,
            runs_target = np.nan,
            match_won_by = "Chennai Super Kings",
            event_match_no = 28,
        ))
        row_id += 1
    for team_balls, team_runs, team_wicket in m2_second:
        rows.append(_delivery_row(
            row_id = row_id,
            match_id = m2,
            match_date = "2026-04-12",
            innings = 2,
            batting_team = "Delhi Capitals",
            bowling_team = "Chennai Super Kings",
            team_balls = team_balls,
            team_runs = team_runs,
            team_wicket = team_wicket,
            overs_limit = 20,
            runs_target = 201,
            match_won_by = "Chennai Super Kings",
            event_match_no = 28,
        ))
        row_id += 1

    return pd.DataFrame(rows)


class DynamicRemainingBallFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.csv_path = ROOT / f"_tmp_preprocessing_dynamic_{os.getpid()}_{id(self)}.csv"
        self.addCleanup(lambda: self.csv_path.unlink(missing_ok = True))
        _build_sample_ball_data().to_csv(self.csv_path, index = False)

    def _model_features(self, loaded_df: pd.DataFrame) -> pd.DataFrame:
        match_ids = sorted(loaded_df["match_id"].unique())
        first_innings_runs = (
            loaded_df[loaded_df["innings"] == 1]
            .groupby("match_id")["team_runs"]
            .max()
            .reindex(match_ids)
            .fillna(0)
            .to_numpy()
        )
        return pd.DataFrame({
            "match_id": match_ids,
            "first_innings_runs": first_innings_runs,
            "year": [2026] * len(match_ids),
            "batting_team_prior_powerplay_NRR": [0.0] * len(match_ids),
            "batting_team_prior_middle_NRR": [0.0] * len(match_ids),
            "batting_team_prior_death_NRR": [0.0] * len(match_ids),
            "bowling_team_prior_powerplay_NRR": [0.0] * len(match_ids),
            "bowling_team_prior_middle_NRR": [0.0] * len(match_ids),
            "bowling_team_prior_death_NRR": [0.0] * len(match_ids),
            "batting_team_prior_run_rate": [8.0] * len(match_ids),
            "batting_team_prior_avg_runs_conceded_per_wicket": [30.0] * len(match_ids),
        })

    def test_load_data_generates_dynamic_remaining_balls_for_11_and_20_over_matches(self) -> None:
        df = prep.load_data(path = self.csv_path)

        short_row = df[(df["match_id"] == 1527686) & (df["innings"] == 2) & (df["team_balls"] == 53)].iloc[0]
        self.assertEqual(short_row["balls_remaining"], 13)
        self.assertEqual(short_row["powerplay_balls_remaining"], 0)
        self.assertEqual(short_row["middle_balls_remaining"], 0)
        self.assertEqual(short_row["death_balls_remaining"], 13)

        full_row = df[(df["match_id"] == 1527999) & (df["innings"] == 1) & (df["team_balls"] == 96)].iloc[0]
        self.assertEqual(full_row["balls_remaining"], 24)
        self.assertEqual(full_row["powerplay_balls_remaining"], 0)
        self.assertEqual(full_row["middle_balls_remaining"], 0)
        self.assertEqual(full_row["death_balls_remaining"], 24)

    def test_load_data_live_generates_same_dynamic_remaining_ball_columns(self) -> None:
        df = prep.load_data_live(path = self.csv_path)
        row = df[(df["match_id"] == 1527686) & (df["innings"] == 2) & (df["team_balls"] == 20)].iloc[0]
        self.assertEqual(row["balls_remaining"], 46)
        self.assertEqual(row["powerplay_balls_remaining"], 0)
        self.assertEqual(row["middle_balls_remaining"], 33)
        self.assertEqual(row["death_balls_remaining"], 13)

    def test_preprocess_functions_preserve_precomputed_remaining_ball_features(self) -> None:
        df = prep.load_data(path = self.csv_path)
        model_features = self._model_features(df)

        first = prep.preprocess_first_innings(df, model_features, min_overs = 0, max_overs = 20)
        src_first = df[(df["match_id"] == 1527686) & (df["innings"] == 1) & (df["team_balls"] == 20)].iloc[0]
        out_first = first[(first["match_id"] == 1527686) & (first["team_balls"] == 20)].iloc[0]
        self.assertEqual(out_first["balls_remaining"], src_first["balls_remaining"])
        self.assertEqual(out_first["powerplay_balls_remaining"], src_first["powerplay_balls_remaining"])
        self.assertEqual(out_first["middle_balls_remaining"], src_first["middle_balls_remaining"])
        self.assertEqual(out_first["death_balls_remaining"], src_first["death_balls_remaining"])

        second = prep.preprocess_second_innings(df, model_features)
        src_second = df[(df["match_id"] == 1527686) & (df["innings"] == 2) & (df["team_balls"] == 53)].iloc[0]
        out_second = second[(second["match_id"] == 1527686) & (second["team_balls"] == 53)].iloc[0]
        self.assertEqual(out_second["balls_remaining"], src_second["balls_remaining"])
        self.assertEqual(out_second["powerplay_balls_remaining"], src_second["powerplay_balls_remaining"])
        self.assertEqual(out_second["middle_balls_remaining"], src_second["middle_balls_remaining"])
        self.assertEqual(out_second["death_balls_remaining"], src_second["death_balls_remaining"])

    def test_match_list_uses_dynamic_phase_boundaries(self) -> None:
        df = prep.load_data(path = self.csv_path)
        matches = prep.match_list(df)

        short = matches[matches["match_id"] == 1527686].iloc[0]
        self.assertEqual(short["first_innings_balls"], 66)
        self.assertEqual(short["first_innings_powerplay_balls"], 20)
        self.assertEqual(short["first_innings_middle_balls"], 33)
        self.assertEqual(short["first_innings_death_balls"], 13)
        self.assertEqual(short["first_innings_powerplay_runs"], 50)
        self.assertEqual(short["first_innings_middle_runs"], 70)
        self.assertEqual(short["first_innings_death_runs"], 30)

        full = matches[matches["match_id"] == 1527999].iloc[0]
        self.assertEqual(full["first_innings_powerplay_balls"], 36)
        self.assertEqual(full["first_innings_middle_balls"], 60)
        self.assertEqual(full["first_innings_death_balls"], 24)

    def test_missing_or_invalid_overs_falls_back_to_120_ball_innings(self) -> None:
        raw = pd.DataFrame({
            "team_runs": [80, 80],
            "team_balls": [60, 60],
            "team_wicket": [3, 3],
            "innings": [2, 2],
            "runs_target": [121, 121],
            "overs": [np.nan, 0],
            "balls_per_over": [np.nan, -1],
        })
        out = prep._apply_dynamic_remaining_ball_features(raw)

        for _, row in out.iterrows():
            self.assertEqual(row["balls_remaining"], 60)
            self.assertEqual(row["powerplay_balls_remaining"], 0)
            self.assertEqual(row["middle_balls_remaining"], 36)
            self.assertEqual(row["death_balls_remaining"], 24)


if __name__ == "__main__":
    unittest.main()
