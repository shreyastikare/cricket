import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
SRC_DIR = ROOT / "src"
for path in [str(APP_DIR), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from leaderboard import Leaderboard  # noqa: E402
from theme_config import PLOTLY_COLORS  # noqa: E402


class LeaderboardImpactTests(unittest.TestCase):
    def test_second_innings_model_mask_excludes_rows_beyond_first_innings_length(self) -> None:
        leaderboard = Leaderboard.__new__(Leaderboard)
        frame = pd.DataFrame(
            {
                "innings": [1, 1, 2, 2, 2],
                "team_balls": [1, 2, 1, 2, 3],
                "required_runs": [None, None, 5, 1, 1],
            }
        )
        base_mask = frame["innings"].eq(2) & pd.to_numeric(frame["required_runs"], errors="coerce").gt(0)

        mask = leaderboard._second_innings_model_mask(frame, base_mask)

        self.assertEqual(mask.tolist(), [False, False, True, True, False])

    def test_batter_rate_charts_use_second_innings_orange(self) -> None:
        leaderboard = Leaderboard.__new__(Leaderboard)
        leaderboard.batter_stats = pd.DataFrame(
            {
                "Batter": ["Batter A", "Batter B"],
                "Team": ["Team A", "Team B"],
                "Matches": [3, 2],
                "Runs": [220, 150],
                "Strike Rate": [165.25, 142.50],
                "Average": [55.0, 37.5],
                "4": [20, 14],
                "6": [8, 3],
            }
        )

        strike_rate_fig = leaderboard.plot_strike_rate(n=2, min_runs=100)
        average_fig = leaderboard.plot_batting_average(n=2, min_runs=100)

        self.assertEqual(strike_rate_fig.data[0].marker.color, PLOTLY_COLORS["innings_2"])
        self.assertEqual(average_fig.data[0].marker.color, PLOTLY_COLORS["innings_2"])

    def test_total_boundaries_sixes_use_impact_blue(self) -> None:
        leaderboard = Leaderboard.__new__(Leaderboard)
        leaderboard.batter_stats = pd.DataFrame(
            {
                "Batter": ["Batter A", "Batter B"],
                "Team": ["Team A", "Team B"],
                "Matches": [3, 2],
                "4": [20, 14],
                "6": [8, 3],
            }
        )

        fig = leaderboard.plot_total_boundaries_stacked(n=2)

        self.assertEqual(fig.data[1].marker.color, PLOTLY_COLORS["leaderboard_impact"])

    def test_total_boundaries_fours_use_success_card_green(self) -> None:
        leaderboard = Leaderboard.__new__(Leaderboard)
        leaderboard.batter_stats = pd.DataFrame(
            {
                "Batter": ["Batter A", "Batter B"],
                "Team": ["Team A", "Team B"],
                "Matches": [3, 2],
                "4": [20, 14],
                "6": [8, 3],
            }
        )

        fig = leaderboard.plot_total_boundaries_stacked(n=2)

        self.assertEqual(fig.data[0].marker.color, PLOTLY_COLORS["success_card"])


if __name__ == "__main__":
    unittest.main()
