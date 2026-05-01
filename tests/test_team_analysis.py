import unittest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from team_analysis import (  # noqa: E402
    aggregate_team_phase_impact,
    aggregate_team_role_impact,
    benchmark_seasons_2008_2025,
    build_match_table_rows,
    compute_profile_percentiles,
    compute_profile_season_ranks,
    compute_team_profile_metrics,
    filter_team_season_matches,
    generate_team_identity_summary,
    identity_category_averages,
    phase_label,
    profile_metric_subtitle,
    result_for_team,
    role_label,
    select_profile_strength_cards,
    team_record,
    team_phase_heatmap_height,
    team_standings_rank,
    team_total_impact,
    top_impact_player_by_match,
)


class TeamAnalysisHelperTests(unittest.TestCase):
    def _matches(self):
        return pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "date": "2026-03-29",
                    "bat_first": "Team A",
                    "bowl_first": "Team B",
                    "match_won_by": "Team A",
                    "win_outcome": "10 runs",
                    "result_type": "",
                    "status": "complete",
                    "event_match_no": "1",
                    "stage": "Group Stage",
                    "playoff_match": 0,
                },
                {
                    "match_id": 2,
                    "date": "2026-03-30",
                    "bat_first": "Team C",
                    "bowl_first": "Team A",
                    "match_won_by": "Team C",
                    "win_outcome": "3 wickets",
                    "result_type": "",
                    "status": "complete",
                    "event_match_no": "2",
                    "stage": "Group Stage",
                    "playoff_match": 0,
                },
                {
                    "match_id": 3,
                    "date": "2026-03-31",
                    "bat_first": "Team B",
                    "bowl_first": "Team C",
                    "match_won_by": "Team B",
                    "win_outcome": "5 runs",
                    "result_type": "",
                    "status": "complete",
                    "event_match_no": "3",
                    "stage": "Group Stage",
                    "playoff_match": 0,
                },
            ]
        )

    def _impact_balls(self):
        return pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "innings": 1,
                    "batting_team": "Team A",
                    "bowling_team": "Team B",
                    "over": 0,
                    "bat_pos": 1,
                    "batter": "A Opener",
                    "bowler": "B Bowler",
                    "batter_delta": 5.0,
                    "bowler_delta": -5.0,
                    "prob_bowling_diff": -5.0,
                    "valid_ball": 1,
                    "runs_batter": 1,
                    "runs_bowler": 1,
                    "bowler_wicket": 0,
                },
                {
                    "match_id": 1,
                    "innings": 1,
                    "batting_team": "Team A",
                    "bowling_team": "Team B",
                    "over": 7,
                    "bat_pos": 4,
                    "batter": "A Middle",
                    "bowler": "B Bowler",
                    "batter_delta": -2.0,
                    "bowler_delta": 2.0,
                    "prob_bowling_diff": 2.0,
                    "valid_ball": 1,
                    "runs_batter": 0,
                    "runs_bowler": 0,
                    "bowler_wicket": 0,
                },
                {
                    "match_id": 1,
                    "innings": 2,
                    "batting_team": "Team B",
                    "bowling_team": "Team A",
                    "over": 15,
                    "bat_pos": 7,
                    "batter": "B Finisher",
                    "bowler": "A Bowler",
                    "batter_delta": -3.0,
                    "bowler_delta": 3.0,
                    "prob_bowling_diff": -3.0,
                    "valid_ball": 1,
                    "runs_batter": 0,
                    "runs_bowler": 0,
                    "bowler_wicket": 1,
                },
                {
                    "match_id": 2,
                    "innings": 2,
                    "batting_team": "Team A",
                    "bowling_team": "Team C",
                    "over": 0,
                    "bat_pos": 7,
                    "batter": "A Finisher",
                    "bowler": "C Bowler",
                    "batter_delta": 1.0,
                    "bowler_delta": -1.0,
                    "prob_bowling_diff": 1.0,
                    "valid_ball": 1,
                    "runs_batter": 0,
                    "runs_bowler": 0,
                    "bowler_wicket": 0,
                },
                {
                    "match_id": 2,
                    "innings": 1,
                    "batting_team": "Team C",
                    "bowling_team": "Team A",
                    "over": 6,
                    "bat_pos": 1,
                    "batter": "C Opener",
                    "bowler": "A Bowler",
                    "batter_delta": 4.0,
                    "bowler_delta": -4.0,
                    "prob_bowling_diff": -4.0,
                    "valid_ball": 1,
                    "runs_batter": 4,
                    "runs_bowler": 4,
                    "bowler_wicket": 0,
                },
            ]
        )

    def test_phase_and_role_boundaries(self):
        self.assertEqual("Powerplay", phase_label(0))
        self.assertEqual("Powerplay", phase_label(5.999))
        self.assertEqual("Middle overs", phase_label(6))
        self.assertEqual("Middle overs", phase_label(14.999))
        self.assertEqual("Death overs", phase_label(15))
        self.assertEqual("Death overs", phase_label(19.999))
        self.assertIsNone(phase_label(20))

        self.assertEqual("Top order", role_label(1))
        self.assertEqual("Top order", role_label(3))
        self.assertEqual("Middle order", role_label(4))
        self.assertEqual("Middle order", role_label(6))
        self.assertEqual("Finishers", role_label(7))
        self.assertIsNone(role_label(None))

    def test_filter_team_season_matches_and_record(self):
        matches = filter_team_season_matches(self._matches(), "Team A")
        self.assertEqual([1, 2], matches["match_id"].tolist())

        record = team_record(self._matches(), "Team A")
        self.assertEqual({"matches": 2, "wins": 1, "losses": 1, "no_results": 0}, record)

    def test_team_phase_and_role_impact(self):
        matches = self._matches()
        impact_balls = self._impact_balls()

        phase = aggregate_team_phase_impact(impact_balls, matches, "Team A")
        heatmap = phase["heatmap"].set_index("match_id")
        self.assertAlmostEqual(5.0, heatmap.loc[1, "PP Batting"])
        self.assertAlmostEqual(-2.0, heatmap.loc[1, "Middle Batting"])
        self.assertAlmostEqual(3.0, heatmap.loc[1, "Death Bowling"])
        self.assertAlmostEqual(-4.0, heatmap.loc[2, "Middle Bowling"])
        self.assertAlmostEqual(3.0, phase["averages"]["PP Batting"])

        roles = aggregate_team_role_impact(impact_balls, matches, "Team A").set_index("Role")
        self.assertAlmostEqual(2.5, roles.loc["Top order", "Avg Impact / Match"])
        self.assertAlmostEqual(-1.0, roles.loc["Middle order", "Avg Impact / Match"])
        self.assertAlmostEqual(0.5, roles.loc["Finishers", "Avg Impact / Match"])
        self.assertAlmostEqual(3.0, team_total_impact(impact_balls, matches, "Team A"))

    def test_phase_heatmap_uses_net_batting_impact(self):
        matches = pd.DataFrame(
            [
                {
                    "match_id": 11,
                    "date": "2026-04-01",
                    "bat_first": "Team A",
                    "bowl_first": "Team B",
                    "match_won_by": "Team B",
                    "win_outcome": "1 wicket",
                    "result_type": "",
                    "status": "complete",
                    "event_match_no": "11",
                    "stage": "Group Stage",
                    "playoff_match": 0,
                }
            ]
        )
        impact_balls = pd.DataFrame(
            [
                {
                    "match_id": 11,
                    "innings": 1,
                    "batting_team": "Team A",
                    "bowling_team": "Team B",
                    "over": 2,
                    "batter_delta": 0.0,
                    "bowler_delta": 8.0,
                    "prob_bowling_diff": 8.0,
                    "wicket_taken": 1,
                }
            ]
        )

        phase = aggregate_team_phase_impact(impact_balls, matches, "Team A")
        heatmap = phase["heatmap"].set_index("match_id")
        self.assertAlmostEqual(-8.0, heatmap.loc[11, "PP Batting"])
        self.assertAlmostEqual(-8.0, phase["averages"]["PP Batting"])

    def test_team_profile_metric_aggregation(self):
        metrics = compute_team_profile_metrics(self._impact_balls(), self._matches(), "Team A").set_index("metric_key")

        self.assertAlmostEqual(3.0, metrics.loc["pp_batting", "raw_value"])
        self.assertAlmostEqual(-1.0, metrics.loc["middle_batting", "raw_value"])
        self.assertAlmostEqual(0.5, metrics.loc["finisher_batting", "raw_value"])
        self.assertAlmostEqual(2.5, metrics.loc["top_order_batting", "raw_value"])
        self.assertAlmostEqual(-2.0, metrics.loc["middle_bowling", "raw_value"])
        self.assertAlmostEqual(1.5, metrics.loc["death_bowling", "raw_value"])
        self.assertAlmostEqual(1.5, metrics.loc["wicket_taking_bowling", "raw_value"])
        self.assertAlmostEqual(-2.0, metrics.loc["bowling_pressure", "raw_value"])
        self.assertEqual("Bowling Pressure", metrics.loc["bowling_pressure", "metric_name"])

    def test_profile_phase_metrics_use_net_batting_impact(self):
        matches = pd.DataFrame(
            [
                {
                    "match_id": 21,
                    "date": "2026-04-02",
                    "bat_first": "Team A",
                    "bowl_first": "Team B",
                    "match_won_by": "Team B",
                    "win_outcome": "1 wicket",
                    "result_type": "",
                    "status": "complete",
                    "event_match_no": "21",
                    "stage": "Group Stage",
                    "playoff_match": 0,
                }
            ]
        )
        impact_balls = pd.DataFrame(
            [
                {
                    "match_id": 21,
                    "innings": 1,
                    "batting_team": "Team A",
                    "bowling_team": "Team B",
                    "over": 1,
                    "bat_pos": 2,
                    "batter": "A Batter",
                    "bowler": "B Bowler",
                    "batter_delta": 0.0,
                    "bowler_delta": 7.5,
                    "prob_bowling_diff": 7.5,
                    "wicket_taken": 1,
                    "bowler_wicket": 1,
                }
            ]
        )

        metrics = compute_team_profile_metrics(impact_balls, matches, "Team A").set_index("metric_key")
        self.assertAlmostEqual(-7.5, metrics.loc["pp_batting", "raw_value"])

    def test_profile_percentiles_use_less_than_or_equal_ties(self):
        selected = pd.DataFrame(
            [
                {
                    "metric_key": "pp_batting",
                    "metric_name": "Powerplay batting",
                    "profile": "batting",
                    "category": "Batting - Phase",
                    "raw_value": 5.0,
                    "display_order": 1,
                    "description": "",
                }
            ]
        )
        benchmark = pd.DataFrame(
            [
                {"metric_key": "pp_batting", "raw_value": 0.0},
                {"metric_key": "pp_batting", "raw_value": 5.0},
                {"metric_key": "pp_batting", "raw_value": 5.0},
                {"metric_key": "pp_batting", "raw_value": 10.0},
            ]
        )

        scored = compute_profile_percentiles(selected, benchmark)

        self.assertAlmostEqual(75.0, scored.iloc[0]["percentile"])

    def test_strength_cards_use_raw_values_not_percentiles(self):
        metrics = compute_team_profile_metrics(self._impact_balls(), self._matches(), "Team A")
        metrics["percentile"] = metrics["metric_key"].map(
            {
                "middle_bowling": 100.0,
                "bowling_pressure": 99.0,
            }
        ).fillna(0.0)

        cards = select_profile_strength_cards(metrics).set_index("card_label")

        self.assertEqual("Powerplay batting", cards.loc["Primary Strength", "metric_name"])
        self.assertEqual("Top-order batting", cards.loc["Secondary Strength", "metric_name"])
        self.assertEqual("Middle-overs bowling", cards.loc["Weakest Area", "metric_name"])

    def test_profile_metric_subtitles_are_specific(self):
        self.assertEqual("Positions 1-3", profile_metric_subtitle("top_order_batting", "Batting - Role"))
        self.assertEqual("Positions 4-6", profile_metric_subtitle("middle_order_batting", "Batting - Role"))
        self.assertEqual("Positions 7+", profile_metric_subtitle("finisher_batting", "Batting - Role"))
        self.assertEqual("Overs 1-6", profile_metric_subtitle("pp_batting", "Batting - Phase"))
        self.assertEqual("Overs 7-15", profile_metric_subtitle("middle_bowling", "Bowling - Phase"))
        self.assertEqual("Overs 16-20", profile_metric_subtitle("death_bowling", "Bowling - Phase"))
        self.assertEqual("Non-wicket impact", profile_metric_subtitle("bowling_pressure", "Bowling - Pressure"))
        self.assertEqual("Impact from wickets", profile_metric_subtitle("wicket_taking_bowling", "Bowling - Wickets"))

    def test_team_standings_rank_uses_selected_team_row(self):
        ranking = pd.DataFrame(
            [
                {"Rank": 1, "Team": "Team B", "Matches": 2},
                {"Rank": 2, "Team": "Team A", "Matches": 2},
            ]
        )

        self.assertEqual(2, team_standings_rank(ranking, "Team A"))
        self.assertIsNone(team_standings_rank(ranking, "Team C"))

    def test_phase_heatmap_height_scales_with_match_count(self):
        self.assertEqual(360, team_phase_heatmap_height(0))
        self.assertEqual(360, team_phase_heatmap_height(2))
        self.assertGreater(team_phase_heatmap_height(12), team_phase_heatmap_height(2))
        self.assertEqual(760, team_phase_heatmap_height(50))

    def test_profile_season_ranks_are_descending_with_ties(self):
        selected = pd.DataFrame(
            [
                {
                    "metric_key": "pp_batting",
                    "metric_name": "Powerplay batting",
                    "profile": "batting",
                    "category": "Batting - Phase",
                    "raw_value": 5.0,
                    "percentile": 50.0,
                    "display_order": 1,
                    "description": "",
                }
            ]
        )
        season_metrics = pd.DataFrame(
            [
                {"team": "Team A", "metric_key": "pp_batting", "raw_value": 5.0},
                {"team": "Team B", "metric_key": "pp_batting", "raw_value": 8.0},
                {"team": "Team C", "metric_key": "pp_batting", "raw_value": 5.0},
            ]
        )

        ranked = compute_profile_season_ranks(selected, season_metrics, "Team A", 2026)

        self.assertEqual(2, ranked.iloc[0]["season_rank"])
        self.assertEqual(3, ranked.iloc[0]["season_rank_total"])
        self.assertEqual(2026, ranked.iloc[0]["season"])

    def test_benchmark_season_filter_excludes_2026(self):
        self.assertEqual([2008, 2024, 2025], benchmark_seasons_2008_2025([2007, 2008, 2024, 2025, 2026]))

    def test_identity_summary_is_deterministic(self):
        phase = aggregate_team_phase_impact(self._impact_balls(), self._matches(), "Team A")
        roles = aggregate_team_role_impact(self._impact_balls(), self._matches(), "Team A")
        categories = identity_category_averages(roles, phase["averages"])

        identity = generate_team_identity_summary("Team A", categories)

        self.assertEqual("Top-order batting impact", identity["primary_strength"])
        self.assertEqual("Death-overs bowling impact", identity["secondary_strength"])
        self.assertEqual("Middle-overs bowling impact", identity["weakest_area"])
        self.assertIn("Team A has most often gained", identity["sentence"])

    def test_result_labels_and_match_table(self):
        matches = self._matches()
        impact_balls = self._impact_balls()
        phase = aggregate_team_phase_impact(impact_balls, matches, "Team A")
        top_players = top_impact_player_by_match(impact_balls, matches, "Team A")

        self.assertEqual("Won by 10 runs", result_for_team(matches.iloc[0], "Team A"))
        self.assertEqual("Lost by 3 wickets", result_for_team(matches.iloc[1], "Team A"))

        table = build_match_table_rows(matches, "Team A", phase["heatmap"], top_players)
        self.assertEqual(["Date", "Match No.", "Opponent", "Result", "Team Total Impact", "Top Impact Player", "Match"], table.columns.tolist())
        self.assertEqual(["Team C", "Team B"], table["Opponent"].tolist())
        self.assertEqual(["Match 2", "Match 1"], table["Match No."].tolist())
        self.assertEqual("A Opener", table.iloc[1]["Top Impact Player"])
        self.assertEqual("[Open](/match-analysis?match_id=1)", table.iloc[1]["Match"])


if __name__ == "__main__":
    unittest.main()
