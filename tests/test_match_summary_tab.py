import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
SRC_DIR = ROOT / "src"


class _Component:
    def __init__(self, *children, **kwargs):
        if "children" in kwargs:
            self.children = kwargs.pop("children")
        elif len(children) == 1:
            self.children = children[0]
        else:
            self.children = list(children)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeDash:
    def __init__(self, *args, **kwargs):
        self.server = object()
        self.layout = None

    def callback(self, *args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    def clientside_callback(self, *args, **kwargs):
        return None


class _FakeFigure:
    def __init__(self):
        self.layout_updates = []
        self.traces = []
        self.xaxis_updates = []
        self.yaxis_updates = []
        self.hlines = []
        self.vlines = []
        self.annotations = []

    def update_layout(self, *args, **kwargs):
        self.layout_updates.append((args, kwargs))
        return self

    def add_trace(self, trace, *args, **kwargs):
        self.traces.append(trace)
        return self

    def update_xaxes(self, *args, **kwargs):
        self.xaxis_updates.append((args, kwargs))
        return self

    def update_yaxes(self, *args, **kwargs):
        self.yaxis_updates.append((args, kwargs))
        return self

    def add_hline(self, *args, **kwargs):
        self.hlines.append((args, kwargs))
        return self

    def add_vline(self, *args, **kwargs):
        self.vlines.append((args, kwargs))
        return self

    def add_annotation(self, *args, **kwargs):
        self.annotations.append((args, kwargs))
        return self


def _factory(name):
    def _make(*args, **kwargs):
        component = _Component(*args, **kwargs)
        component.component_name = name
        return component

    return _make


def _install_module(name: str, module: types.ModuleType, replacements: dict[str, types.ModuleType | None]) -> None:
    replacements[name] = sys.modules.get(name)
    sys.modules[name] = module


def _restore_modules(replacements: dict[str, types.ModuleType | None]) -> None:
    for name, original in replacements.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _load_dashboard_app_module():
    replacements: dict[str, types.ModuleType | None] = {}

    dash_mod = types.ModuleType("dash")
    dash_mod.ALL = object()
    dash_mod.Dash = _FakeDash
    dash_mod.Input = lambda *args, **kwargs: ("Input", args, kwargs)
    dash_mod.Output = lambda *args, **kwargs: ("Output", args, kwargs)
    dash_mod.State = lambda *args, **kwargs: ("State", args, kwargs)
    dash_mod.ctx = types.SimpleNamespace(triggered_id = None)
    dash_mod.dcc = types.SimpleNamespace(
        Location = _factory("Location"),
        Store = _factory("Store"),
        Interval = _factory("Interval"),
        Link = _factory("Link"),
        Tabs = _factory("Tabs"),
        Tab = _factory("Tab"),
        Graph = _factory("Graph"),
    )
    dash_mod.html = types.SimpleNamespace(
        Div = _factory("Div"),
        H1 = _factory("H1"),
        H2 = _factory("H2"),
        H4 = _factory("H4"),
        P = _factory("P"),
        Ul = _factory("Ul"),
        Li = _factory("Li"),
        Span = _factory("Span"),
        A = _factory("A"),
        Table = _factory("Table"),
        Thead = _factory("Thead"),
        Tbody = _factory("Tbody"),
        Tr = _factory("Tr"),
        Th = _factory("Th"),
        Td = _factory("Td"),
    )
    dash_mod.dash_table = types.SimpleNamespace(DataTable = _factory("DataTable"))
    _install_module("dash", dash_mod, replacements)

    dbc_mod = types.ModuleType("dash_bootstrap_components")
    dbc_mod.themes = types.SimpleNamespace(FLATLY = "flatly")
    dbc_mod.Card = _factory("Card")
    dbc_mod.CardBody = _factory("CardBody")
    dbc_mod.Button = _factory("Button")
    dbc_mod.Alert = _factory("Alert")
    dbc_mod.Pagination = _factory("Pagination")
    table_factory = _factory("Table")
    table_factory.from_dataframe = lambda *args, **kwargs: _Component(*args, **kwargs)
    dbc_mod.Table = table_factory
    _install_module("dash_bootstrap_components", dbc_mod, replacements)

    leaderboard_mod = types.ModuleType("leaderboard")
    leaderboard_mod.Leaderboard = object
    leaderboard_mod.get_default_leaderboard_season = lambda: None
    leaderboard_mod.get_finalized_season_options = lambda: []
    _install_module("leaderboard", leaderboard_mod, replacements)

    match_mod = types.ModuleType("match")
    match_mod.Match = object
    match_mod.resource_params = {}
    _install_module("match", match_mod, replacements)

    selector_mod = types.ModuleType("match_selector")
    selector_mod.get_match_options = lambda year = None, team = None: []
    selector_mod.get_team_options_for_year = lambda year = None: []
    selector_mod.get_year_options = lambda: []
    _install_module("match_selector", selector_mod, replacements)

    ui_mod = types.ModuleType("ui")
    ui_mod.APP_CONTENT_SHELL_STYLE = {}
    ui_mod.PAGE_WRAPPER_STYLE = {}
    ui_mod.build_match_analysis_page = lambda *args, **kwargs: _Component()
    ui_mod.build_navbar = lambda *args, **kwargs: _Component()
    ui_mod.build_season_leaderboard_page = lambda *args, **kwargs: _Component()
    ui_mod.build_team_analysis_page = lambda *args, **kwargs: _Component()
    _install_module("ui", ui_mod, replacements)

    theme_mod = types.ModuleType("theme_config")
    theme_mod.APP_FONT_STACK = "test-font"
    theme_mod.PLOTLY_BASE_FONT_SIZE = 14
    theme_mod.PLOTLY_FONT_FAMILY = "test-font"
    theme_mod.PLOTLY_HEADER_MARGIN_LEFT = 90
    theme_mod.PLOTLY_HEADER_MARGIN_RIGHT = 90
    theme_mod.PLOTLY_HEADER_MARGIN_TOP = 20
    theme_mod.PLOTLY_HEADER_TITLE_Y = 0.98
    theme_mod.PLOTLY_LABEL_FONT_SIZE = 16
    theme_mod.PLOTLY_COLORS = {
        "innings_1": "#1f77b4",
        "innings_2": "#ff7f0e",
        "reference_line": "#2ca02c",
        "leaderboard_primary": "#1f77b4",
        "leaderboard_secondary": "#ff7f0e",
    }
    _install_module("theme_config", theme_mod, replacements)

    ipl_mod = types.ModuleType("ipl")
    ipl_mod.resource_function = lambda frame, params: np.zeros(len(frame))
    ipl_mod.__path__ = []
    _install_module("ipl", ipl_mod, replacements)

    ipl_storage_mod = types.ModuleType("ipl.storage")
    ipl_storage_mod.__path__ = []
    _install_module("ipl.storage", ipl_storage_mod, replacements)

    team_profile_metrics_mod = types.ModuleType("ipl.storage.team_profile_metrics")
    team_profile_metrics_mod.load_team_profile_metrics = lambda *args, **kwargs: pd.DataFrame()
    team_profile_metrics_mod.refresh_team_profile_metrics_for_season = lambda *args, **kwargs: {"status": "skipped"}
    _install_module("ipl.storage.team_profile_metrics", team_profile_metrics_mod, replacements)

    go_mod = types.ModuleType("plotly.graph_objects")
    go_mod.Figure = _FakeFigure
    go_mod.Bar = lambda *args, **kwargs: ("Bar", args, kwargs)
    go_mod.Scatter = lambda *args, **kwargs: ("Scatter", args, kwargs)
    go_mod.Scatterpolar = lambda *args, **kwargs: ("Scatterpolar", args, kwargs)
    go_mod.Heatmap = lambda *args, **kwargs: ("Heatmap", args, kwargs)
    go_mod.Pie = lambda *args, **kwargs: ("Pie", args, kwargs)
    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod
    _install_module("plotly", plotly_mod, replacements)
    _install_module("plotly.graph_objects", go_mod, replacements)

    spec = importlib.util.spec_from_file_location("dashboard_app_under_test", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    old_path = list(sys.path)
    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(SRC_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        _restore_modules(replacements)
    return module


def _load_match_module():
    replacements: dict[str, types.ModuleType | None] = {}

    ipl_mod = types.ModuleType("ipl")
    ipl_mod.load_resource_params = lambda: {}
    ipl_mod.first_innings_glm_load = lambda: object()
    ipl_mod.second_innings_glm_load = lambda: object()
    ipl_mod.resource_function = lambda frame, params: np.zeros(len(frame))
    ipl_mod.calculate_impact = lambda frame: frame
    ipl_mod.aggregate_impact = lambda frame: pd.DataFrame()
    _install_module("ipl", ipl_mod, replacements)

    plot_theme_mod = types.ModuleType("plot_theme")
    plot_theme_mod.apply_plot_theme = lambda fig: fig
    plot_theme_mod.header_legend_layout = lambda: {}
    plot_theme_mod.innings_color = lambda innings: "#1f77b4" if int(innings) == 1 else "#ff7f0e"
    _install_module("plot_theme", plot_theme_mod, replacements)

    theme_mod = types.ModuleType("theme_config")
    theme_mod.APP_FONT_STACK = "test-font"
    theme_mod.PLOTLY_COLORS = {"innings_1": "#1f77b4", "innings_2": "#ff7f0e", "reference_line": "#2ca02c"}
    theme_mod.PLOTLY_AXIS_TICK_FONT_SIZE = 12
    theme_mod.PLOTLY_BASE_FONT_SIZE = 14
    theme_mod.PLOTLY_FONT_FAMILY = "test-font"
    theme_mod.PLOTLY_LABEL_FONT_SIZE = 16
    theme_mod.PLOTLY_HEADER_TITLE_Y = 0.98
    theme_mod.PLOTLY_HEADER_PLOT_TOP = 1
    theme_mod.PLOTLY_HEADER_MARGIN_TOP = 20
    theme_mod.PLOTLY_HEADER_MARGIN_LEFT = 90
    theme_mod.PLOTLY_HEADER_MARGIN_RIGHT = 90
    theme_mod.PLOTLY_REFERENCE_LINE_WIDTH = 1
    _install_module("theme_config", theme_mod, replacements)

    go_mod = types.ModuleType("plotly.graph_objects")
    go_mod.Figure = _FakeFigure
    go_mod.Bar = lambda *args, **kwargs: ("Bar", args, kwargs)
    go_mod.Scatter = lambda *args, **kwargs: ("Scatter", args, kwargs)
    go_mod.Scatterpolar = lambda *args, **kwargs: ("Scatterpolar", args, kwargs)
    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod
    _install_module("plotly", plotly_mod, replacements)
    _install_module("plotly.graph_objects", go_mod, replacements)

    sklearn_mod = types.ModuleType("sklearn")
    sklearn_preprocessing = types.ModuleType("sklearn.preprocessing")
    sklearn_preprocessing.StandardScaler = object
    sklearn_preprocessing.OneHotEncoder = object
    sklearn_impute = types.ModuleType("sklearn.impute")
    sklearn_impute.SimpleImputer = object
    sklearn_compose = types.ModuleType("sklearn.compose")
    sklearn_compose.ColumnTransformer = object
    sklearn_compose.TransformedTargetRegressor = object
    sklearn_pipeline = types.ModuleType("sklearn.pipeline")
    sklearn_pipeline.Pipeline = object
    sklearn_linear = types.ModuleType("sklearn.linear_model")
    sklearn_linear.LinearRegression = object
    sklearn_linear.RidgeCV = object
    sklearn_linear.ElasticNetCV = object
    sklearn_linear.LogisticRegressionCV = object
    for name, module in [
        ("sklearn", sklearn_mod),
        ("sklearn.preprocessing", sklearn_preprocessing),
        ("sklearn.impute", sklearn_impute),
        ("sklearn.compose", sklearn_compose),
        ("sklearn.pipeline", sklearn_pipeline),
        ("sklearn.linear_model", sklearn_linear),
        ("joblib", types.ModuleType("joblib")),
    ]:
        _install_module(name, module, replacements)

    spec = importlib.util.spec_from_file_location("match_under_test", APP_DIR / "match.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    old_path = list(sys.path)
    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(SRC_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        _restore_modules(replacements)
    return module


class SummaryTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_mod = _load_dashboard_app_module()

    def test_analysis_tabs_default_to_summary_first(self):
        tabs = self.app_mod._analysis_tabs_component()

        self.assertEqual("summary", self.app_mod._analysis_tab_default_value(tabs))
        self.assertEqual(["Summary", "Play-by-Play", "Scorecard", "Total Impact"], self.app_mod._analysis_tab_labels(tabs))
        self.assertNotIn("Batting Stats", self.app_mod._analysis_tab_labels(tabs))
        self.assertFalse(hasattr(tabs, "persistence"))

    def test_team_analysis_route_imports(self):
        nav, page = self.app_mod.render_page("/team-analysis")

        self.assertIsNotNone(nav)
        self.assertIsNotNone(page)

    def test_dominance_index_uses_role_batting_and_non_phase_bowling_percentiles(self):
        metrics = pd.DataFrame(
            [
                {"metric_key": "pp_batting", "percentile": 0.0},
                {"metric_key": "middle_batting", "percentile": 0.0},
                {"metric_key": "death_batting", "percentile": 0.0},
                {"metric_key": "top_order_batting", "percentile": 100.0},
                {"metric_key": "middle_order_batting", "percentile": 80.0},
                {"metric_key": "finisher_batting", "percentile": 60.0},
                {"metric_key": "pp_bowling", "percentile": 0.0},
                {"metric_key": "middle_bowling", "percentile": 0.0},
                {"metric_key": "death_bowling", "percentile": 0.0},
                {"metric_key": "wicket_taking_bowling", "percentile": 40.0},
                {"metric_key": "bowling_pressure", "percentile": 20.0},
            ]
        )

        self.assertAlmostEqual(55.0, self.app_mod._dominance_index(metrics))

    def test_season_leader_cards_use_historical_team_season_percentile_benchmark(self):
        metric_keys = [
            "pp_batting",
            "middle_batting",
            "death_batting",
            "top_order_batting",
            "middle_order_batting",
            "finisher_batting",
            "pp_bowling",
            "middle_bowling",
            "death_bowling",
            "wicket_taking_bowling",
            "bowling_pressure",
        ]
        selected_metrics = pd.DataFrame(
            [
                {
                    "team": "Team A",
                    "metric_key": metric_key,
                    "raw_value": 40.0 if "pp_" in metric_key or "death_" in metric_key else 20.0,
                }
                for metric_key in metric_keys
            ]
        )
        benchmark_metrics = pd.DataFrame(
            [
                {"team": f"Benchmark {idx}", "metric_key": metric_key, "raw_value": value}
                for metric_key in metric_keys
                for idx, value in enumerate([10.0, 20.0, 30.0, 40.0], start=1)
            ]
        )
        leaderboard = types.SimpleNamespace(
            batter_stats=pd.DataFrame(),
            bowler_stats=pd.DataFrame(),
            player_impact_stats=pd.DataFrame(),
            matches=pd.DataFrame([{"match_id": 1}]),
            _impact_balls=pd.DataFrame([{"match_id": 1}]),
            team_ranking=pd.DataFrame([{"Team": "Team A"}]),
            season=2026,
        )
        captured = []

        original_season_metrics = self.app_mod._team_identity_season_metrics
        original_benchmarks = self.app_mod._load_team_identity_benchmarks
        original_cache_token = self.app_mod._team_identity_benchmark_cache_token
        original_leader_item = self.app_mod._team_analysis_leader_item
        self.app_mod._team_identity_season_metrics = lambda _leaderboard: selected_metrics
        self.app_mod._load_team_identity_benchmarks = lambda *_args, **_kwargs: benchmark_metrics
        self.app_mod._team_identity_benchmark_cache_token = lambda: 0
        self.app_mod._team_analysis_leader_item = lambda label, player_name, value_text, **kwargs: captured.append(
            {"label": label, "player_name": player_name, "value_text": value_text, **kwargs}
        ) or _Component()
        try:
            self.app_mod._build_season_leader_cards(leaderboard)
        finally:
            self.app_mod._team_identity_season_metrics = original_season_metrics
            self.app_mod._load_team_identity_benchmarks = original_benchmarks
            self.app_mod._team_identity_benchmark_cache_token = original_cache_token
            self.app_mod._team_analysis_leader_item = original_leader_item

        dominance_card = next(item for item in captured if item["label"] == "Most Complete Team")
        self.assertEqual("Team A", dominance_card["player_name"])
        self.assertEqual("50.00", dominance_card["value_text"])
        self.assertEqual("Averaged Batting and Bowling Strength", dominance_card["subtitle"])
        self.assertEqual("info", dominance_card["card_color"])
        self.assertTrue(dominance_card["inverse"])

    def test_most_impactful_player_summary_formats_batting_and_bowling(self):
        match = types.SimpleNamespace(
            impact = pd.DataFrame(
                [
                    {"Player": "Hardik Pandya", "Team": "Mumbai Indians", "Total Impact": 45.34},
                    {"Player": "Other Player", "Team": "Mumbai Indians", "Total Impact": 12.0},
                ]
            ),
            bat = pd.DataFrame(
                [
                    {"Batter": "Hardik Pandya", "Runs": 34, "Balls": 15, "Status": "not out"},
                ]
            ),
            bowl = pd.DataFrame(
                [
                    {"Bowler": "Hardik Pandya", "Overs": "4.0", "Runs": 19, "Wickets": 2},
                ]
            ),
        )

        result = self.app_mod._most_impactful_player_summary(match)

        self.assertEqual("Hardik Pandya", result["player"])
        self.assertEqual("Mumbai Indians", result["team"])
        self.assertEqual("34* (15)", result["batting"])
        self.assertEqual("2/19", result["bowling"])
        self.assertEqual("34* (15) & 2/19, Total Impact: 45.34", result["summary"])

    def test_player_summary_fallbacks(self):
        match = types.SimpleNamespace(
            bat = pd.DataFrame(columns = ["Batter", "Runs", "Balls", "Status"]),
            bowl = pd.DataFrame(columns = ["Bowler", "Overs", "Runs", "Wickets"]),
        )

        self.assertEqual("DNB", self.app_mod._format_player_batting_summary(match, "Missing Player"))
        self.assertEqual("Did not bowl", self.app_mod._format_player_bowling_summary(match, "Missing Player"))

    def test_summary_tab_handles_missing_second_innings(self):
        class FakeMatch:
            impact = pd.DataFrame(columns = ["Player", "Team", "Total Impact"])

            def most_impactful_over(self):
                return None

            def innings_phase_summary(self, innings):
                return pd.DataFrame(
                    [
                        {"Phase": "Powerplay", "Runs": 0, "Wickets": 0, "RR": np.nan},
                        {"Phase": "Middle Overs", "Runs": 0, "Wickets": 0, "RR": np.nan},
                        {"Phase": "Death Overs", "Runs": 0, "Wickets": 0, "RR": np.nan},
                    ]
                )

            def summary_runs_by_over_graph(self, innings):
                return _FakeFigure()

            def projected_score_by_over_graph(self):
                return _FakeFigure()

            def chase_run_rate_by_over_graph(self):
                return _FakeFigure()

        component = self.app_mod._build_summary_tab(FakeMatch())

        self.assertIsNotNone(component)

    def test_dashboard_shell_uses_smoothed_win_probability_graph(self):
        class FakeMatch:
            status = "live"
            status_detail = None
            match_won_by = None
            match_info_row = None

            def __init__(self):
                self.smooth_called = False
                self.balls = pd.DataFrame(
                    [
                        {
                            "innings": 1,
                            "team_balls": 6,
                            "over": 0,
                            "ball": 6,
                            "id": 1,
                            "team_runs": 10,
                            "team_wicket": 0,
                            "y_prob": 0.42,
                            "wickets_remaining": 10,
                            "balls_remaining": 114,
                        }
                    ]
                )

            def match_summary(self):
                return {
                    "team1": "Team A",
                    "team2": "Team B",
                    "innings1": {"runs": 10, "wickets": 0, "overs": "1.0", "balls": 6},
                    "innings2": {"runs": 0, "wickets": 0, "overs": "0.0", "balls": 0},
                    "result_text": "Team A batting",
                    "current_innings": 1,
                    "date": "2026-04-30",
                    "scheduled_start_ts": None,
                    "stage": "League",
                    "event_match_no": "1",
                    "playoff_match": 0,
                    "venue": "Test Venue",
                    "toss_winner": None,
                    "toss_decision": None,
                }

            def predict(self):
                raise AssertionError("raw win probability graph should not be used")

            def predict_smooth(self):
                self.smooth_called = True
                return _FakeFigure()

            def worm(self):
                return _FakeFigure()

            def _second_innings_terminal_probability(self):
                return None

            def _target(self, first_innings_runs):
                return int(first_innings_runs) + 1

            def _innings_ball_limit(self, innings):
                return 120

        match = FakeMatch()

        component = self.app_mod._build_dashboard_shell(match)

        self.assertIsNotNone(component)
        self.assertTrue(match.smooth_called)


class MatchSummaryOverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.match_mod = _load_match_module()

    def _sample_match(self):
        match = self.match_mod.Match.__new__(self.match_mod.Match)
        match.batting_team = "Team A"
        match.bowling_team = "Team B"
        match._exclude_placeholder_rows = lambda frame: frame
        match._target = lambda first_innings_runs: int(first_innings_runs) + 1
        match._innings_ball_limit = lambda innings: 120
        match.balls = pd.DataFrame(
            [
                {
                    "innings": 1,
                    "over": 0,
                    "ball": 6,
                    "team_balls": 6,
                    "team_runs": 8,
                    "team_wicket": 0,
                    "runs_total": 8,
                    "wicket_taken": 0,
                    "valid_ball": 6,
                    "y_prob": 0.55,
                    "prob_bowling_diff": 5.0,
                    "balls_remaining": 114,
                    "wickets_remaining": 10,
                    "batting_team": "Team A",
                    "id": 1,
                },
                {
                    "innings": 1,
                    "over": 1,
                    "ball": 6,
                    "team_balls": 12,
                    "team_runs": 20,
                    "team_wicket": 1,
                    "runs_total": 12,
                    "wicket_taken": 1,
                    "valid_ball": 6,
                    "y_prob": 0.35,
                    "prob_bowling_diff": -20.0,
                    "balls_remaining": 108,
                    "wickets_remaining": 9,
                    "batting_team": "Team A",
                    "id": 2,
                },
                {
                    "innings": 2,
                    "over": 0,
                    "ball": 6,
                    "team_balls": 6,
                    "team_runs": 10,
                    "team_wicket": 0,
                    "runs_total": 10,
                    "wicket_taken": 0,
                    "valid_ball": 6,
                    "y_prob": 0.50,
                    "prob_bowling_diff": 15.0,
                    "balls_remaining": 114,
                    "wickets_remaining": 10,
                    "batting_team": "Team B",
                    "id": 3,
                },
                {
                    "innings": 2,
                    "over": 1,
                    "ball": 6,
                    "team_balls": 12,
                    "team_runs": 35,
                    "team_wicket": 1,
                    "runs_total": 25,
                    "wicket_taken": 1,
                    "valid_ball": 6,
                    "y_prob": 0.68,
                    "prob_bowling_diff": 18.0,
                    "balls_remaining": 108,
                    "wickets_remaining": 9,
                    "batting_team": "Team B",
                    "id": 4,
                },
            ]
        )
        return match

    def test_most_impactful_over_uses_win_probability_swing(self):
        match = self._sample_match()

        over = match.most_impactful_over()

        self.assertEqual(1, over["innings"])
        self.assertEqual(2, over["over_display"])
        self.assertAlmostEqual(0.20, over["win_prob_swing_abs"])
        self.assertEqual("Team A", over["beneficiary_team"])

    def test_predict_smooth_plots_smoothed_line_but_tooltip_uses_raw_probability(self):
        match = self._sample_match()
        match.status = "live"
        match.balls = match.balls.assign(player_out = np.nan)

        fig = match.predict_smooth(n = 2)

        team2_line_trace = fig.traces[3]
        trace_kwargs = team2_line_trace[2]
        plotted_y = pd.Series(trace_kwargs["y"]).reset_index(drop = True)
        customdata = trace_kwargs["customdata"]

        self.assertAlmostEqual(45.0, float(plotted_y.iloc[1]))
        self.assertEqual("Team A", customdata[1][0])
        self.assertAlmostEqual(65.0, float(customdata[1][1]))

    def test_phase_summary_uses_over_level_legal_balls(self):
        match = self._sample_match()

        phases = match.innings_phase_summary(innings = 1)
        powerplay = phases[phases["Phase"] == "Powerplay"].iloc[0]

        self.assertEqual(20, powerplay["Runs"])
        self.assertEqual(1, powerplay["Wickets"])
        self.assertAlmostEqual(10.0, powerplay["RR"])
        self.assertEqual("Team A", powerplay["Team"])
        self.assertAlmostEqual(15.0, powerplay["Net Batting Impact"])

    def test_second_innings_phase_summary_impact_is_for_chasing_team(self):
        match = self._sample_match()

        phases = match.innings_phase_summary(innings = 2)
        powerplay = phases[phases["Phase"] == "Powerplay"].iloc[0]

        self.assertEqual("Team B", powerplay["Team"])
        self.assertAlmostEqual(33.0, powerplay["Net Batting Impact"])

    def test_play_by_play_feed_includes_batting_win_probability_delta(self):
        match = self._sample_match()

        feed = match.play_by_play_feed()
        first_innings_second_over = [
            over for over in feed if over["innings"] == 1 and over["over_display"] == 2
        ][0]
        second_innings_first_over = [
            over for over in feed if over["innings"] == 2 and over["over_display"] == 1
        ][0]
        second_innings_second_over = [
            over for over in feed if over["innings"] == 2 and over["over_display"] == 2
        ][0]

        self.assertAlmostEqual(20.0, first_innings_second_over["batting_win_prob_delta"])
        self.assertAlmostEqual(15.0, second_innings_first_over["batting_win_prob_delta"])
        self.assertAlmostEqual(18.0, second_innings_second_over["batting_win_prob_delta"])
        self.assertEqual(8.0, first_innings_second_over["previous_projected_score"])
        self.assertEqual(20, first_innings_second_over["projected_score"])
        self.assertEqual(12.0, first_innings_second_over["projected_score_delta"])
        self.assertAlmostEqual(11 / 19, second_innings_second_over["previous_required_rr"])
        self.assertEqual(0.0, second_innings_second_over["required_rr"])
        self.assertAlmostEqual(-(11 / 19), second_innings_second_over["required_rr_delta"])
        self.assertAlmostEqual(65.0, first_innings_second_over["team1_win_probability"])
        self.assertAlmostEqual(35.0, first_innings_second_over["team2_win_probability"])

    def test_play_by_play_over_end_probability_uses_last_ball_in_over(self):
        match = self._sample_match()
        match.balls = pd.DataFrame(
            [
                {
                    "innings": 1,
                    "over": 7.1,
                    "ball": 1,
                    "team_balls": 43,
                    "team_runs": 40,
                    "team_wicket": 1,
                    "runs_total": 1,
                    "wicket_taken": 0,
                    "valid_ball": 1,
                    "y_prob": 0.55,
                    "prob_bowling_diff": -1.0,
                    "balls_remaining": 77,
                    "wickets_remaining": 9,
                    "batting_team": "Team A",
                    "overs": 20,
                    "id": 1,
                },
                {
                    "innings": 1,
                    "over": 7.6,
                    "ball": 6,
                    "team_balls": 48,
                    "team_runs": 48,
                    "team_wicket": 1,
                    "runs_total": 4,
                    "wicket_taken": 0,
                    "valid_ball": 1,
                    "y_prob": 0.60,
                    "prob_bowling_diff": -2.0,
                    "balls_remaining": 72,
                    "wickets_remaining": 9,
                    "batting_team": "Team A",
                    "overs": 20,
                    "id": 2,
                },
                {
                    "innings": 1,
                    "over": 8.1,
                    "ball": 1,
                    "team_balls": 49,
                    "team_runs": 49,
                    "team_wicket": 1,
                    "runs_total": 1,
                    "wicket_taken": 0,
                    "valid_ball": 1,
                    "y_prob": 0.20,
                    "prob_bowling_diff": -50.0,
                    "balls_remaining": 71,
                    "wickets_remaining": 9,
                    "batting_team": "Team A",
                    "overs": 20,
                    "id": 3,
                },
            ]
        )

        feed = match.play_by_play_feed()
        over_eight = [over for over in feed if over["innings"] == 1 and over["over_display"] == 8][0]
        over_table = match.summary_over_table()
        table_over_eight = over_table[over_table["over_display"] == 8].iloc[0]

        self.assertAlmostEqual(0.60, over_eight["y_prob"])
        self.assertAlmostEqual(40.0, over_eight["team1_win_probability"])
        self.assertAlmostEqual(60.0, over_eight["team2_win_probability"])
        self.assertAlmostEqual(3.0, over_eight["batting_win_prob_delta"])
        self.assertEqual("48/1", over_eight["score_text"])
        self.assertAlmostEqual(0.60, table_over_eight["y_prob"])
        self.assertAlmostEqual(-0.03, table_over_eight["win_prob_swing"])

    def test_first_innings_first_over_projection_delta_uses_resource_baseline(self):
        match = self._sample_match()
        match.balls = pd.DataFrame(
            [
                {
                    "innings": 1,
                    "over": 0,
                    "ball": 6,
                    "team_balls": 6,
                    "team_runs": 8,
                    "team_wicket": 0,
                    "runs_total": 8,
                    "wicket_taken": 0,
                    "valid_ball": 6,
                    "y_prob": 0.55,
                    "prob_bowling_diff": 5.0,
                    "balls_remaining": 6,
                    "wickets_remaining": 10,
                    "batting_team": "Team A",
                    "overs": 2,
                    "balls_per_over": 6,
                    "id": 1,
                },
            ]
        )
        old_resource_function = self.match_mod.ipl.resource_function
        self.match_mod.ipl.resource_function = lambda frame, params: pd.to_numeric(frame["balls_remaining"], errors="coerce") / 6
        try:
            first_over = match.play_by_play_feed()[0]
        finally:
            self.match_mod.ipl.resource_function = old_resource_function

        self.assertEqual(2.0, first_over["previous_projected_score"])
        self.assertEqual(9, first_over["projected_score"])
        self.assertEqual(7.0, first_over["projected_score_delta"])

    def test_play_by_play_feed_includes_super_over_without_projection_metrics(self):
        match = self._sample_match()
        match.balls = pd.concat(
            [
                match.balls,
                pd.DataFrame(
                    [
                        {
                            "innings": 3,
                            "over": 0,
                            "ball": 1,
                            "team_balls": 1,
                            "team_runs": 1,
                            "team_wicket": 0,
                            "runs_total": 1,
                            "wicket_taken": 0,
                            "valid_ball": 1,
                            "y_prob": np.nan,
                            "prob_bowling_diff": 0.0,
                            "balls_remaining": 5,
                            "wickets_remaining": 1,
                            "batting_team": "Team A",
                            "id": 5,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        feed = match.play_by_play_feed()
        super_over = [over for over in feed if over["innings"] == 3 and over["over_display"] == 1][0]

        self.assertEqual("Super Over", super_over["status_text"])
        self.assertEqual("-", super_over["required_run_rate_text"])
        self.assertTrue(pd.isna(super_over["required_rr"]))
        self.assertTrue(pd.isna(super_over["projected_score"]))


if __name__ == "__main__":
    unittest.main()
