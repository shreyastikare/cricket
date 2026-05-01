import ipl

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import os
import random

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, LogisticRegressionCV

import sqlite3
import joblib

import plotly.graph_objects as go

try:
    from ipl.storage.win_probability_cache import load_cached_prediction_impact
except ImportError:
    load_cached_prediction_impact = None

try:
    from plot_theme import apply_plot_theme as _apply_plot_theme, header_legend_layout as _header_legend_layout, innings_color as _innings_color
    from theme_config import (
        PLOTLY_COLORS,
        PLOTLY_AXIS_TICK_FONT_SIZE,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_FONT_FAMILY,
        PLOTLY_LABEL_FONT_SIZE,
        PLOTLY_HEADER_TITLE_Y,
        PLOTLY_HEADER_PLOT_TOP,
        PLOTLY_HEADER_MARGIN_TOP,
        PLOTLY_HEADER_MARGIN_LEFT,
        PLOTLY_HEADER_MARGIN_RIGHT,
        PLOTLY_REFERENCE_LINE_WIDTH,
    )
except ModuleNotFoundError:
    from app.plot_theme import apply_plot_theme as _apply_plot_theme, header_legend_layout as _header_legend_layout, innings_color as _innings_color
    from app.theme_config import (
        PLOTLY_COLORS,
        PLOTLY_AXIS_TICK_FONT_SIZE,
        PLOTLY_BASE_FONT_SIZE,
        PLOTLY_FONT_FAMILY,
        PLOTLY_LABEL_FONT_SIZE,
        PLOTLY_HEADER_TITLE_Y,
        PLOTLY_HEADER_PLOT_TOP,
        PLOTLY_HEADER_MARGIN_TOP,
        PLOTLY_HEADER_MARGIN_LEFT,
        PLOTLY_HEADER_MARGIN_RIGHT,
        PLOTLY_REFERENCE_LINE_WIDTH,
    )


def _format_fielders(value):
    if pd.isna(value):
        return ''
    fielders = [x.strip() for x in str(value).split(',') if x.strip()]
    return ', '.join(fielders)


def _dismissal_summary(wicket_kind, bowler, fielders):
    kind = '' if pd.isna(wicket_kind) else str(wicket_kind).strip().lower()
    bowler_name = '' if pd.isna(bowler) else str(bowler).strip()
    fielders_text = _format_fielders(fielders)

    if kind == 'bowled':
        return f'b {bowler_name}' if bowler_name else 'bowled'
    if kind == 'caught':
        if bowler_name and fielders_text:
            return f'b {bowler_name} c {fielders_text}'
        if bowler_name:
            return f'b {bowler_name} c'
        if fielders_text:
            return f'c {fielders_text}'
        return 'caught'
    if kind == 'caught and bowled':
        return f'b {bowler_name} c {bowler_name}' if bowler_name else 'caught and bowled'
    if kind == 'lbw':
        return f'lbw b {bowler_name}' if bowler_name else 'lbw'
    if kind == 'stumped':
        if bowler_name and fielders_text:
            return f'b {bowler_name} st {fielders_text}'
        if bowler_name:
            return f'b {bowler_name} st'
        if fielders_text:
            return f'st {fielders_text}'
        return 'stumped'
    if kind == 'run out':
        return f'run out ({fielders_text})' if fielders_text else 'run out'
    if kind == 'hit wicket':
        return f'hit wicket b {bowler_name}' if bowler_name else 'hit wicket'
    if kind in ['obstructing the field', 'retired hurt', 'retired out']:
        return kind
    if kind and bowler_name:
        return f'{kind} b {bowler_name}'
    if kind:
        return kind
    return 'out'


DB_PATH = os.getenv('DB_PATH', 'data/sqlite/ipl.db')
TERMINAL_MATCH_STATUSES = {'complete', 'abandoned', 'no_result'}

resource_params = ipl.load_resource_params()
first_innings_glm = ipl.first_innings_glm_load()
second_innings_glm = ipl.second_innings_glm_load()


class Match:
    def __init__(self, id):
        self.match_id = id
        self.matches_with_prior = self.prior_match_stats()
        self._identity_lookup_loaded = False
        self._alias_to_canonical = {}
        self._alias_lower_to_canonical = {}
        self._player_id_to_canonical = {}
        self._espn_id_to_canonical = {}
        self.scorecard_consistency = {"has_issue": False, "innings": {}, "message": None}
        self.match_info()
        self.refresh_match()
        self.update_scorecard()
        
        
        return
    
    def _load_identity_lookup(self):
        if self._identity_lookup_loaded:
            return

        alias_to_canonical = {}
        alias_lower_to_canonical = {}
        player_id_to_canonical = {}
        espn_id_to_canonical = {}

        with sqlite3.connect(DB_PATH) as conn:
            player_rows = conn.execute(
                '''
                SELECT player_id, espn_athlete_id, canonical_name
                FROM player_master
                WHERE canonical_name IS NOT NULL
                  AND TRIM(canonical_name) <> ''
                '''
            ).fetchall()
            for player_id, espn_id, canonical_name in player_rows:
                canonical = str(canonical_name).strip()
                if not canonical:
                    continue
                player_id_to_canonical[str(player_id).strip()] = canonical
                if espn_id is not None and str(espn_id).strip():
                    espn_id_to_canonical[str(espn_id).strip()] = canonical
                alias_to_canonical[canonical] = canonical
                alias_lower_to_canonical[canonical.lower()] = canonical

            alias_rows = conn.execute(
                '''
                SELECT pa.alias_name, pm.canonical_name
                FROM player_alias pa
                JOIN player_master pm
                  ON pm.player_id = pa.player_id
                WHERE pa.alias_name IS NOT NULL
                  AND TRIM(pa.alias_name) <> ''
                  AND pm.canonical_name IS NOT NULL
                  AND TRIM(pm.canonical_name) <> ''
                '''
            ).fetchall()
            for alias_name, canonical_name in alias_rows:
                alias = str(alias_name).strip()
                canonical = str(canonical_name).strip()
                if not alias or not canonical:
                    continue
                alias_to_canonical[alias] = canonical
                alias_lower_to_canonical[alias.lower()] = canonical

        self._alias_to_canonical = alias_to_canonical
        self._alias_lower_to_canonical = alias_lower_to_canonical
        self._player_id_to_canonical = player_id_to_canonical
        self._espn_id_to_canonical = espn_id_to_canonical
        self._identity_lookup_loaded = True

    def _canonical_name_from_alias(self, value):
        if pd.isna(value):
            return ''
        name = str(value).strip()
        if not name:
            return ''
        if name in self._alias_to_canonical:
            return self._alias_to_canonical[name]
        return self._alias_lower_to_canonical.get(name.lower(), name)

    def _parse_multi_tokens(self, value):
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text:
            return []
        for sep in ['|', ';', '/', '&']:
            text = text.replace(sep, ',')
        text = text.replace('[', '').replace(']', '')
        return [x.strip() for x in text.split(',') if x.strip()]

    def _canonicalize_fielders(self, fielders, fielder_player_ids = None, fielder_espn_ids = None):
        self._load_identity_lookup()

        names = self._parse_multi_tokens(fielders)
        if not names:
            return ''

        player_id_tokens = self._parse_multi_tokens(fielder_player_ids)
        espn_id_tokens = self._parse_multi_tokens(fielder_espn_ids)

        display_names = []
        for i, raw_name in enumerate(names):
            canonical = None

            if i < len(player_id_tokens):
                canonical = self._player_id_to_canonical.get(str(player_id_tokens[i]).strip())
            if canonical is None and i < len(espn_id_tokens):
                canonical = self._espn_id_to_canonical.get(str(espn_id_tokens[i]).strip())
            if canonical is None:
                canonical = self._canonical_name_from_alias(raw_name)

            display_names.append(canonical if canonical else raw_name)

        return ', '.join(display_names)
    
    
    def match_info(self):
        with sqlite3.connect(DB_PATH) as conn:
            query = '''
                    SELECT * FROM match_list
                    WHERE match_id = ?
                '''
            match_info = pd.read_sql_query(query, conn, params = [self.match_id])
        if match_info.empty:
            raise ValueError(f'match_id={self.match_id} not found in match_list')

        row = match_info.iloc[0]
        self.match_info_row = row
        self.batting_team = row.get('bat_first')
        self.bowling_team = row.get('bowl_first')
        self.date = row.get('date')
        self.venue = row.get('venue')
        self.city = row.get('city')
        self.match_won_by = row.get('match_won_by')
        self.win_outcome = row.get('win_outcome')
        self.toss_winner = row.get('toss_winner')
        self.toss_decision = row.get('toss_decision')
        self.superover_winner = row.get('superover_winner')
        self.result_type = row.get('result_type')
        self.method = row.get('method')
        self.status = row.get('status')
        self.status_detail = row.get('status_detail')
        self.stage = row.get('stage')
        self.event_match_no = row.get('event_match_no')
        playoff_value = row.get('playoff_match')
        playoff_num = pd.to_numeric(pd.Series([playoff_value]), errors = 'coerce').iloc[0]
        if pd.notna(playoff_num):
            self.playoff_match = int(playoff_num)
        else:
            stage_value = '' if pd.isna(self.stage) else str(self.stage).strip().lower()
            group_like = {'', 'unknown', 'group stage', 'league stage', 'regular'}
            self.playoff_match = 0 if stage_value in group_like else 1

        needs_schedule_fallback = any(
            (
                pd.isna(self.venue) or str(self.venue).strip() == '',
                pd.isna(self.city) or str(self.city).strip() == '',
                pd.isna(self.event_match_no) or str(self.event_match_no).strip() == '',
                pd.isna(self.stage) or str(self.stage).strip().lower() in {'', 'unknown'},
                pd.isna(self.status_detail) or str(self.status_detail).strip() == '',
            )
        )
        if needs_schedule_fallback:
            with sqlite3.connect(DB_PATH) as conn:
                try:
                    sched = pd.read_sql_query(
                        '''
                            SELECT venue, city, event_match_no, stage, status_detail
                            FROM match_schedule
                            WHERE match_id = ?
                            LIMIT 1
                        ''',
                        conn,
                        params = [self.match_id]
                    )
                except Exception:
                    sched = pd.read_sql_query(
                        '''
                            SELECT venue, city, status_detail
                            FROM match_schedule
                            WHERE match_id = ?
                            LIMIT 1
                        ''',
                        conn,
                        params = [self.match_id]
                    )
            if not sched.empty:
                sched_row = sched.iloc[0]
                if pd.isna(self.venue) or str(self.venue).strip() == '':
                    self.venue = sched_row.get('venue')
                if pd.isna(self.city) or str(self.city).strip() == '':
                    self.city = sched_row.get('city')
                if pd.isna(self.event_match_no) or str(self.event_match_no).strip() == '':
                    self.event_match_no = sched_row.get('event_match_no')
                if pd.isna(self.stage) or str(self.stage).strip().lower() in {'', 'unknown'}:
                    self.stage = sched_row.get('stage')
                if pd.isna(self.status_detail) or str(self.status_detail).strip() == '':
                    self.status_detail = sched_row.get('status_detail')
        self.scheduled_start_ts = None
        with sqlite3.connect(DB_PATH) as conn:
            sched_ts_row = conn.execute(
                '''
                    SELECT scheduled_start_ts
                    FROM match_schedule
                    WHERE match_id = ?
                    LIMIT 1
                ''',
                [self.match_id]
            ).fetchone()
        if sched_ts_row is not None and sched_ts_row[0] is not None:
            ts_text = str(sched_ts_row[0]).strip()
            self.scheduled_start_ts = ts_text if ts_text else None
        if self.playoff_match is None:
            stage_value = '' if pd.isna(self.stage) else str(self.stage).strip().lower()
            group_like = {'', 'unknown', 'group stage', 'league stage', 'regular'}
            self.playoff_match = 0 if stage_value in group_like else 1
    
    
    def prior_match_stats(self):
        with sqlite3.connect(DB_PATH) as conn:
            query = 'SELECT * FROM match_list'
            match_list = pd.read_sql_query(query, conn)
        matches_with_prior = ipl.prior_match_stats(match_list)
        return matches_with_prior
    
    def _second_innings_terminal_probability(self):
        Xi = self.balls.loc[self.balls['innings'] == 2].copy()
        if Xi.empty:
            return None

        sort_cols = [col for col in ['team_balls', 'over', 'ball', 'id'] if col in Xi.columns]
        if sort_cols:
            Xi = Xi.sort_values(sort_cols)
        last_idx = Xi.index[-1]
        last = Xi.iloc[-1]

        second_runs = pd.to_numeric(pd.Series([last.get('team_runs')]), errors = 'coerce').iloc[0]
        second_wickets = pd.to_numeric(pd.Series([last.get('team_wicket')]), errors = 'coerce').iloc[0]
        second_balls = pd.to_numeric(pd.Series([last.get('team_balls')]), errors = 'coerce').iloc[0]
        required_runs_last = pd.to_numeric(pd.Series([last.get('required_runs')]), errors = 'coerce').iloc[0]
        if pd.isna(second_runs):
            return None

        first_innings_runs = pd.to_numeric(
            self.balls.loc[self.balls['innings'] == 1, 'team_runs'],
            errors = 'coerce'
        ).dropna()
        first_end_runs = int(first_innings_runs.max()) if not first_innings_runs.empty else 0
        target = self._target(first_end_runs)
        second_ball_limit = self._innings_ball_limit(innings = 2)

        innings_over = False
        if pd.notna(required_runs_last) and float(required_runs_last) <= 0:
            innings_over = True
        if pd.notna(second_wickets) and int(second_wickets) >= 10:
            innings_over = True
        if pd.notna(second_balls) and int(second_balls) >= int(second_ball_limit):
            innings_over = True

        status_clean = str(self.status).strip().lower()
        if status_clean != 'complete':
            return None
        if not innings_over:
            return None

        second_runs_int = int(second_runs)
        target_int = int(target)
        if second_runs_int == target_int - 1:
            return last_idx, 0.5

        batting_team = str(self._match_team_name(self.batting_team)).strip().lower()
        bowling_team = str(self._match_team_name(self.bowling_team)).strip().lower()
        winner = '' if pd.isna(self.match_won_by) else str(self.match_won_by).strip().lower()
        if winner == bowling_team:
            return last_idx, 1.0
        if winner == batting_team:
            return last_idx, 0.0

        if second_runs_int >= target_int:
            return last_idx, 1.0
        return last_idx, 0.0
    
    
    def refresh_match(self):
        with sqlite3.connect(DB_PATH) as conn:
            query = '''
                SELECT * FROM ball_by_ball
                WHERE match_id = ?
            '''
            self.balls = pd.read_sql_query(query, conn, params = [self.match_id])

        cached_balls = None
        if (
            load_cached_prediction_impact is not None
            and str(self.status).strip().lower() == 'complete'
        ):
            cached_balls = load_cached_prediction_impact(
                self.balls,
                db_path = DB_PATH,
                match_id = int(self.match_id),
            )

        if cached_balls is not None:
            self.balls = cached_balls
        else:
            X_first = ipl.preprocess_first_innings(self.balls, self.matches_with_prior).copy()
            X_second = ipl.preprocess_second_innings(self.balls, self.matches_with_prior).copy()

            self.balls['y_prob'] = np.nan
            mask_1 = self.balls['innings'] == 1
            required_runs = pd.to_numeric(self.balls.get('required_runs'), errors = 'coerce')
            mask_2 = (self.balls['innings'] == 2) & (required_runs > 0)
            team_balls_num = pd.to_numeric(self.balls.get('team_balls'), errors = 'coerce')
            first_team_balls = set(
                team_balls_num.loc[mask_1 & team_balls_num.notna()]
                .round()
                .astype(int)
                .tolist()
            )
            mask_2_model = mask_2 & team_balls_num.round().astype('Int64').isin(first_team_balls)
            mask_2_chased = (self.balls['innings'] == 2) & (required_runs <= 0)

            def _assign_probs(mask, y_prob):
                idx = self.balls.index[mask]
                if len(idx) == len(y_prob):
                    self.balls.loc[idx, 'y_prob'] = y_prob
                    return

                n = min(len(idx), len(y_prob))
                if n > 0:
                    self.balls.loc[idx[:n], 'y_prob'] = y_prob[:n]

            if not X_first.empty:
                y_p1 = first_innings_glm.predict_proba(X_first)[:, 1]
                _assign_probs(mask_1, y_p1)
            if not X_second.empty:
                y_p2 = second_innings_glm.predict_proba(X_second)[:, 1]
                _assign_probs(mask_2_model, y_p2)
            self.balls.loc[mask_2_chased, 'y_prob'] = 1.0

            terminal_second = self._second_innings_terminal_probability()
            if terminal_second is not None:
                terminal_idx, terminal_prob = terminal_second
                self.balls.loc[terminal_idx, 'y_prob'] = float(terminal_prob)
            self.balls['y_prob'] = self.balls.groupby(['match_id', 'innings'], sort = False)['y_prob'].ffill()
            self.balls['y_prob'] = self.balls['y_prob'].fillna(0.5)

            innings_num = pd.to_numeric(self.balls.get('innings'), errors = 'coerce')
            regular_mask = innings_num.isin([1, 2])
            regular_balls = self.balls.loc[regular_mask].copy()
            extra_balls = self.balls.loc[~regular_mask].copy()

            if X_first.empty:
                scored_regular = ipl.calculate_impact(regular_balls) if not regular_balls.empty else regular_balls
            else:
                baseline_by_match = {
                    match_id: ipl.get_baseline(match_frame, first_innings_glm)
                    for match_id, match_frame in X_first.groupby('match_id', sort = False)
                }

                class _CachedBaselineModel:
                    def predict_proba(self, frame):
                        match_id = frame['match_id'].iloc[0]
                        baseline = float(baseline_by_match.get(match_id, 0.5))
                        return np.array([[1 - baseline, baseline]])

                scored_regular = (
                    ipl.calculate_impact(regular_balls, model = _CachedBaselineModel())
                    if not regular_balls.empty
                    else regular_balls
                )

            if not extra_balls.empty:
                extra_balls['y_prob'] = pd.to_numeric(extra_balls.get('y_prob'), errors = 'coerce').fillna(0.5)
                for col in ['prob_bowling_diff', 'prob_batting_diff', 'bowler_delta', 'batter_delta']:
                    extra_balls[col] = 0.0
                self.balls = pd.concat([scored_regular, extra_balls], ignore_index = True, sort = False)
            else:
                self.balls = scored_regular
        self.impact = ipl.aggregate_impact(self.balls)
        self._summary_over_table_cache = None
        
        self.in_progress = str(self.status).strip().lower() not in TERMINAL_MATCH_STATUSES
        return

    def impact_graph(self):
        return self.total_impact_graph()


    def _match_team_name(self, value):
        return value.iloc[0] if isinstance(value, pd.Series) else value


    def _first_innings_projected_score_baseline(self, frame: pd.DataFrame | None = None):
        source = self.balls if frame is None else frame
        if source is None or source.empty:
            return np.nan

        innings = pd.to_numeric(source.get('innings'), errors = 'coerce')
        first_innings = source.loc[innings == 1].copy() if 'innings' in source.columns else source.copy()
        if first_innings.empty:
            return np.nan

        overs_limit = np.nan
        if 'overs' in first_innings.columns:
            overs_values = pd.to_numeric(first_innings['overs'], errors = 'coerce').dropna()
            overs_values = overs_values[overs_values > 0]
            if not overs_values.empty:
                overs_limit = float(overs_values.max())

        balls_per_over = 6.0
        if 'balls_per_over' in first_innings.columns:
            bpo_values = pd.to_numeric(first_innings['balls_per_over'], errors = 'coerce').dropna()
            bpo_values = bpo_values[bpo_values > 0]
            if not bpo_values.empty:
                balls_per_over = float(bpo_values.iloc[-1])

        if pd.isna(overs_limit):
            try:
                total_balls = float(self._innings_ball_limit(innings = 1))
            except Exception:
                total_balls = np.nan
        else:
            total_balls = float(overs_limit) * float(balls_per_over)

        if pd.isna(total_balls) or total_balls <= 0:
            return np.nan

        resource_df = pd.DataFrame({
            'wickets_remaining': [10],
            'balls_remaining': [float(total_balls)],
        })
        resource_val = float(ipl.resource_function(resource_df, resource_params)[0])
        return float(int(round(resource_val)))


    def _impact_bar_graph(self, impact_df, value_col, title_text, *, show_legend = False, show = False):
        batting_team = self._match_team_name(self.batting_team)
        bowling_team = self._match_team_name(self.bowling_team)

        team_colors = {
            str(batting_team): PLOTLY_COLORS['innings_1'],
            str(bowling_team): PLOTLY_COLORS['innings_2'],
        }
        fallback_color = '#9AA5B1'

        plot_df = impact_df.copy()
        plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors = 'coerce')
        plot_df = plot_df.dropna(subset = ['Player', 'Team', value_col])
        plot_df['Player'] = plot_df['Player'].astype(str)
        plot_df['Team'] = plot_df['Team'].astype(str)
        plot_df = plot_df.sort_values(value_col, ascending = False).reset_index(drop = True)
        player_order = plot_df['Player'].tolist()

        fig = go.Figure()

        team_order = [str(batting_team), str(bowling_team)]
        extra_teams = [team for team in plot_df['Team'].unique() if team not in team_order]
        ordered_teams = team_order + extra_teams

        for team_name in ordered_teams:
            team_rows = plot_df.loc[plot_df['Team'] == team_name].copy()
            if team_rows.empty:
                continue

            team_color = team_colors.get(team_name, fallback_color)
            team_legend_shown = False
            team_values = pd.to_numeric(team_rows[value_col], errors = 'coerce')
            rows_by_sign = [
                (team_rows.loc[team_values >= 0].copy(), ''),
                (team_rows.loc[team_values < 0].copy(), '/'),
            ]

            for rows, pattern_shape in rows_by_sign:
                if rows.empty:
                    continue

                row_values = pd.to_numeric(rows[value_col], errors = 'coerce')
                if pattern_shape == '':
                    bar_text = row_values.apply(
                        lambda v: f'{v:.2f}' if pd.notna(v) and v > 0 else ''
                    )
                    text_position = 'outside'
                    inside_anchor = None
                else:
                    bar_text = row_values.apply(
                        lambda v: f'{v:.2f}' if pd.notna(v) and v < 0 else ''
                    )
                    text_position = 'outside'
                    inside_anchor = None

                hover_customdata = np.column_stack([rows['Team'], rows['Player']])
                fig.add_trace(
                    go.Bar(
                        x = rows[value_col],
                        y = rows['Player'],
                        orientation = 'h',
                        name = team_name,
                        marker = dict(
                            color = team_color,
                            pattern = dict(
                                shape = pattern_shape,
                                fillmode = 'overlay',
                                fgcolor = 'rgba(255, 255, 255, 0.95)',
                                size = 8,
                                solidity = 0.25
                            )
                        ),
                        text = bar_text,
                        textposition = text_position,
                        insidetextanchor = inside_anchor,
                        textfont = dict(
                            size = PLOTLY_BASE_FONT_SIZE,
                            color = '#1F2A37',
                            family = PLOTLY_FONT_FAMILY,
                        ),
                        customdata = hover_customdata,
                        hovertemplate = (
                            'Player = %{customdata[1]}<br>'
                            'Team = %{customdata[0]}<br>'
                            f'{value_col} = %{{x:.2f}}'
                            '<extra></extra>'
                        ),
                        showlegend = show_legend and (not team_legend_shown)
                    )
                )
                team_legend_shown = True

        if plot_df.empty:
            fig.add_annotation(
                text = 'No impact values available for this view.',
                x = 0.5,
                y = 0.5,
                xref = 'paper',
                yref = 'paper',
                xanchor = 'center',
                yanchor = 'middle',
                showarrow = False,
                font = dict(
                    size = PLOTLY_LABEL_FONT_SIZE,
                    color = '#5D728A',
                    family = PLOTLY_FONT_FAMILY,
                )
            )

        layout_kwargs = dict(
            title = dict(
                text = title_text,
                x = 0.5,
                xanchor = 'center',
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = 'top'
            ),
            xaxis_title = f'<b>{value_col}</b>',
            yaxis_title = None,
            width = 1000,
            height = 600,
            margin = dict(
                t = PLOTLY_HEADER_MARGIN_TOP,
                l = PLOTLY_HEADER_MARGIN_LEFT,
                r = PLOTLY_HEADER_MARGIN_RIGHT
            ),
            barmode = 'overlay'
        )

        if show_legend:
            layout_kwargs['legend'] = _header_legend_layout()

        fig.update_layout(**layout_kwargs)
        _apply_plot_theme(fig)
        fig.update_traces(cliponaxis = False)
        xaxis_kwargs = dict(fixedrange = True)
        if (not show_legend) and (not plot_df.empty):
            values = pd.to_numeric(plot_df[value_col], errors = 'coerce').dropna()
            if not values.empty:
                x_min = min(0.0, float(values.min()))
                x_max = max(0.0, float(values.max()))
                span = x_max - x_min
                if span <= 0:
                    span = max(abs(x_min), abs(x_max), 1.0)

                left_pad = span * (0.24 if x_min < 0 else 0.06)
                right_pad = span * (0.18 if x_max > 0 else 0.06)
                xaxis_kwargs['range'] = [x_min - left_pad, x_max + right_pad]

        xaxis_kwargs['showgrid'] = False
        fig.update_xaxes(**xaxis_kwargs)
        fig.update_yaxes(
            categoryorder = 'array',
            categoryarray = player_order,
            autorange = 'reversed',
            fixedrange = True,
            domain = [0, PLOTLY_HEADER_PLOT_TOP],
            tickfont = dict(size = PLOTLY_BASE_FONT_SIZE, family = PLOTLY_FONT_FAMILY),
        )

        if show:
            fig.show(config = {
                'displayModeBar': False,
                'scrollZoom': False
            })
        return fig


    def batter_impact_graph(self, innings, show = False):
        if innings not in [1, 2]:
            raise ValueError('innings must be 1 or 2')

        batting_team = self._match_team_name(self.batting_team)
        bowling_team = self._match_team_name(self.bowling_team)
        team_name = batting_team if innings == 1 else bowling_team

        plot_df = self.impact.copy()
        plot_df = plot_df.loc[plot_df['Team'] == team_name]
        plot_df = plot_df.loc[pd.to_numeric(plot_df['Batting Impact'], errors = 'coerce') != 0]
        plot_df = plot_df.sort_values('Batting Impact', ascending = False)

        return self._impact_bar_graph(
            plot_df,
            'Batting Impact',
            f'<b>Batting Impact - {team_name}</b>',
            show_legend = False,
            show = show
        )


    def bowler_impact_graph(self, innings, show = False):
        if innings not in [1, 2]:
            raise ValueError('innings must be 1 or 2')

        batting_team = self._match_team_name(self.batting_team)
        bowling_team = self._match_team_name(self.bowling_team)
        team_name = bowling_team if innings == 1 else batting_team

        plot_df = self.impact.copy()
        plot_df = plot_df.loc[plot_df['Team'] == team_name]
        plot_df = plot_df.loc[pd.to_numeric(plot_df['Bowling Impact'], errors = 'coerce') != 0]
        plot_df = plot_df.sort_values('Bowling Impact', ascending = False)

        return self._impact_bar_graph(
            plot_df,
            'Bowling Impact',
            f'<b>Bowling Impact - {team_name}</b>',
            show_legend = False,
            show = show
        )


    def total_impact_graph(self, show = False):
        plot_df = self.impact.copy().sort_values('Total Impact', ascending = False)
        return self._impact_bar_graph(
            plot_df,
            'Total Impact',
            '<b>Total Impact Score</b>',
            show_legend = True,
            show = show
        )


    def _balls_to_overs(self, balls):
        if pd.isna(balls):
            return '0.0'
        balls = int(balls)
        return f'{balls // 6}.{balls % 6}'


    def _innings_snapshot(self, innings):
        Xi = self.balls.loc[self.balls['innings'] == innings].copy()
        if Xi.empty:
            return {'runs': 0, 'wickets': 0, 'balls': 0, 'overs': '0.0'}

        Xi = Xi.sort_values('team_balls')
        last = Xi.iloc[-1]
        runs = int(last['team_runs']) if pd.notna(last['team_runs']) else 0
        wickets = int(last['team_wicket']) if pd.notna(last['team_wicket']) else 0
        balls = int(last['team_balls']) if pd.notna(last['team_balls']) else 0
        return {
            'runs': runs,
            'wickets': wickets,
            'balls': balls,
            'overs': self._balls_to_overs(balls)
        }


    def _target(self, first_innings_runs):
        innings_two = self.balls.loc[self.balls['innings'] == 2]
        if not innings_two.empty and 'runs_target' in innings_two.columns:
            target_values = pd.to_numeric(innings_two['runs_target'], errors = 'coerce').dropna()
            target_values = target_values[target_values > 0]
            if not target_values.empty:
                return int(target_values.max())
        return int(first_innings_runs) + 1


    def _innings_ball_limit(self, innings):
        Xi = self.balls.loc[self.balls['innings'] == innings].copy()
        if Xi.empty:
            return 120

        Xi = Xi.sort_values('team_balls')
        last = Xi.iloc[-1]
        overs_limit = pd.to_numeric(pd.Series([last.get('overs')]), errors = 'coerce').iloc[0]
        balls_per_over = pd.to_numeric(pd.Series([last.get('balls_per_over')]), errors = 'coerce').iloc[0]
        if pd.isna(overs_limit) or float(overs_limit) <= 0:
            return 120

        bpo = float(balls_per_over) if pd.notna(balls_per_over) and float(balls_per_over) > 0 else 6.0
        ball_limit = int(round(float(overs_limit) * bpo))
        return ball_limit if ball_limit > 0 else 120


    def _derived_completed_result(self, first, second):
        target = self._target(first['runs'])
        second_runs = int(second['runs'])
        second_wickets = int(second['wickets'])
        second_balls = int(second['balls'])

        if second_runs >= target:
            wickets_remaining = max(0, 10 - second_wickets)
            wicket_label = 'wicket' if wickets_remaining == 1 else 'wickets'
            return str(self.bowling_team), f'{wickets_remaining} {wicket_label}'

        innings_over = second_wickets >= 10 or second_balls >= 120
        if not innings_over:
            return '', ''

        runs_margin = max(0, (target - 1) - second_runs)
        if runs_margin == 0:
            return '', 'tie'
        run_label = 'run' if runs_margin == 1 else 'runs'
        return str(self.batting_team), f'{runs_margin} {run_label}'


    def _live_result_text(self, current_innings, first, second):
        if current_innings <= 1:
            return 'Projected Score:'

        target = self._target(first['runs'])
        runs_needed = max(0, target - int(second['runs']))
        second_innings_ball_limit = self._innings_ball_limit(innings = 2)
        balls_left = max(0, second_innings_ball_limit - int(second['balls']))
        chasing_team = str(self.bowling_team)

        if runs_needed == 0:
            return f'{chasing_team} have reached the target'
        return f'{chasing_team} need {runs_needed} runs from {balls_left} balls'

    def _exclude_placeholder_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        working = df.copy()
        id_values = (
            working['id'].fillna('').astype(str).str.strip()
            if 'id' in working.columns
            else pd.Series('', index = working.index)
        )
        over_values = pd.to_numeric(working.get('over'), errors = 'coerce').fillna(0)
        ball_values = pd.to_numeric(working.get('ball'), errors = 'coerce').fillna(0)
        team_balls_values = pd.to_numeric(working.get('team_balls'), errors = 'coerce').fillna(0)
        runs_total_values = pd.to_numeric(working.get('runs_total'), errors = 'coerce').fillna(0)
        batter_values = (
            working['batter'].fillna('').astype(str).str.strip()
            if 'batter' in working.columns
            else pd.Series('', index = working.index)
        )
        bowler_values = (
            working['bowler'].fillna('').astype(str).str.strip()
            if 'bowler' in working.columns
            else pd.Series('', index = working.index)
        )

        placeholder_mask = (
            id_values.eq('999999999999999')
            | (
                over_values.eq(0)
                & ball_values.eq(0)
                & team_balls_values.le(0)
                & runs_total_values.eq(0)
                & batter_values.eq('')
                & bowler_values.eq('')
            )
        )

        if placeholder_mask.any():
            working = working.loc[~placeholder_mask].copy()
        return working

    def _play_result_token(self, row):
        wicket_kind = '' if pd.isna(row.get('wicket_kind')) else str(row.get('wicket_kind')).strip()
        runs_total = int(pd.to_numeric(pd.Series([row.get('runs_total')]), errors = 'coerce').fillna(0).iloc[0])
        runs_batter = int(pd.to_numeric(pd.Series([row.get('runs_batter')]), errors = 'coerce').fillna(0).iloc[0])
        runs_extras = int(pd.to_numeric(pd.Series([row.get('runs_extras')]), errors = 'coerce').fillna(0).iloc[0])
        extra_type = '' if pd.isna(row.get('extra_type')) else str(row.get('extra_type')).strip().lower()

        if wicket_kind:
            token = 'W' if runs_total <= 0 else f'{runs_total}W'
            return token, 'wicket'

        if 'wides' in extra_type:
            return f'{max(0, runs_extras)}w', 'default'
        if 'noballs' in extra_type:
            return f'{max(0, runs_extras)}nb', 'default'
        if 'byes' in extra_type and 'leg' not in extra_type:
            return f'{max(0, runs_extras)}b', 'default'
        if 'legbyes' in extra_type:
            return f'{max(0, runs_extras)}lb', 'default'

        token = '•' if runs_batter == 0 else str(max(0, runs_batter))
        if runs_batter == 4:
            return token, 'four'
        if runs_batter == 6:
            return token, 'six'
        return token, 'default'

    def _play_result_text(self, row):
        wicket_kind = '' if pd.isna(row.get('wicket_kind')) else str(row.get('wicket_kind')).strip().lower()
        runs_batter = int(pd.to_numeric(pd.Series([row.get('runs_batter')]), errors = 'coerce').fillna(0).iloc[0])
        runs_extras = int(pd.to_numeric(pd.Series([row.get('runs_extras')]), errors = 'coerce').fillna(0).iloc[0])
        extra_type = '' if pd.isna(row.get('extra_type')) else str(row.get('extra_type')).strip().lower()

        if 'wides' in extra_type:
            base_text = f"{max(0, runs_extras)} wide" if runs_extras == 1 else f"{max(0, runs_extras)} wides"
        elif 'noballs' in extra_type:
            base_text = f"{max(0, runs_extras)} no ball" if runs_extras == 1 else f"{max(0, runs_extras)} no balls"
        elif 'legbyes' in extra_type:
            base_text = f"{max(0, runs_extras)} leg bye" if runs_extras == 1 else f"{max(0, runs_extras)} leg byes"
        elif 'byes' in extra_type:
            base_text = f"{max(0, runs_extras)} bye" if runs_extras == 1 else f"{max(0, runs_extras)} byes"
        else:
            if runs_batter == 0:
                base_text = 'dot ball'
            elif runs_batter == 1:
                base_text = '1 run'
            elif runs_batter == 4:
                base_text = 'four'
            elif runs_batter == 6:
                base_text = 'six'
            else:
                base_text = f'{runs_batter} runs'

        if wicket_kind:
            wicket_map = {
                'leg before wicket': 'lbw',
            }
            wicket_text = wicket_map.get(wicket_kind, wicket_kind)
            return f'{base_text}, {wicket_text}' if base_text else wicket_text
        return base_text

    def play_by_play_feed(self):
        Xi = self._exclude_placeholder_rows(self.balls.copy())
        if Xi.empty:
            return []

        def _display_col(df, raw_col, canonical_col):
            raw = df[raw_col].fillna('').astype(str).str.strip() if raw_col in df.columns else pd.Series('', index = df.index)
            if canonical_col in df.columns:
                canonical = df[canonical_col].fillna('').astype(str).str.strip()
                return np.where(canonical != '', canonical, raw)
            return raw

        Xi['innings_num'] = pd.to_numeric(Xi.get('innings'), errors = 'coerce')
        Xi = Xi[Xi['innings_num'].notna()].copy()
        if Xi.empty:
            return []

        Xi['over_num'] = pd.to_numeric(Xi.get('over'), errors = 'coerce')
        Xi['ball_num'] = pd.to_numeric(Xi.get('ball'), errors = 'coerce')
        Xi['team_balls_num'] = pd.to_numeric(Xi.get('team_balls'), errors = 'coerce')
        Xi['_id_num'] = pd.to_numeric(Xi.get('id'), errors = 'coerce')
        Xi = Xi[Xi['over_num'].notna() & Xi['ball_num'].notna()].copy()
        if Xi.empty:
            return []

        Xi['batter_display'] = _display_col(Xi, 'batter', 'batter_canonical_name')
        Xi['bowler_display'] = _display_col(Xi, 'bowler', 'bowler_canonical_name')
        Xi['player_out_display'] = _display_col(Xi, 'player_out', 'player_out_canonical_name')
        Xi['batting_team_display'] = Xi.get('batting_team', pd.Series('', index = Xi.index)).fillna('').astype(str).str.strip()
        Xi['runs_total_num'] = pd.to_numeric(Xi.get('runs_total'), errors = 'coerce').fillna(0)
        Xi['team_runs_num'] = pd.to_numeric(Xi.get('team_runs'), errors = 'coerce').fillna(0)
        Xi['team_wicket_num'] = pd.to_numeric(Xi.get('team_wicket'), errors = 'coerce').fillna(0)
        Xi['y_prob_num'] = pd.to_numeric(Xi.get('y_prob'), errors = 'coerce')
        Xi['prob_bowling_diff_num'] = pd.to_numeric(
            Xi['prob_bowling_diff'] if 'prob_bowling_diff' in Xi.columns else pd.Series(0, index = Xi.index),
            errors = 'coerce'
        ).fillna(0)

        Xi['innings_num'] = np.floor(Xi['innings_num']).astype(int)
        Xi['over_num'] = np.floor(Xi['over_num']).astype(int)
        Xi['ball_num'] = Xi['ball_num'].round().astype(int)

        over_end_rows = []
        for (innings_num, over_num), Xo in Xi.groupby(['innings_num', 'over_num'], sort = False):
            Xo_sorted = Xo.sort_values(
                ['team_balls_num', 'ball_num', '_id_num'],
                ascending = [False, False, False],
                na_position = 'last'
            )
            if Xo_sorted.empty:
                continue
            end_row = Xo_sorted.iloc[0]
            over_end_rows.append(
                {
                    'innings_num': int(innings_num),
                    'over_num': int(over_num),
                    'team_wicket_num': int(pd.to_numeric(pd.Series([end_row.get('team_wicket_num')]), errors = 'coerce').fillna(0).iloc[0]),
                }
            )

        over_end_df = pd.DataFrame(over_end_rows)
        prev_wicket_map = {}
        if not over_end_df.empty:
            over_end_df = over_end_df.sort_values(['innings_num', 'over_num']).copy()
            over_end_df['prev_team_wicket_num'] = (
                over_end_df.groupby('innings_num')['team_wicket_num'].shift(1).fillna(0)
            )
            prev_wicket_map = {
                (int(row['innings_num']), int(row['over_num'])): int(row['prev_team_wicket_num'])
                for _, row in over_end_df.iterrows()
            }

        first_innings_runs = pd.to_numeric(
            Xi.loc[Xi['innings_num'] == 1, 'team_runs_num'],
            errors = 'coerce'
        ).dropna()
        first_end_runs = int(first_innings_runs.max()) if not first_innings_runs.empty else 0
        target = self._target(first_end_runs)
        second_innings_ball_limit = self._innings_ball_limit(innings = 2)

        overs = []
        group_cols = ['innings_num', 'over_num']
        for (innings_num, over_num), Xo in Xi.groupby(group_cols, sort = False):
            Xo = Xo.sort_values(
                ['team_balls_num', 'ball_num', '_id_num'],
                ascending = [False, False, False],
                na_position = 'last'
            ).copy()
            if Xo.empty:
                continue

            team_series = Xo['batting_team_display']
            team_series = team_series[team_series.astype(str).str.strip() != '']
            if not team_series.empty:
                team_name = team_series.iloc[0]
            else:
                team_name = str(self.batting_team) if int(innings_num) == 1 else str(self.bowling_team)

            deliveries = []
            for _, row in Xo.iterrows():
                bowler = str(row.get('bowler_display') or '').strip() or 'Unknown bowler'
                batter = str(row.get('batter_display') or '').strip() or 'Unknown batter'
                result_text = self._play_result_text(row)
                token_text, token_style = self._play_result_token(row)
                line_text = f'{bowler} to {batter}, {result_text}'
                ball_num_display = int(pd.to_numeric(pd.Series([row.get('ball_num')]), errors = 'coerce').fillna(0).iloc[0])
                deliveries.append(
                    {
                        'line_text': line_text,
                        'ball_display': f"{int(over_num)}.{ball_num_display}",
                        'token_text': token_text,
                        'token_style': token_style,
                    }
                )

            max_team_balls = pd.to_numeric(Xo['team_balls_num'], errors = 'coerce').max()
            max_team_balls = -1 if pd.isna(max_team_balls) else int(max_team_balls)
            end_row = Xo.iloc[0]
            team_runs_end = int(pd.to_numeric(pd.Series([end_row.get('team_runs_num')]), errors = 'coerce').fillna(0).iloc[0])
            team_wickets_end = int(pd.to_numeric(pd.Series([end_row.get('team_wicket_num')]), errors = 'coerce').fillna(0).iloc[0])
            team_balls_end = int(pd.to_numeric(pd.Series([end_row.get('team_balls_num')]), errors = 'coerce').fillna(0).iloc[0])

            score_text = f'{team_runs_end}/{team_wickets_end}'
            over_runs = int(pd.to_numeric(Xo['runs_total_num'], errors = 'coerce').fillna(0).sum())
            win_prob_swing = float(pd.to_numeric(Xo['prob_bowling_diff_num'], errors = 'coerce').fillna(0).sum()) / 100.0
            prev_wickets = int(prev_wicket_map.get((int(innings_num), int(over_num)), 0))
            over_wickets = max(0, team_wickets_end - prev_wickets)

            wicket_kind_values = Xo.get('wicket_kind', pd.Series('', index = Xo.index))
            over_dismissals = Xo[
                Xo['player_out_display'].astype(str).str.strip().ne('')
                & wicket_kind_values.fillna('').astype(str).str.strip().str.lower().ne('retired hurt')
            ].copy()
            dismissed_batters = (
                over_dismissals['player_out_display']
                .astype(str)
                .str.strip()
                .replace('', pd.NA)
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            run_rate_text = '-'
            if team_balls_end > 0:
                run_rate_text = f"{team_runs_end / (team_balls_end / 6.0):.2f}"

            if int(innings_num) == 1:
                w_int = int(
                    pd.to_numeric(pd.Series([end_row.get('wickets_remaining')]), errors='coerce')
                    .fillna(10).clip(0, 10).astype(int).iloc[0]
                )
                b_float = float(
                    pd.to_numeric(pd.Series([end_row.get('balls_remaining')]), errors='coerce')
                    .fillna(0).clip(lower=0).iloc[0]
                )
                resource_df = pd.DataFrame({'wickets_remaining': [w_int], 'balls_remaining': [b_float]})
                resource_val = float(ipl.resource_function(resource_df, resource_params)[0])
                projected_score = int(round(team_runs_end + resource_val))
                status_text = f'Projected Score: {projected_score}'
                required_run_rate_text = str(projected_score)
                required_rr = np.nan
            elif int(innings_num) == 2:
                projected_score = np.nan
                runs_needed = max(0, int(target) - team_runs_end)
                balls_left = max(0, int(second_innings_ball_limit) - team_balls_end)
                chasing_team = str(self.bowling_team)
                if runs_needed == 0:
                    status_text = f'{chasing_team} have reached the target'
                else:
                    status_text = f'{chasing_team} need {runs_needed} runs from {balls_left} balls'

                if balls_left > 0:
                    required_rr = runs_needed / (balls_left / 6.0)
                    required_run_rate_text = f"{required_rr:.2f}"
                elif runs_needed == 0:
                    required_rr = 0.0
                    required_run_rate_text = '0.00'
                else:
                    required_rr = np.nan
                    required_run_rate_text = '-'
            else:
                projected_score = np.nan
                required_rr = np.nan
                required_run_rate_text = '-'
                status_text = 'Super Over'

            overs.append(
                {
                    'innings': int(innings_num),
                    'team': str(team_name),
                    'over': int(over_num),
                    'over_display': int(over_num) + 1,
                    'max_team_balls': int(max_team_balls),
                    'score_text': score_text,
                    'status_text': status_text,
                    'over_runs': int(over_runs),
                    'over_wickets': int(over_wickets),
                    'y_prob': pd.to_numeric(pd.Series([end_row.get('y_prob_num')]), errors = 'coerce').iloc[0],
                    'win_prob_swing': win_prob_swing if int(innings_num) in [1, 2] else np.nan,
                    'batting_win_prob_delta': np.nan,
                    'team1_win_probability': np.nan,
                    'team2_win_probability': np.nan,
                    'projected_score': projected_score,
                    'previous_projected_score': np.nan,
                    'projected_score_delta': np.nan,
                    'required_rr': required_rr,
                    'previous_required_rr': np.nan,
                    'required_rr_delta': np.nan,
                    'dismissed_batters': dismissed_batters,
                    'run_rate_text': run_rate_text,
                    'required_run_rate_text': required_run_rate_text,
                    'deliveries': deliveries,
                }
        )

        overs_for_delta = sorted(overs, key = lambda o: (o['innings'], o['over']))
        for over in overs_for_delta:
            innings_int = int(over.get('innings', 0) or 0)
            if innings_int not in [1, 2]:
                continue
            prob = pd.to_numeric(pd.Series([over.get('y_prob')]), errors = 'coerce').iloc[0]
            swing = pd.to_numeric(pd.Series([over.get('win_prob_swing')]), errors = 'coerce').iloc[0]
            if pd.isna(swing):
                swing = 0.0
            batting_direction = -1 if innings_int == 1 else 1
            over['win_prob_swing'] = swing
            over['batting_win_prob_delta'] = swing * batting_direction * 100
            if pd.notna(prob):
                over['team2_win_probability'] = float(prob) * 100
                over['team1_win_probability'] = (1 - float(prob)) * 100

        prev_projected_score = self._first_innings_projected_score_baseline(Xi)
        prev_required_rr = np.nan
        for over in overs_for_delta:
            innings_int = int(over.get('innings', 0) or 0)
            if innings_int == 1:
                projected = pd.to_numeric(pd.Series([over.get('projected_score')]), errors = 'coerce').iloc[0]
                over['previous_projected_score'] = prev_projected_score
                if pd.notna(projected) and pd.notna(prev_projected_score):
                    over['projected_score_delta'] = float(projected) - float(prev_projected_score)
                if pd.notna(projected):
                    prev_projected_score = float(projected)
            elif innings_int == 2:
                required_rr = pd.to_numeric(pd.Series([over.get('required_rr')]), errors = 'coerce').iloc[0]
                over['previous_required_rr'] = prev_required_rr
                if pd.notna(required_rr) and pd.notna(prev_required_rr):
                    over['required_rr_delta'] = float(required_rr) - float(prev_required_rr)
                if pd.notna(required_rr):
                    prev_required_rr = float(required_rr)

        overs.sort(key = lambda o: (o['innings'], o['max_team_balls'], o['over']), reverse = True)
        return overs


    def match_summary(self):
        first = self._innings_snapshot(innings = 1)
        second = self._innings_snapshot(innings = 2)

        def _first_non_blank_ball_col(col: str) -> str:
            if col not in self.balls.columns:
                return ""
            values = self.balls[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            return "" if values.empty else str(values.iloc[0]).strip()

        if self.balls.empty:
            current_innings = 1
            current_over = '0.0'
        else:
            latest = self.balls.sort_values(['innings', 'team_balls']).iloc[-1]
            current_innings = int(latest['innings']) if pd.notna(latest['innings']) else 1
            current_balls = int(latest['team_balls']) if pd.notna(latest['team_balls']) else 0
            current_over = self._balls_to_overs(current_balls)

        status_clean = str(self.status).strip().lower()
        is_terminal = status_clean in TERMINAL_MATCH_STATUSES
        is_complete = status_clean == 'complete'
        winner = '' if pd.isna(self.match_won_by) else str(self.match_won_by).strip()
        outcome = '' if pd.isna(self.win_outcome) else str(self.win_outcome).strip()
        method_text = '' if pd.isna(self.method) else str(self.method).strip()
        result_type_text = '' if pd.isna(self.result_type) else str(self.result_type).strip()
        superover_winner_text = '' if pd.isna(self.superover_winner) else str(self.superover_winner).strip()
        if not method_text:
            method_text = _first_non_blank_ball_col('method')
        if not result_type_text:
            result_type_text = _first_non_blank_ball_col('result_type')
        if not superover_winner_text:
            superover_winner_text = _first_non_blank_ball_col('superover_winner')

        if is_terminal:
            if status_clean == 'abandoned':
                result_text = 'Match abandoned'
            elif status_clean == 'no_result':
                result_text = 'No result'
            else:
                result_text = 'Completed'

            if (not winner) or (not outcome):
                derived_winner, derived_outcome = self._derived_completed_result(first, second)
                winner = winner or derived_winner
                outcome = outcome or derived_outcome
            if is_complete and winner and winner.lower() != 'unknown':
                result_text = f'{winner} won by {outcome}' if outcome else f'{winner} won'
            elif is_complete and outcome.lower() == 'tie':
                result_text = 'Match tied'

            if is_complete:
                method_lower = method_text.lower()
                result_type_lower = result_type_text.lower()
                super_over_flag = bool(superover_winner_text) or ('super over' in result_type_lower) or ('superover' in result_type_lower)
                dls_flag = any(token in method_lower for token in ['dls', 'duckworth', 'lewis', 'stern'])
                if super_over_flag and 'super over' not in result_text.lower():
                    result_text = f'{result_text} (Super Over)'
                if dls_flag and 'dls' not in result_text.lower():
                    result_text = f'{result_text} (DLS)'
        else:
            result_text = self._live_result_text(current_innings, first, second)

        venue_parts = [x for x in [self.city, self.venue] if isinstance(x, str) and x.strip()]
        venue_display = ', '.join(venue_parts) if venue_parts else 'Unknown venue'

        return {
            'match_id': int(self.match_id),
            'team1': self.batting_team,
            'team2': self.bowling_team,
            'innings1': first,
            'innings2': second,
            'current_innings': current_innings,
            'current_over': current_over,
            'date': self.date,
            'venue': venue_display,
            'is_live': not is_terminal,
            'result_text': result_text,
            'scheduled_start_ts': self.scheduled_start_ts,
            'toss_winner': self.toss_winner,
            'toss_decision': self.toss_decision,
            'method': method_text,
            'result_type': result_type_text,
            'superover_winner': superover_winner_text,
            'stage': self.stage,
            'event_match_no': self.event_match_no,
            'playoff_match': self.playoff_match,
            'scorecard_has_consistency_issue': bool(self.scorecard_consistency.get('has_issue', False)),
            'scorecard_consistency_message': self.scorecard_consistency.get('message'),
            'scorecard_consistency_by_innings': self.scorecard_consistency.get('innings', {}),
        }
    
    
    def update_scorecard(self):
        self._load_identity_lookup()

        scorecard_balls = self._exclude_placeholder_rows(self.balls[self.balls['innings'].isin([1, 2])].copy())

        consistency_by_innings = {}
        inconsistent_innings = []
        for innings_no in [1, 2]:
            Xi = scorecard_balls[pd.to_numeric(scorecard_balls.get('innings'), errors = 'coerce') == innings_no].copy()
            if Xi.empty:
                consistency_by_innings[innings_no] = {
                    'rows': 0,
                    'legal_ball_contiguous': True,
                    'runs_total_matches_team': True,
                    'is_consistent': True,
                }
                continue

            Xi['valid_ball_num'] = pd.to_numeric(Xi.get('valid_ball'), errors = 'coerce').fillna(0).astype(int)
            Xi['team_balls_num'] = pd.to_numeric(Xi.get('team_balls'), errors = 'coerce')
            Xi['runs_total_num'] = pd.to_numeric(Xi.get('runs_total'), errors = 'coerce').fillna(0)
            Xi['team_runs_num'] = pd.to_numeric(Xi.get('team_runs'), errors = 'coerce')

            legal_team_balls = (
                Xi.loc[
                    (Xi['valid_ball_num'] == 1) & Xi['team_balls_num'].notna() & (Xi['team_balls_num'] > 0),
                    'team_balls_num'
                ]
                .round()
                .astype(int)
                .tolist()
            )
            max_legal_ball = max(legal_team_balls) if legal_team_balls else 0
            distinct_legal_balls = len(set(legal_team_balls))
            legal_ball_contiguous = (max_legal_ball == 0) or (distinct_legal_balls == max_legal_ball)

            sum_runs_total = int(round(Xi['runs_total_num'].sum()))
            max_team_runs = int(round(Xi['team_runs_num'].max())) if Xi['team_runs_num'].notna().any() else 0
            runs_total_matches_team = (max_team_runs == 0) or (sum_runs_total == max_team_runs)

            is_consistent = bool(legal_ball_contiguous and runs_total_matches_team)
            if not is_consistent:
                inconsistent_innings.append(innings_no)

            consistency_by_innings[innings_no] = {
                'rows': int(len(Xi)),
                'max_legal_ball': int(max_legal_ball),
                'distinct_legal_balls': int(distinct_legal_balls),
                'sum_runs_total': int(sum_runs_total),
                'max_team_runs': int(max_team_runs),
                'legal_ball_contiguous': bool(legal_ball_contiguous),
                'runs_total_matches_team': bool(runs_total_matches_team),
                'is_consistent': bool(is_consistent),
            }

        consistency_message = None
        if inconsistent_innings:
            innings_text = ", ".join(str(x) for x in inconsistent_innings)
            consistency_message = f"Live feed is still reconciling for innings {innings_text}. Scorecard values may be incomplete."

        self.scorecard_consistency = {
            'has_issue': bool(len(inconsistent_innings) > 0),
            'innings': consistency_by_innings,
            'message': consistency_message,
        }

        bat_balls = scorecard_balls.copy()
        bat_balls['runs_batter'] = pd.to_numeric(bat_balls['runs_batter'], errors = 'coerce').fillna(0)
        bat_balls['balls_faced'] = pd.to_numeric(bat_balls['balls_faced'], errors = 'coerce').fillna(0)
        bat_balls['bat_pos'] = pd.to_numeric(bat_balls.get('bat_pos'), errors = 'coerce')
        bat_balls['team_balls'] = pd.to_numeric(bat_balls.get('team_balls'), errors = 'coerce')
        bat_balls['over'] = pd.to_numeric(bat_balls.get('over'), errors = 'coerce')
        bat_balls['ball'] = pd.to_numeric(bat_balls.get('ball'), errors = 'coerce')
        bat_balls['_id_num'] = pd.to_numeric(bat_balls.get('id'), errors = 'coerce')
        if 'batter_runs' in bat_balls.columns:
            bat_balls['batter_runs'] = pd.to_numeric(bat_balls['batter_runs'], errors = 'coerce')
        if 'batter_balls' in bat_balls.columns:
            bat_balls['batter_balls'] = pd.to_numeric(bat_balls['batter_balls'], errors = 'coerce')

        def _display_name(df, raw_col, canonical_col):
            raw = df[raw_col].fillna('').astype(str).str.strip() if raw_col in df.columns else pd.Series('', index = df.index)
            if canonical_col in df.columns:
                canonical = df[canonical_col].fillna('').astype(str).str.strip()
                return np.where(canonical != '', canonical, raw)
            return raw

        bat_balls['batter_display'] = _display_name(bat_balls, 'batter', 'batter_canonical_name')
        bat_balls['non_striker_display'] = _display_name(bat_balls, 'non_striker', 'non_striker_canonical_name')
        bat_balls['bowler_display'] = _display_name(bat_balls, 'bowler', 'bowler_canonical_name')
        bat_balls['player_out_display'] = _display_name(bat_balls, 'player_out', 'player_out_canonical_name')
        bat_balls = bat_balls[bat_balls['batter_display'].astype(str).str.strip() != ''].copy()

        bat_sort_cols = [col for col in ['innings', 'team_balls', 'over', 'ball', '_id_num'] if col in bat_balls.columns]
        if bat_sort_cols:
            bat_balls = bat_balls.sort_values(bat_sort_cols).copy()
        bat_balls['_appearance_seq'] = np.arange(len(bat_balls))

        fielder_player_id_col = next(
            (c for c in ['fielders_player_id', 'fielder_player_id'] if c in bat_balls.columns),
            None
        )
        fielder_espn_id_col = next(
            (c for c in ['fielders_espn_athlete_id', 'fielder_espn_athlete_id'] if c in bat_balls.columns),
            None
        )
        bat_balls['fielders_display'] = bat_balls.apply(
            lambda row: self._canonicalize_fielders(
                fielders = row.get('fielders'),
                fielder_player_ids = row.get(fielder_player_id_col) if fielder_player_id_col else None,
                fielder_espn_ids = row.get(fielder_espn_id_col) if fielder_espn_id_col else None
            ),
            axis = 1
        )

        if 'four' not in bat_balls.columns:
            bat_balls['four'] = np.where(bat_balls['runs_batter'] == 4, 1, 0)
        if 'six' not in bat_balls.columns:
            bat_balls['six'] = np.where(bat_balls['runs_batter'] == 6, 1, 0)

        bat_stats = bat_balls.groupby(['innings', 'batter_display'], as_index = False).agg({
            'bat_pos': 'min',
            'runs_batter': 'sum', 
            'balls_faced': 'sum', 
            'four': 'sum', 
            'six': 'sum',
        })

        appearance_sources = [
            bat_balls[['innings', 'batter_display', '_appearance_seq']].assign(_role_priority = 0),
            bat_balls[['innings', 'non_striker_display', '_appearance_seq']].rename(columns = {'non_striker_display': 'batter_display'}).assign(_role_priority = 1),
        ]
        appearance_order = pd.concat(appearance_sources, ignore_index = True)
        appearance_order['batter_display'] = appearance_order['batter_display'].fillna('').astype(str).str.strip()
        appearance_order = appearance_order[appearance_order['batter_display'] != ''].copy()
        appearance_order = (
            appearance_order
            .sort_values(['innings', '_appearance_seq', '_role_priority'])
            .groupby(['innings', 'batter_display'], as_index = False)
            .first()
            .sort_values(['innings', '_appearance_seq', '_role_priority'])
        )
        appearance_order['position_by_appearance'] = appearance_order.groupby('innings').cumcount() + 1

        dismissal_rows_actual = (
            bat_balls.loc[
                bat_balls['player_out_display'].astype(str).str.strip() != '',
                ['innings', 'team_balls', 'over', 'ball', '_id_num', 'player_out_display', 'wicket_kind', 'bowler_display', 'fielders_display']
            ]
            .sort_values(['innings', 'team_balls', 'over', 'ball', '_id_num'])
            .drop_duplicates(['innings', 'player_out_display'], keep = 'last')
            .rename(columns = {'player_out_display': 'batter_display'})
        )
        dismissal_rows = dismissal_rows_actual.copy()
        if not dismissal_rows.empty:
            dismissal_rows['batter_display'] = dismissal_rows['batter_display'].fillna('').astype(str).str.strip()
            dismissal_rows = dismissal_rows[dismissal_rows['batter_display'] != ''].copy()
            dismissal_rows = (
                dismissal_rows
                .sort_values(['innings', 'team_balls', 'over', 'ball', '_id_num'])
                .drop_duplicates(['innings', 'batter_display'], keep = 'last')
                .assign(dismissed = True)
            )

        participants = pd.concat(
            [
                bat_stats[['innings', 'batter_display']],
                dismissal_rows[['innings', 'batter_display']] if not dismissal_rows.empty else pd.DataFrame(columns = ['innings', 'batter_display']),
            ],
            ignore_index = True,
        ).drop_duplicates()

        bat = participants.merge(
            bat_stats,
            on = ['innings', 'batter_display'],
            how = 'left'
        )
        bat = bat.merge(
            appearance_order[['innings', 'batter_display', 'position_by_appearance']],
            on = ['innings', 'batter_display'],
            how = 'left'
        )
        bat = bat.merge(
            dismissal_rows,
            on = ['innings', 'batter_display'],
            how = 'left'
        )

        bat = bat.rename(columns = {
            'position_by_appearance': 'Position',
            'runs_batter': 'Runs',
            'balls_faced': 'Balls',
            'four': '4',
            'six': '6',
            'batter_display': 'Batter'
        })
        bat['Position'] = pd.to_numeric(bat['Position'], errors = 'coerce')
        bat['bat_pos'] = pd.to_numeric(bat['bat_pos'], errors = 'coerce')
        bat['Position'] = bat['Position'].fillna(bat['bat_pos']).fillna(99)
        bat['Position'] = bat['Position'].round().astype(int)
        bat['Runs'] = pd.to_numeric(bat['Runs'], errors = 'coerce').fillna(0).round().astype(int)
        bat['Balls'] = pd.to_numeric(bat['Balls'], errors = 'coerce').fillna(0).round().astype(int)
        bat['4'] = pd.to_numeric(bat['4'], errors = 'coerce').fillna(0).round().astype(int)
        bat['6'] = pd.to_numeric(bat['6'], errors = 'coerce').fillna(0).round().astype(int)
        bat['batter_out_in_innings'] = bat['dismissed'].fillna(False).astype(bool)

        def _dismissal_status(row):
            if not row['batter_out_in_innings']:
                return 'not out'
            return _dismissal_summary(
                wicket_kind = row['wicket_kind'],
                bowler = row['bowler_display'],
                fielders = row['fielders_display']
            )

        bat['Status'] = bat.apply(_dismissal_status, axis = 1)
        bat['Strike Rate'] = np.where(bat['Balls'] > 0, bat['Runs'] / bat['Balls'] * 100, 0)

        bat = bat[['innings', 'Position', 'Batter', 'Status', 'Runs', 'Balls', '4', '6', 'Strike Rate']]

        bat1 = (
            bat[bat['innings'] == 1]
            .sort_values('Position')
            .drop(columns = 'innings')
            .set_index('Position')
        )
        bat2 = (
            bat[bat['innings'] == 2]
            .sort_values('Position')
            .drop(columns = 'innings')
            .set_index('Position')
        )

        self.bat = bat.copy()
        self.bat1 = bat1
        self.bat2 = bat2

        bowl_balls = scorecard_balls.copy()
        bowl_balls['valid_ball'] = pd.to_numeric(bowl_balls['valid_ball'], errors = 'coerce').fillna(0)
        bowl_balls['runs_bowler'] = pd.to_numeric(bowl_balls['runs_bowler'], errors = 'coerce').fillna(0)
        bowl_balls['bowler_wicket'] = pd.to_numeric(bowl_balls['bowler_wicket'], errors = 'coerce').fillna(0)
        bowl_balls['runs_extras'] = pd.to_numeric(bowl_balls['runs_extras'], errors = 'coerce').fillna(0)
        bowl_balls['bowler_display'] = _display_name(bowl_balls, 'bowler', 'bowler_canonical_name')
        bowl_balls['extra_type_str'] = bowl_balls['extra_type'].fillna('').astype(str).str.lower()
        bowl_balls['wides_runs'] = np.where(
            bowl_balls['extra_type_str'].str.contains(r'\bwides\b', regex = True),
            bowl_balls['runs_extras'],
            0
        )
        bowl_balls['noballs_runs'] = np.where(
            bowl_balls['extra_type_str'].str.contains(r'\bnoballs\b', regex = True),
            bowl_balls['runs_extras'],
            0
        )

        bowl = bowl_balls.groupby(['innings', 'bowler_display'], as_index = False, sort = False).agg({
            'valid_ball': 'sum',
            'runs_bowler': 'sum',
            'bowler_wicket': 'sum',
            'wides_runs': 'sum',
            'noballs_runs': 'sum'
        }).rename(columns = {
            'runs_bowler': 'Runs',
            'bowler_wicket': 'Wickets',
            'wides_runs': 'Wides',
            'noballs_runs': 'No Balls',
            'bowler_display': 'Bowler'
        })

        bowl['valid_ball'] = bowl['valid_ball'].round().astype(int)
        bowl['Runs'] = bowl['Runs'].round().astype(int)
        bowl['Wickets'] = bowl['Wickets'].round().astype(int)
        bowl['Wides'] = bowl['Wides'].round().astype(int)
        bowl['No Balls'] = bowl['No Balls'].round().astype(int)
        bowl['Economy'] = np.where(
            bowl['valid_ball'] > 0,
            bowl['Runs'] / (bowl['valid_ball'] / 6),
            0
        )

        def _overs_from_legal_balls(n):
            overs = int(n) // 6
            balls = int(n) % 6
            return f'{overs}.{balls}'

        bowl['Overs'] = bowl['valid_ball'].apply(_overs_from_legal_balls)
        bowl = bowl[['innings', 'Bowler', 'Overs', 'Runs', 'Wickets', 'Economy', 'Wides', 'No Balls']]

        bowl1 = (
            bowl[bowl['innings'] == 1]
            .drop(columns = 'innings')
            .set_index('Bowler')
        )
        bowl2 = (
            bowl[bowl['innings'] == 2]
            .drop(columns = 'innings')
            .set_index('Bowler')
        )

        self.bowl = bowl.copy()
        self.bowl1 = bowl1
        self.bowl2 = bowl2

        return bat1, bat2, bowl1, bowl2


    def summary_over_table(self):
        cached = getattr(self, '_summary_over_table_cache', None)
        if isinstance(cached, pd.DataFrame):
            return cached.copy()

        Xi = self._exclude_placeholder_rows(self.balls[self.balls['innings'].isin([1, 2])].copy())
        columns = [
            'innings',
            'over',
            'over_display',
            'team',
            'over_runs',
            'over_wickets',
            'legal_balls',
            'team_runs',
            'team_wickets',
            'team_balls',
            'run_rate',
            'y_prob',
            'win_prob_swing',
            'win_prob_swing_abs',
            'beneficiary_team',
            'projected_score',
            'current_rr',
            'required_rr',
        ]
        if Xi.empty:
            empty = pd.DataFrame(columns = columns)
            self._summary_over_table_cache = empty.copy()
            return empty

        for col in [
            'innings',
            'over',
            'ball',
            'team_balls',
            'team_runs',
            'team_wicket',
            'runs_total',
            'wicket_taken',
            'valid_ball',
            'y_prob',
            'required_runs',
            'balls_remaining',
            'wickets_remaining',
            'prob_bowling_diff',
            'id',
        ]:
            if col not in Xi.columns:
                Xi[col] = np.nan

        Xi['innings_num'] = pd.to_numeric(Xi['innings'], errors = 'coerce')
        Xi['over_num'] = pd.to_numeric(Xi['over'], errors = 'coerce')
        Xi['ball_num'] = pd.to_numeric(Xi['ball'], errors = 'coerce')
        Xi['team_balls_num'] = pd.to_numeric(Xi['team_balls'], errors = 'coerce')
        Xi['team_runs_num'] = pd.to_numeric(Xi['team_runs'], errors = 'coerce')
        Xi['team_wicket_num'] = pd.to_numeric(Xi['team_wicket'], errors = 'coerce')
        Xi['runs_total_num'] = pd.to_numeric(Xi['runs_total'], errors = 'coerce').fillna(0)
        Xi['wicket_taken_num'] = pd.to_numeric(Xi['wicket_taken'], errors = 'coerce').fillna(0)
        Xi['valid_ball_num'] = pd.to_numeric(Xi['valid_ball'], errors = 'coerce').fillna(0)
        Xi['y_prob_num'] = pd.to_numeric(Xi['y_prob'], errors = 'coerce')
        Xi['prob_bowling_diff_num'] = pd.to_numeric(Xi['prob_bowling_diff'], errors = 'coerce').fillna(0)
        Xi['_id_num'] = pd.to_numeric(Xi['id'], errors = 'coerce')

        Xi = Xi[Xi['innings_num'].isin([1, 2]) & Xi['over_num'].notna()].copy()
        if Xi.empty:
            empty = pd.DataFrame(columns = columns)
            self._summary_over_table_cache = empty.copy()
            return empty

        Xi['innings_num'] = np.floor(Xi['innings_num']).astype(int)
        Xi['over_num'] = np.floor(Xi['over_num']).astype(int)

        first_innings_runs = Xi.loc[Xi['innings_num'] == 1, 'team_runs_num'].dropna()
        target = self._target(int(first_innings_runs.max())) if not first_innings_runs.empty else 1
        second_ball_limit = self._innings_ball_limit(innings = 2)

        rows = []
        group_cols = ['innings_num', 'over_num']
        for (innings_num, over_num), Xo in Xi.groupby(group_cols, sort = False):
            Xo_sorted = Xo.sort_values(
                ['team_balls_num', 'ball_num', '_id_num'],
                ascending = [True, True, True],
                na_position = 'last'
            )
            if Xo_sorted.empty:
                continue

            end_row = Xo_sorted.iloc[-1]
            innings_int = int(innings_num)
            over_int = int(over_num)
            team_name = str(
                self._match_team_name(self.batting_team)
                if innings_int == 1
                else self._match_team_name(self.bowling_team)
            )
            batting_team_values = (
                Xo_sorted.get('batting_team', pd.Series('', index = Xo_sorted.index))
                .fillna('')
                .astype(str)
                .str.strip()
            )
            batting_team_values = batting_team_values[batting_team_values != '']
            if not batting_team_values.empty:
                team_name = str(batting_team_values.iloc[-1])

            team_runs_end = int(pd.to_numeric(pd.Series([end_row.get('team_runs_num')]), errors = 'coerce').fillna(0).iloc[0])
            team_wickets_end = int(pd.to_numeric(pd.Series([end_row.get('team_wicket_num')]), errors = 'coerce').fillna(0).iloc[0])
            team_balls_end = int(pd.to_numeric(pd.Series([end_row.get('team_balls_num')]), errors = 'coerce').fillna(0).iloc[0])
            over_runs = int(round(float(pd.to_numeric(Xo_sorted['runs_total_num'], errors = 'coerce').fillna(0).sum())))
            over_wickets = int(round(float(pd.to_numeric(Xo_sorted['wicket_taken_num'], errors = 'coerce').fillna(0).sum())))
            legal_balls = int(round(float(pd.to_numeric(Xo_sorted['valid_ball_num'], errors = 'coerce').fillna(0).sum())))
            win_prob_swing = float(pd.to_numeric(Xo_sorted['prob_bowling_diff_num'], errors = 'coerce').fillna(0).sum()) / 100.0

            run_rate = np.nan
            if team_balls_end > 0:
                run_rate = team_runs_end / (team_balls_end / 6.0)

            projected_score = np.nan
            current_rr = np.nan
            required_rr = np.nan

            if innings_int == 1:
                w_int = int(
                    pd.to_numeric(pd.Series([end_row.get('wickets_remaining')]), errors = 'coerce')
                    .fillna(10).clip(0, 10).astype(int).iloc[0]
                )
                b_float = float(
                    pd.to_numeric(pd.Series([end_row.get('balls_remaining')]), errors = 'coerce')
                    .fillna(0).clip(lower = 0).iloc[0]
                )
                resource_df = pd.DataFrame({'wickets_remaining': [w_int], 'balls_remaining': [b_float]})
                resource_val = float(ipl.resource_function(resource_df, resource_params)[0])
                projected_score = int(round(team_runs_end + resource_val))
            elif innings_int == 2:
                if team_balls_end > 0:
                    current_rr = team_runs_end / (team_balls_end / 6.0)
                runs_needed = max(0, int(target) - team_runs_end)
                balls_left = max(0, int(second_ball_limit) - team_balls_end)
                if balls_left > 0:
                    required_rr = runs_needed / (balls_left / 6.0)
                elif runs_needed == 0:
                    required_rr = 0.0

            rows.append(
                {
                    'innings': innings_int,
                    'over': over_int,
                    'over_display': over_int + 1,
                    'team': team_name,
                    'over_runs': over_runs,
                    'over_wickets': over_wickets,
                    'legal_balls': legal_balls,
                    'team_runs': team_runs_end,
                    'team_wickets': team_wickets_end,
                    'team_balls': team_balls_end,
                    'run_rate': run_rate,
                    'y_prob': pd.to_numeric(pd.Series([end_row.get('y_prob_num')]), errors = 'coerce').iloc[0],
                    'win_prob_swing': win_prob_swing,
                    'win_prob_swing_abs': abs(win_prob_swing),
                    'beneficiary_team': '',
                    'projected_score': projected_score,
                    'current_rr': current_rr,
                    'required_rr': required_rr,
                }
            )

        over_df = pd.DataFrame(rows, columns = columns)
        if over_df.empty:
            self._summary_over_table_cache = over_df.copy()
            return over_df

        over_df = over_df.sort_values(['innings', 'over', 'team_balls']).reset_index(drop = True)
        batting_team = str(self._match_team_name(self.batting_team))
        bowling_team = str(self._match_team_name(self.bowling_team))
        for row_idx in over_df.index:
            swing = pd.to_numeric(pd.Series([over_df.loc[row_idx, 'win_prob_swing']]), errors = 'coerce').iloc[0]
            if pd.isna(swing):
                swing = 0.0
                over_df.loc[row_idx, 'win_prob_swing'] = swing
                over_df.loc[row_idx, 'win_prob_swing_abs'] = 0.0
            if swing > 0:
                over_df.loc[row_idx, 'beneficiary_team'] = bowling_team
            elif swing < 0:
                over_df.loc[row_idx, 'beneficiary_team'] = batting_team
            else:
                over_df.loc[row_idx, 'beneficiary_team'] = 'Even'

        self._summary_over_table_cache = over_df.copy()
        return over_df


    def most_impactful_over(self):
        over_df = self.summary_over_table()
        if over_df.empty:
            return None

        scored = over_df[pd.to_numeric(over_df['win_prob_swing_abs'], errors = 'coerce').notna()].copy()
        if scored.empty:
            return None

        scored = scored.sort_values(['win_prob_swing_abs', 'innings', 'over'], ascending = [False, True, True])
        row = scored.iloc[0]
        return row.to_dict()


    def innings_phase_summary(self, innings):
        if innings not in [1, 2]:
            raise ValueError('innings must be 1 or 2')

        over_df = self.summary_over_table()
        over_df = over_df[over_df['innings'] == int(innings)].copy()

        phases = [
            ('Powerplay', 1, 6),
            ('Middle Overs', 7, 15),
            ('Death Overs', 16, 20),
        ]
        rows = []
        for phase_name, start_over, end_over in phases:
            phase = over_df[
                (pd.to_numeric(over_df['over_display'], errors = 'coerce') >= start_over)
                & (pd.to_numeric(over_df['over_display'], errors = 'coerce') <= end_over)
            ].copy()
            runs = int(pd.to_numeric(phase.get('over_runs'), errors = 'coerce').fillna(0).sum()) if not phase.empty else 0
            wickets = int(pd.to_numeric(phase.get('over_wickets'), errors = 'coerce').fillna(0).sum()) if not phase.empty else 0
            legal_balls = int(pd.to_numeric(phase.get('legal_balls'), errors = 'coerce').fillna(0).sum()) if not phase.empty else 0
            run_rate = runs / (legal_balls / 6.0) if legal_balls > 0 else np.nan
            team_values = (
                phase.get('team', pd.Series('', index = phase.index))
                .fillna('')
                .astype(str)
                .str.strip()
            ) if not phase.empty else pd.Series(dtype = object)
            team_values = team_values[team_values != '']
            batting_team = str(team_values.iloc[-1]) if not team_values.empty else str(
                self._match_team_name(self.batting_team)
                if int(innings) == 1
                else self._match_team_name(self.bowling_team)
            )
            win_prob_swing = (
                pd.to_numeric(phase.get('win_prob_swing'), errors = 'coerce').dropna()
                if not phase.empty
                else pd.Series(dtype = float)
            )
            batting_direction = -1 if int(innings) == 1 else 1
            net_batting_impact = (
                float(win_prob_swing.sum()) * batting_direction * 100
                if not win_prob_swing.empty
                else np.nan
            )
            rows.append(
                {
                    'Phase': phase_name,
                    'Overs': f'{start_over}-{end_over}',
                    'Team': batting_team,
                    'Runs': runs,
                    'Wickets': wickets,
                    'RR': run_rate,
                    'Net Batting Impact': net_batting_impact,
                }
            )
        return pd.DataFrame(rows)


    def summary_runs_by_over_graph(self, innings, show = False):
        if innings not in [1, 2]:
            raise ValueError('innings must be 1 or 2')

        over_df = self.summary_over_table()
        plot_df = over_df[over_df['innings'] == int(innings)].copy()

        fig = go.Figure()
        if not plot_df.empty:
            hover_customdata = np.column_stack(
                [
                    plot_df['innings'].astype(int),
                    plot_df['over_runs'].astype(int),
                    plot_df['over_wickets'].astype(int),
                    plot_df['team_runs'].astype(int),
                    plot_df['team_wickets'].astype(int),
                ]
            )
            fig.add_trace(
                go.Bar(
                    x = plot_df['over_display'],
                    y = plot_df['over_runs'],
                    marker = dict(color = _innings_color(innings)),
                    customdata = hover_customdata,
                    hovertemplate = (
                        'Innings = %{customdata[0]}<br>'
                        'Over Runs = %{customdata[1]}<br>'
                        'Wickets = %{customdata[2]}<br>'
                        'Score = %{customdata[3]}/%{customdata[4]}'
                        '<extra></extra>'
                    ),
                    showlegend = False,
                )
            )

            wicket_x = []
            wicket_y = []
            marker_step = max(0.75, float(plot_df['over_runs'].max()) * 0.05)
            for _, row in plot_df.iterrows():
                wickets_in_over = int(row['over_wickets']) if pd.notna(row['over_wickets']) else 0
                if wickets_in_over <= 0:
                    continue
                bar_top = float(row['over_runs']) if pd.notna(row['over_runs']) else 0.0
                for i in range(wickets_in_over):
                    wicket_x.append(row['over_display'])
                    wicket_y.append(bar_top + marker_step * (i + 1))

            if wicket_x:
                fig.add_trace(
                    go.Scatter(
                        x = wicket_x,
                        y = wicket_y,
                        mode = 'markers',
                        marker = dict(symbol = 'x', size = 11, color = 'red'),
                        hoverinfo = 'skip',
                        showlegend = False,
                    )
                )

            team_runs = pd.to_numeric(plot_df['team_runs'], errors = 'coerce').dropna()
            team_balls = pd.to_numeric(plot_df['team_balls'], errors = 'coerce').dropna()
            if not team_runs.empty and not team_balls.empty and float(team_balls.max()) > 0:
                innings_rr = float(team_runs.max()) / (float(team_balls.max()) / 6.0)
                fig.add_shape(
                    type = 'line',
                    x0 = 0.35,
                    x1 = 20.65,
                    y0 = innings_rr,
                    y1 = innings_rr,
                    line = dict(color = PLOTLY_COLORS['reference_line'], width = 2, dash = 'dash'),
                )
                fig.add_annotation(
                    text = f'RR: {innings_rr:.2f}',
                    x = 20.65,
                    y = innings_rr,
                    xref = 'x',
                    yref = 'y',
                    xanchor = 'right',
                    yanchor = 'bottom',
                    showarrow = False,
                    font = dict(size = max(9, PLOTLY_LABEL_FONT_SIZE - 3), family = PLOTLY_FONT_FAMILY, color = PLOTLY_COLORS['reference_line']),
                    yshift = 1,
                )
        else:
            fig.add_annotation(
                text = 'No over data available.',
                x = 0.5,
                y = 0.5,
                xref = 'paper',
                yref = 'paper',
                showarrow = False,
            )

        fig.update_layout(
            title = dict(
                text = f'<b>Runs by Over (Innings {innings})</b>',
                x = 0.5,
                xanchor = 'center',
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = 'top',
            ),
            xaxis_title = '<b>Over</b>',
            yaxis_title = '<b>Runs</b>',
            width = 1000,
            height = 600,
            margin = dict(t = PLOTLY_HEADER_MARGIN_TOP, l = 50, r = 25, b = 40),
        )
        _apply_plot_theme(fig)
        fig.update_xaxes(
            tickmode = 'array',
            tickvals = list(range(4, 21, 4)),
            range = [0.35, 20.65],
            fixedrange = True,
            showgrid = False,
        )
        y_upper = None
        if not plot_df.empty:
            max_runs = float(plot_df['over_runs'].max()) if pd.notna(plot_df['over_runs'].max()) else 0.0
            rr_values = pd.to_numeric(plot_df['run_rate'], errors = 'coerce').dropna()
            max_rr = float(rr_values.max()) if not rr_values.empty else 0.0
            y_upper = max(6.0, max_runs * 1.25, max_rr * 1.18)
        fig.update_yaxes(range = [0, y_upper], domain = [0, PLOTLY_HEADER_PLOT_TOP], fixedrange = True, showgrid = False)

        if show:
            fig.show(config = {'displayModeBar': False, 'scrollZoom': False})
        return fig


    def projected_score_by_over_graph(self, show = False):
        over_df = self.summary_over_table()
        plot_df = over_df[(over_df['innings'] == 1) & pd.to_numeric(over_df['projected_score'], errors = 'coerce').notna()].copy()

        fig = go.Figure()
        if not plot_df.empty:
            customdata = np.column_stack(
                [
                    plot_df['over_display'].astype(int),
                    plot_df['team_runs'].astype(int),
                    plot_df['team_wickets'].astype(int),
                    plot_df['projected_score'].astype(int),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x = plot_df['over_display'],
                    y = plot_df['projected_score'],
                    mode = 'lines',
                    line = dict(color = _innings_color(1), width = 2),
                    customdata = customdata,
                    hovertemplate = (
                        'Over = %{customdata[0]}<br>'
                        'Score = %{customdata[1]}/%{customdata[2]}<br>'
                        'Projected Score = %{customdata[3]}'
                        '<extra></extra>'
                    ),
                    showlegend = False,
                )
            )

            status_clean = str(getattr(self, 'status', '') or '').strip().lower()
            first_innings_complete = (
                status_clean in TERMINAL_MATCH_STATUSES
                or status_clean == 'innings_break'
                or not over_df[over_df['innings'] == 2].empty
            )
            actual_score = pd.to_numeric(plot_df['team_runs'], errors = 'coerce').dropna()
            if first_innings_complete and not actual_score.empty:
                actual_total = float(actual_score.max())
                fig.add_shape(
                    type = 'line',
                    x0 = 0.35,
                    x1 = 20.65,
                    y0 = actual_total,
                    y1 = actual_total,
                    line = dict(color = PLOTLY_COLORS['reference_line'], width = 2, dash = 'dash'),
                )
                fig.add_annotation(
                    text = f'Actual: {int(round(actual_total))}',
                    x = 20.65,
                    y = actual_total,
                    xref = 'x',
                    yref = 'y',
                    xanchor = 'right',
                    yanchor = 'bottom',
                    showarrow = False,
                    font = dict(size = max(9, PLOTLY_LABEL_FONT_SIZE - 3), family = PLOTLY_FONT_FAMILY, color = PLOTLY_COLORS['reference_line']),
                    yshift = 1,
                )
        else:
            fig.add_annotation(
                text = 'No projected score data available.',
                x = 0.5,
                y = 0.5,
                xref = 'paper',
                yref = 'paper',
                showarrow = False,
            )

        fig.update_layout(
            title = dict(
                text = '<b>Projected Score by Over (Innings 1)</b>',
                x = 0.5,
                xanchor = 'center',
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = 'top',
            ),
            xaxis_title = '<b>Over</b>',
            yaxis_title = '<b>Projected Score</b>',
            width = 1000,
            height = 600,
            margin = dict(t = PLOTLY_HEADER_MARGIN_TOP, l = 55, r = 25, b = 40),
        )
        _apply_plot_theme(fig)
        fig.update_xaxes(
            tickmode = 'array',
            tickvals = list(range(4, 21, 4)),
            range = [0.35, 20.65],
            fixedrange = True,
            showgrid = True,
        )
        fig.update_yaxes(domain = [0, PLOTLY_HEADER_PLOT_TOP], fixedrange = True)

        if show:
            fig.show(config = {'displayModeBar': False, 'scrollZoom': False})
        return fig


    def chase_run_rate_by_over_graph(self, show = False):
        over_df = self.summary_over_table()
        plot_df = over_df[over_df['innings'] == 2].copy()

        fig = go.Figure()
        if not plot_df.empty:
            current = pd.to_numeric(plot_df['current_rr'], errors = 'coerce')
            required = pd.to_numeric(plot_df['required_rr'], errors = 'coerce')
            score_data = np.column_stack(
                [
                    plot_df['over_display'].astype(int),
                    plot_df['team_runs'].astype(int),
                    plot_df['team_wickets'].astype(int),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x = plot_df['over_display'],
                    y = current,
                    mode = 'lines',
                    name = 'Actual RR',
                    line = dict(color = _innings_color(2), width = 2),
                    customdata = score_data,
                    hovertemplate = (
                        'Over = %{customdata[0]}<br>'
                        'Score = %{customdata[1]}/%{customdata[2]}<br>'
                        'Actual RR = %{y:.2f}'
                        '<extra></extra>'
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x = plot_df['over_display'],
                    y = required,
                    mode = 'lines',
                    name = 'Required RR',
                    line = dict(color = PLOTLY_COLORS['reference_line'], width = 2, dash = 'dash'),
                    customdata = score_data,
                    hovertemplate = (
                        'Over = %{customdata[0]}<br>'
                        'Score = %{customdata[1]}/%{customdata[2]}<br>'
                        'Required RR = %{y:.2f}'
                        '<extra></extra>'
                    ),
                )
            )
        else:
            fig.add_annotation(
                text = 'No chase run-rate data available.',
                x = 0.5,
                y = 0.5,
                xref = 'paper',
                yref = 'paper',
                showarrow = False,
            )

        legend_layout = _header_legend_layout()
        legend_layout['y'] = float(legend_layout.get('y', 0.95)) - 0.01
        fig.update_layout(
            title = dict(
                text = '<b>Actual RR vs. Required RR (Innings 2)</b>',
                x = 0.5,
                xanchor = 'center',
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = 'top',
            ),
            xaxis_title = '<b>Over</b>',
            yaxis_title = '<b>Run Rate</b>',
            width = 1000,
            height = 600,
            legend = legend_layout,
            margin = dict(t = PLOTLY_HEADER_MARGIN_TOP, l = 55, r = 25, b = 40),
        )
        _apply_plot_theme(fig)
        fig.update_xaxes(
            tickmode = 'array',
            tickvals = list(range(4, 21, 4)),
            range = [0.35, 20.65],
            fixedrange = True,
            showgrid = True,
        )
        fig.update_yaxes(rangemode = 'tozero', domain = [0, PLOTLY_HEADER_PLOT_TOP], fixedrange = True)

        if show:
            fig.show(config = {'displayModeBar': False, 'scrollZoom': False})
        return fig
    
    
    def run_chart_by_over(self, innings, show = False):
        if innings not in [1, 2]:
            raise ValueError('innings must be 1 or 2')

        #color = 'blue' if innings == 1 else 'red'

        Xi = self.balls.loc[self.balls['innings'] == innings].copy()
        over_runs = (
            Xi.groupby('over', as_index = False)
            .agg(
                runs_scored = ('runs_total', 'sum'),
                wickets_taken = ('wicket_taken', 'sum')
            )
            .sort_values('over')
        )

        over_runs['innings'] = innings
        hover_customdata = np.column_stack(
            [
                over_runs['innings'].astype(int),
                over_runs['runs_scored'].astype(int),
                over_runs['wickets_taken'].astype(int)
            ]
        )

        fig = go.Figure(
            data = [
                go.Bar(
                    x = over_runs['over'],
                    y = over_runs['runs_scored'],
                    marker = dict(color = _innings_color(innings)),
                    customdata = hover_customdata,
                    hovertemplate = (
                        'Innings = %{customdata[0]}<br>'
                        'Runs Scored = %{customdata[1]}<br>'
                        'Wickets Taken = %{customdata[2]}'
                        '<extra></extra>'
                    ),
                    showlegend = False
                )
            ]
        )

        wicket_x = []
        wicket_y = []
        marker_step = max(0.75, float(over_runs['runs_scored'].max()) * 0.05) if not over_runs.empty else 0.75
        stack_padding = max(0.14, marker_step * 0.30)
        marker_step_with_padding = marker_step + stack_padding
        for _, row in over_runs.iterrows():
            wickets_in_over = int(row['wickets_taken']) if pd.notna(row['wickets_taken']) else 0
            if wickets_in_over <= 0:
                continue
            bar_top = float(row['runs_scored']) if pd.notna(row['runs_scored']) else 0.0
            for i in range(wickets_in_over):
                wicket_x.append(row['over'])
                wicket_y.append(bar_top + marker_step_with_padding * (i + 1))

        if wicket_x:
            fig.add_trace(
                go.Scatter(
                    x = wicket_x,
                    y = wicket_y,
                    mode = 'markers',
                    marker = dict(symbol = 'x', size = 12, color = 'red'),
                    hoverinfo = 'skip',
                    showlegend = False
                )
            )

        fig.update_layout(
            title = dict(
                text = f'<b>Runs by Over (Innings {innings})</b>',
                x = 0.5,
                xanchor = 'center',
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = 'top'
            ),
            xaxis_title = '<b>Over</b>',
            yaxis_title = '<b>Runs Scored</b>',
            width = 1000,
            height = 600,
            margin = dict(
                t = PLOTLY_HEADER_MARGIN_TOP,
                l = PLOTLY_HEADER_MARGIN_LEFT,
                r = PLOTLY_HEADER_MARGIN_RIGHT
            )
        )
        _apply_plot_theme(fig)

        if not Xi.empty:
            last_row = Xi.iloc[-1]
            innings_runs = last_row['team_runs']
            innings_balls = last_row['team_balls']
            if pd.notna(innings_runs) and pd.notna(innings_balls) and innings_balls > 0:
                overall_run_rate = innings_runs / innings_balls * 6
                if np.isfinite(overall_run_rate):
                    fig.add_hline(
                        y = overall_run_rate,
                        line_dash = 'dash',
                        line_color = PLOTLY_COLORS['reference_line'],
                        line_width = PLOTLY_REFERENCE_LINE_WIDTH
                    )

        fig.update_xaxes(
            range = [-0.65, 19.65],
            tickmode = 'array',
            tickvals = list(range(0, 20)),
            fixedrange = True
        )
        if wicket_y:
            y_upper = max(
                float(over_runs['runs_scored'].max()) if not over_runs.empty else 0.0,
                max(wicket_y) + marker_step * 0.5
            )
        else:
            y_upper = None
        fig.update_yaxes(
            range = [0, y_upper],
            domain = [0, PLOTLY_HEADER_PLOT_TOP],
            fixedrange = True
        )

        if show:
            fig.show(config = {
                'displayModeBar': False,
                'scrollZoom': False
            })
        return fig
    
    
    def ball_ratios(self, innings = 1):
        
        return
    
    
    def _build_predict_figure(self, smooth_window = None):
        X_first = self.balls[self.balls['innings'] == 1].copy()
        X_second = self.balls[self.balls['innings'] == 2].copy()
        first_sort_cols = [col for col in ['team_balls', 'over', 'ball', 'id'] if col in X_first.columns]
        second_sort_cols = [col for col in ['team_balls', 'over', 'ball', 'id'] if col in X_second.columns]
        if first_sort_cols:
            X_first = X_first.sort_values(first_sort_cols)
        if second_sort_cols:
            X_second = X_second.sort_values(second_sort_cols)

        if smooth_window is not None:
            try:
                smooth_window = int(smooth_window)
            except (TypeError, ValueError):
                raise ValueError('n must be a positive integer')
            if smooth_window < 1:
                raise ValueError('n must be a positive integer')

            # Smooth each innings independently so the break is preserved.
            X_first['y_prob_plot'] = X_first['y_prob'].rolling(window = smooth_window, min_periods = 1).mean()
            X_second['y_prob_plot'] = X_second['y_prob'].rolling(window = smooth_window, min_periods = 1).mean()
        else:
            X_first['y_prob_plot'] = X_first['y_prob']
            X_second['y_prob_plot'] = X_second['y_prob']

        # Keep terminal match state exact on the final plotted second-innings point.
        terminal_second = self._second_innings_terminal_probability()
        if terminal_second is not None:
            terminal_idx, terminal_prob = terminal_second
            if terminal_idx in X_second.index:
                X_second.loc[terminal_idx, 'y_prob_plot'] = float(terminal_prob)

        batting_team = self.batting_team.iloc[0] if isinstance(self.batting_team, pd.Series) else self.batting_team
        bowling_team = self.bowling_team.iloc[0] if isinstance(self.bowling_team, pd.Series) else self.bowling_team

        first_end_ball = int(X_first['team_balls'].max()) if not X_first.empty else 0
        second_end_ball = int(X_second['team_balls'].max()) if not X_second.empty else 0
        first_end_wickets = int(X_first['team_wicket'].max()) if not X_first.empty else 0
        second_end_wickets = int(X_second['team_wicket'].max()) if not X_second.empty else 0
        first_end_runs = int(X_first['team_runs'].max()) if not X_first.empty else 0
        second_end_runs = int(X_second['team_runs'].max()) if not X_second.empty else 0

        status_complete = str(self.status).strip().lower() == 'complete'

        # Truncate an innings only when it is actually complete.
        first_innings_complete = (
            (not X_first.empty) and (
                first_end_ball >= 120
                or first_end_wickets >= 10
                or not X_second.empty
            )
        )
        second_innings_complete = (
            (not X_second.empty) and (
                second_end_ball >= 120
                or second_end_wickets >= 10
                or status_complete
            )
        )
        if (not second_innings_complete) and (not X_first.empty) and (not X_second.empty):
            target = self._target(first_end_runs)
            second_innings_complete = second_end_runs >= target

        first_display_balls = first_end_ball if first_innings_complete else 120
        second_display_balls = second_end_ball if second_innings_complete else 120

        X_first['global_ball'] = X_first['team_balls']
        X_second['global_ball'] = X_second['team_balls'] + first_display_balls

        # -------------------------------------------------------------------
        # Create one continuous line:
        # - first innings points
        # - then second innings points
        #
        # Because these are in one trace, Plotly will connect the final point
        # of innings 1 directly to the first point of innings 2.
        # -------------------------------------------------------------------
        plot_df = pd.concat(
            [X_first, X_second],
            ignore_index = True
        )

        x_vals = plot_df['global_ball'] / 6
        y_vals = plot_df['y_prob_plot'] * 100
        raw_y_vals = pd.to_numeric(plot_df['y_prob'], errors = 'coerce') * 100
        raw_y_vals = raw_y_vals.where(raw_y_vals.notna(), y_vals)
        innings2_prob = y_vals
        innings1_prob = 100 - y_vals
        innings1_ahead = innings1_prob >= 50
        raw_innings2_prob = raw_y_vals
        raw_innings1_prob = 100 - raw_y_vals
        raw_innings1_ahead = raw_innings1_prob >= 50

        # Cricket-style over notation from ball count:
        # 1 -> 0.1, 6 -> 0.6, 7 -> 1.1, ..., 96 -> 15.6
        over_num = (plot_df['team_balls'].astype(int) - 1) // 6
        ball_in_over = ((plot_df['team_balls'].astype(int) - 1) % 6) + 1
        over_display = over_num.astype(str) + '.' + ball_in_over.astype(str)
        score_display = (
            plot_df['team_runs'].astype(int).astype(str)
            + '/'
            + plot_df['team_wicket'].astype(int).astype(str)
        )
        lead_team = np.where(raw_innings1_ahead, batting_team, bowling_team)
        lead_team_prob = np.where(raw_innings1_ahead, raw_innings1_prob, raw_innings2_prob)
        hover_customdata = np.column_stack(
            [lead_team, lead_team_prob, plot_df['innings'].astype(int), over_display, score_display]
        )

        # -------------------------------------------------------------------
        # Custom x-axis ticks:
        # show overs relative to each innings, i.e.
        # 1, 2, ..., 20 | 1, 2, ..., 20
        #
        # Tick positions are on the continuous global axis.
        # Tick labels reset after the innings break.
        # -------------------------------------------------------------------
        def _segment_ticks(display_balls):
            display_overs = float(display_balls) / 6.0
            ticks = [ov for ov in range(0, 21, 4) if ov <= display_overs + 1e-9]
            if not ticks:
                ticks = [0.0]
            if abs(ticks[-1] - display_overs) > 1e-9:
                ticks.append(display_overs)
            return ticks

        def _tick_label(overs):
            if abs(float(overs) - round(float(overs))) < 1e-9:
                return str(int(round(float(overs))))
            return f'{float(overs):.1f}'

        first_rel_ticks = _segment_ticks(first_display_balls)
        second_rel_ticks = _segment_ticks(second_display_balls)
        tickvals = []
        ticktext = []
        tick_index = {}

        for ov in first_rel_ticks:
            x_val = float(ov)
            key = round(x_val, 6)
            if key in tick_index:
                ticktext[tick_index[key]] = _tick_label(ov)
            else:
                tick_index[key] = len(tickvals)
                tickvals.append(x_val)
                ticktext.append(_tick_label(ov))

        for ov in second_rel_ticks:
            x_val = float(ov + (first_display_balls / 6.0))
            key = round(x_val, 6)
            if key in tick_index:
                ticktext[tick_index[key]] = _tick_label(ov)
            else:
                tick_index[key] = len(tickvals)
                tickvals.append(x_val)
                ticktext.append(_tick_label(ov))

        # Innings break location: just before innings 2 begins
        innings_break_x = first_display_balls / 6.0
        x_range_end = (first_display_balls + second_display_balls) / 6.0 + 0.5

        # Plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x = [None],
                y = [None],
                mode = 'lines',
                line = dict(width = 2, color = PLOTLY_COLORS['innings_1']),
                name = str(batting_team),
                hoverinfo = 'skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = [None],
                y = [None],
                mode = 'lines',
                line = dict(width = 2, color = PLOTLY_COLORS['innings_2']),
                name = str(bowling_team),
                hoverinfo = 'skip'
            )
        )

        n_points = len(y_vals)
        team1_points = np.zeros(n_points, dtype = bool)
        team2_points = np.zeros(n_points, dtype = bool)
        if n_points >= 2:
            segment_team1 = (y_vals.iloc[:-1] <= 50).to_numpy()
            for i, is_team1 in enumerate(segment_team1):
                if is_team1:
                    team1_points[i] = True
                    team1_points[i + 1] = True
                else:
                    team2_points[i] = True
                    team2_points[i + 1] = True
        elif n_points == 1:
            if float(y_vals.iloc[0]) <= 50:
                team1_points[0] = True
            else:
                team2_points[0] = True

        line_team1 = y_vals.where(team1_points, np.nan)
        line_team2 = y_vals.where(team2_points, np.nan)
        line_hovertemplate = (
            '%{customdata[0]}<br>'
            'Win Probability = %{customdata[1]:.1f}%<br>'
            'Innings = %{customdata[2]}<br>'
            'Overs = %{customdata[3]}<br>'
            'Score = %{customdata[4]}'
            '<extra></extra>'
        )

        fig.add_trace(go.Scatter(
            x = x_vals,
            y = line_team1,
            customdata = hover_customdata,
            hovertemplate = line_hovertemplate,
            mode = 'lines',
            line = dict(width = 2, color = PLOTLY_COLORS['innings_1']),
            showlegend = False
        ))
        fig.add_trace(go.Scatter(
            x = x_vals,
            y = line_team2,
            customdata = hover_customdata,
            hovertemplate = line_hovertemplate,
            mode = 'lines',
            line = dict(width = 2, color = PLOTLY_COLORS['innings_2']),
            showlegend = False
        ))

        wickets = plot_df.loc[plot_df['player_out'].notna()].copy()
        if not wickets.empty:
            wickets = (
                wickets.sort_values(['innings', 'over', 'ball'])
                .drop_duplicates(['innings', 'player_out'], keep = 'last')
            )
            wickets['team_name'] = np.where(wickets['innings'] == 1, batting_team, bowling_team)
            wickets['dismissal_summary'] = wickets.apply(
                lambda row: _dismissal_summary(
                    wicket_kind = row['wicket_kind'],
                    bowler = row['bowler'],
                    fielders = row['fielders']
                ),
                axis = 1
            )
            wickets['x_plot'] = wickets['global_ball'] / 6
            wickets['y_plot'] = wickets['y_prob_plot'] * 100
            wicket_over_num = (wickets['team_balls'].astype(int) - 1) // 6
            wicket_ball_in_over = ((wickets['team_balls'].astype(int) - 1) % 6) + 1
            wickets['over_display'] = wicket_over_num.astype(str) + '.' + wicket_ball_in_over.astype(str)
            wickets['score_display'] = (
                wickets['team_runs'].astype(int).astype(str)
                + '/'
                + wickets['team_wicket'].astype(int).astype(str)
            )

            for inn in [1, 2]:
                Wi = wickets.loc[wickets['innings'] == inn].copy()
                if Wi.empty:
                    continue
                marker_customdata = np.column_stack(
                    [
                        Wi['team_name'],
                        Wi['player_out'].astype(str),
                        Wi['dismissal_summary'],
                        Wi['innings'].astype(int),
                        Wi['over_display'],
                        Wi['score_display']
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x = Wi['x_plot'],
                        y = Wi['y_plot'],
                        mode = 'markers',
                        marker = dict(symbol = 'x', size = 9, color = _innings_color(inn)),
                        customdata = marker_customdata,
                        hovertemplate = (
                            '%{customdata[0]}<br>'
                            'Player Out = %{customdata[1]}<br>'
                            '%{customdata[2]}<br>'
                            'Innings = %{customdata[3]}<br>'
                            'Overs = %{customdata[4]}<br>'
                            'Score = %{customdata[5]}'
                            '<extra></extra>'
                        ),
                        showlegend = False
                    )
                )

        legend_layout = _header_legend_layout()
        legend_layout['y'] = float(legend_layout.get('y', 0.95)) - 0.01
        fig.update_layout(
            title = dict(
                text = '<b>Win Probability (%)</b>',
                x = 0.5,
                xanchor = 'center',
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = 'top'
            ),
            xaxis_title = '<b>Over</b>',
            yaxis_title = '<b>Win Probability (%)</b>',
            width=1000,
            height=600,
            legend = legend_layout,
            margin = dict(
                t = PLOTLY_HEADER_MARGIN_TOP,
                l = PLOTLY_HEADER_MARGIN_LEFT,
                r = PLOTLY_HEADER_MARGIN_RIGHT
            )
        )
        _apply_plot_theme(fig)

        fig.update_xaxes(
            tickmode = 'array',
            tickvals = tickvals,
            ticktext = ticktext,
            range = [0, x_range_end]
        )

        fig.update_yaxes(
            range = [0, 100],
            domain = [0, PLOTLY_HEADER_PLOT_TOP]
        )

        fig.add_hline(
            y = 50,
            line_dash = 'dash',
            line_color = PLOTLY_COLORS['reference_line'],
            line_width = PLOTLY_REFERENCE_LINE_WIDTH
        )
        fig.add_annotation(
            x = x_range_end - 0.02,
            y = 50,
            xref = 'x',
            yref = 'y',
            text = '50%',
            showarrow = False,
            xanchor = 'right',
            yanchor = 'bottom',
            font = dict(
                family = PLOTLY_FONT_FAMILY,
                size = PLOTLY_AXIS_TICK_FONT_SIZE,
                color = PLOTLY_COLORS['reference_line']
            ),
        )

        # Dashed separator between innings
        fig.add_vline(
            x=innings_break_x,
            line_dash='dash',
            line_color=PLOTLY_COLORS['reference_line'],
            line_width = PLOTLY_REFERENCE_LINE_WIDTH
        )

        return fig

    def predict(self, show = False):
        fig = self._build_predict_figure()
        if show:
            fig.show()
        return fig

    def predict_smooth(self, n = 3, show = False):
        fig = self._build_predict_figure(smooth_window=n)
        if show:
            fig.show()
        return fig
    
    
    def worm(self, show = False):
        fig = go.Figure()

        batting_team = self.batting_team.iloc[0] if isinstance(self.batting_team, pd.Series) else self.batting_team
        bowling_team = self.bowling_team.iloc[0] if isinstance(self.bowling_team, pd.Series) else self.bowling_team
        innings_team = {1: batting_team, 2: bowling_team}

        for inn in [1, 2]:
            Xi = self.balls.loc[self.balls['innings'] == inn].copy()
            if Xi.empty:
                continue

            Xi = Xi.sort_values('team_balls')
            Xi['overs_plot'] = Xi['team_balls'] / 6.0
            over_num = (Xi['team_balls'].astype(int) - 1) // 6
            ball_in_over = ((Xi['team_balls'].astype(int) - 1) % 6) + 1
            Xi['over_display'] = over_num.astype(str) + '.' + ball_in_over.astype(str)
            Xi['score_display'] = Xi['team_runs'].astype(int).astype(str) + '/' + Xi['team_wicket'].astype(int).astype(str)
            Xi['team_name'] = innings_team.get(inn, '')

            line_customdata = np.column_stack(
                [Xi['team_name'], Xi['innings'].astype(int), Xi['over_display'], Xi['score_display']]
            )

            fig.add_trace(
                go.Scatter(
                    x = Xi['overs_plot'],
                    y = Xi['team_runs'],
                    mode = 'lines',
                    name = str(innings_team.get(inn, f'Innings {inn}')),
                    line = dict(color = _innings_color(inn), width = 2),
                    customdata = line_customdata,
                    hovertemplate = (
                        '%{customdata[0]}<br>'
                        'Innings = %{customdata[1]}<br>'
                        'Overs = %{customdata[2]}<br>'
                        'Score = %{customdata[3]}'
                        '<extra></extra>'
                    )
                )
            )

            W = Xi.loc[Xi['wicket_kind'].notna()].copy()
            if W.empty:
                continue

            wicket_customdata = np.column_stack(
                [W['team_name'], W['innings'].astype(int), W['over_display'], W['score_display']]
            )

            fig.add_trace(
                go.Scatter(
                    x = W['overs_plot'],
                    y = W['team_runs'],
                    mode = 'markers',
                    marker = dict(symbol = 'x', size = 8, color = _innings_color(inn)),
                    showlegend = False,
                    customdata = wicket_customdata,
                    hovertemplate = (
                        '%{customdata[0]}<br>'
                        'Innings = %{customdata[1]}<br>'
                        'Overs = %{customdata[2]}<br>'
                        'Score = %{customdata[3]}'
                        '<extra></extra>'
                    )
                )
            )

        fig.update_xaxes(
            tickmode = 'auto',
            rangemode = 'tozero',
            showline = False,
            zeroline = False,
            mirror = False,
            fixedrange = True
        )

        fig.update_yaxes(
            showline = False,
            zeroline = False,
            mirror = False,
            fixedrange = True,
            domain = [0, PLOTLY_HEADER_PLOT_TOP]
        )

        legend_layout = _header_legend_layout()
        legend_layout['y'] = float(legend_layout.get('y', 0.95)) - 0.01
        fig.update_layout(
            xaxis_title = '<b>Overs</b>',
            yaxis_title = '<b>Total Runs</b>',
            title = dict(
                text = '<b>Innings Progression</b>',
                x = 0.5,
                xanchor = 'center',
                y = PLOTLY_HEADER_TITLE_Y,
                yanchor = 'top'
            ),
            width = 800,
            height = 600,
            dragmode = False,
            legend = legend_layout,
            margin = dict(
                t = PLOTLY_HEADER_MARGIN_TOP,
                l = PLOTLY_HEADER_MARGIN_LEFT,
                r = PLOTLY_HEADER_MARGIN_RIGHT
            )
        )
        _apply_plot_theme(fig)

        if show:
            fig.show(config = {
                'displayModeBar': False,
                'scrollZoom': False
            })
        return fig
    
    def update_loop(self):
        return
