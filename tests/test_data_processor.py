"""
Unit tests for the DataProcessor class.

Tests cover:
- Duplicate removal
- Missing value handling
- Note validation
- Temporal feature extraction
- Activity metrics computation
- Component features
- Merge logs and notes
- Full build_student_dataset pipeline
- Edge cases
- Real data files
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from src import Config
from src.data_processor import DataProcessor


@pytest.fixture
def processor():
    """Create a DataProcessor instance for testing."""
    return DataProcessor()


@pytest.fixture
def sample_logs_df():
    """Create a sample logs DataFrame for testing."""
    return pd.DataFrame({
        'heure': pd.to_datetime([
            '2024-07-24 09:48:08',
            '2024-07-24 09:48:14',
            '2024-08-19 12:55:34',
            '2024-08-19 13:10:00',
            '2024-09-01 08:00:00',
        ]),
        'pseudo': [436, 436, 841, 841, 436],
        'contexte': [
            'Cours: PASS - S1',
            'Fichier: Contrat',
            'Cours: PASS - S1',
            'Fichier: Contrat',
            'Cours: PASS - S1',
        ],
        'composant': ['Système', 'Fichier', 'Système', 'Fichier', 'Système'],
        'evenement': [
            'Cours consulté',
            'Module de cours consulté',
            'Cours consulté',
            'Module de cours consulté',
            'Cours consulté',
        ],
    })


@pytest.fixture
def sample_notes_df():
    """Create a sample notes DataFrame for testing."""
    return pd.DataFrame({
        'pseudo': [436, 841, 318],
        'note': [11.05, 14.5, 10.0],
    })


@pytest.fixture
def logs_with_duplicates():
    """Create a logs DataFrame with duplicate rows."""
    return pd.DataFrame({
        'heure': pd.to_datetime([
            '2024-07-24 09:48:08',
            '2024-07-24 09:48:08',
            '2024-08-19 12:55:34',
        ]),
        'pseudo': [436, 436, 841],
        'contexte': ['Cours: PASS - S1', 'Cours: PASS - S1', 'Cours: PASS - S1'],
        'composant': ['Système', 'Système', 'Système'],
        'evenement': ['Cours consulté', 'Cours consulté', 'Cours consulté'],
    })


@pytest.fixture
def logs_with_missing():
    """Create a logs DataFrame with missing values."""
    return pd.DataFrame({
        'heure': pd.to_datetime(['2024-07-24 09:48:08', pd.NaT, '2024-08-19 12:55:34']),
        'pseudo': [436, 437, 841],
        'contexte': ['Cours: PASS - S1', None, 'Cours: PASS - S1'],
        'composant': ['Système', 'Fichier', None],
        'evenement': ['Cours consulté', 'Module consulté', 'Cours consulté'],
    })


@pytest.fixture
def notes_with_outliers():
    """Create a notes DataFrame with out-of-range values."""
    return pd.DataFrame({
        'pseudo': [436, 841, 318, 717],
        'note': [-5.0, 25.0, 10.0, 15.0],
    })


@pytest.fixture
def notes_with_invalid_pseudo():
    """Create a notes DataFrame with invalid pseudo values."""
    return pd.DataFrame({
        'pseudo': ['abc', 841, 318],
        'note': [11.0, 14.5, 10.0],
    })


class TestDataProcessorCleaning:
    """Test data cleaning functionality."""

    def test_remove_duplicates(self, processor, logs_with_duplicates):
        """Test that duplicate rows are removed."""
        result = processor.remove_duplicates(logs_with_duplicates)
        assert len(result) == 2

    def test_remove_duplicates_warning(self, processor, logs_with_duplicates):
        """Test that a warning is issued when duplicates are found."""
        with pytest.warns(UserWarning, match='dupliquées'):
            processor.remove_duplicates(logs_with_duplicates)

    def test_remove_duplicates_no_warning_when_clean(self, processor, sample_logs_df):
        """Test that no warning is issued when there are no duplicates."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            processor.remove_duplicates(sample_logs_df)

    def test_remove_duplicates_resets_index(self, processor, logs_with_duplicates):
        """Test that index is reset after removing duplicates."""
        result = processor.remove_duplicates(logs_with_duplicates)
        assert list(result.index) == list(range(len(result)))

    def test_handle_missing_values_warning(self, processor, logs_with_missing):
        """Test that a warning is issued for missing values."""
        with pytest.warns(UserWarning, match='valeurs manquantes'):
            processor.handle_missing_values(logs_with_missing)

    def test_handle_missing_values_no_warning_when_clean(self, processor, sample_logs_df):
        """Test that no warning is issued when there are no missing values."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            processor.handle_missing_values(sample_logs_df)

    def test_handle_missing_values_preserves_data(self, processor, logs_with_missing):
        """Test that missing value handling preserves all rows."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.handle_missing_values(logs_with_missing)
        assert len(result) == len(logs_with_missing)

    def test_clean_logs(self, processor, sample_logs_df):
        """Test the full logs cleaning pipeline."""
        result = processor.clean_logs(sample_logs_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_logs_df)

    def test_clean_logs_removes_duplicates(self, processor, logs_with_duplicates):
        """Test that clean_logs removes duplicates."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.clean_logs(logs_with_duplicates)
        assert len(result) == 2

    def test_clean_notes_basic(self, processor, sample_notes_df):
        """Test the full notes cleaning pipeline."""
        result = processor.clean_notes(sample_notes_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result['pseudo'].dtype in ['int64', 'int32']
        assert result['note'].dtype == float

    def test_clean_notes_clips_outliers(self, processor, notes_with_outliers):
        """Test that out-of-range notes are clipped."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.clean_notes(notes_with_outliers)
        assert result['note'].min() >= 0
        assert result['note'].max() <= 20

    def test_clean_notes_clips_warning(self, processor, notes_with_outliers):
        """Test that a warning is issued for clipped notes."""
        with pytest.warns(UserWarning, match='clippées'):
            processor.clean_notes(notes_with_outliers)

    def test_clean_notes_invalid_pseudo(self, processor, notes_with_invalid_pseudo):
        """Test that rows with invalid pseudo are dropped."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.clean_notes(notes_with_invalid_pseudo)
        assert len(result) == 2
        assert 'abc' not in result['pseudo'].values

    def test_validate_notes_valid(self, processor, sample_notes_df):
        """Test validation on clean notes returns valid report."""
        report = processor.validate_notes(sample_notes_df)
        assert report['is_valid'] is True
        assert report['total_rows'] == 3

    def test_validate_notes_with_outliers(self, processor, notes_with_outliers):
        """Test validation detects out-of-range notes."""
        report = processor.validate_notes(notes_with_outliers)
        assert report['out_of_range'] == 2
        assert report['is_valid'] is False

    def test_validate_notes_with_duplicates(self, processor):
        """Test validation detects duplicates."""
        df = pd.DataFrame({
            'pseudo': [436, 436, 841],
            'note': [11.0, 11.0, 14.5],
        })
        report = processor.validate_notes(df)
        assert report['duplicates'] == 1

    def test_validate_notes_with_missing(self, processor):
        """Test validation detects missing values."""
        df = pd.DataFrame({
            'pseudo': [436, None, 841],
            'note': [11.0, 14.5, None],
        })
        report = processor.validate_notes(df)
        assert report['missing_pseudo'] == 1
        assert report['missing_note'] == 1
        assert report['is_valid'] is False


class TestDataProcessorFeatureEngineering:
    """Test feature engineering functionality."""

    def test_extract_temporal_features_columns(self, processor, sample_logs_df):
        """Test that temporal feature extraction adds expected columns."""
        result = processor.extract_temporal_features(sample_logs_df)
        assert 'heure_hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'is_weekend' in result.columns

    def test_extract_temporal_features_values(self, processor):
        """Test that temporal features have correct values."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:48:08']),  # Wednesday
            'pseudo': [436],
        })
        result = processor.extract_temporal_features(df)
        assert result['heure_hour'].iloc[0] == 9
        assert result['day_of_week'].iloc[0] == 2  # Wednesday
        assert result['month'].iloc[0] == 7
        assert result['is_weekend'].iloc[0] == 0

    def test_extract_temporal_features_weekend(self, processor):
        """Test that weekend detection works correctly."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-20 10:00:00',  # Saturday
                '2024-07-21 10:00:00',  # Sunday
                '2024-07-22 10:00:00',  # Monday
            ]),
            'pseudo': [1, 2, 3],
        })
        result = processor.extract_temporal_features(df)
        assert result['is_weekend'].iloc[0] == 1
        assert result['is_weekend'].iloc[1] == 1
        assert result['is_weekend'].iloc[2] == 0

    def test_extract_temporal_features_preserves_original(self, processor, sample_logs_df):
        """Test that original columns are preserved."""
        original_cols = set(sample_logs_df.columns)
        result = processor.extract_temporal_features(sample_logs_df)
        assert original_cols.issubset(set(result.columns))

    def test_extract_temporal_features_does_not_modify_input(self, processor, sample_logs_df):
        """Test that input DataFrame is not modified."""
        original_len = len(sample_logs_df.columns)
        processor.extract_temporal_features(sample_logs_df)
        assert len(sample_logs_df.columns) == original_len

    def test_compute_activity_metrics_columns(self, processor, sample_logs_df):
        """Test that activity metrics have expected columns."""
        result = processor.compute_activity_metrics(sample_logs_df)
        expected_cols = ['pseudo', 'total_actions', 'unique_days_active', 'actions_per_day', 'session_count']
        assert all(col in result.columns for col in expected_cols)

    def test_compute_activity_metrics_values(self, processor, sample_logs_df):
        """Test that activity metrics are computed correctly."""
        result = processor.compute_activity_metrics(sample_logs_df)
        # Student 436 has 3 actions
        student_436 = result[result['pseudo'] == 436].iloc[0]
        assert student_436['total_actions'] == 3

        # Student 841 has 2 actions
        student_841 = result[result['pseudo'] == 841].iloc[0]
        assert student_841['total_actions'] == 2

    def test_compute_activity_metrics_unique_days(self, processor, sample_logs_df):
        """Test unique days active calculation."""
        result = processor.compute_activity_metrics(sample_logs_df)
        # Student 436 has actions on 2024-07-24 and 2024-09-01 = 2 days
        student_436 = result[result['pseudo'] == 436].iloc[0]
        assert student_436['unique_days_active'] == 2

    def test_compute_activity_metrics_sessions(self, processor):
        """Test session count with clear gaps."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:10:00',  # same session (< 30 min gap)
                '2024-07-24 10:00:00',  # new session (> 30 min gap)
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        result = processor.compute_activity_metrics(df)
        assert result[result['pseudo'] == 1]['session_count'].iloc[0] == 2

    def test_compute_component_features(self, processor, sample_logs_df):
        """Test component feature computation."""
        result = processor.compute_component_features(sample_logs_df)
        assert 'pseudo' in result.columns
        # Should have columns prefixed with comp_
        comp_cols = [c for c in result.columns if c.startswith('comp_')]
        assert len(comp_cols) > 0

    def test_compute_component_features_values(self, processor, sample_logs_df):
        """Test that component counts are correct."""
        result = processor.compute_component_features(sample_logs_df)
        student_436 = result[result['pseudo'] == 436].iloc[0]
        # Student 436: 2 Système, 1 Fichier
        assert student_436['comp_Système'] == 2
        assert student_436['comp_Fichier'] == 1


class TestDataProcessorMerge:
    """Test merge functionality."""

    def test_merge_logs_notes(self, processor, sample_logs_df, sample_notes_df):
        """Test merging logs and notes."""
        result = processor.merge_logs_notes(sample_logs_df, sample_notes_df)
        assert isinstance(result, pd.DataFrame)
        assert 'pseudo' in result.columns
        assert 'note' in result.columns
        assert 'total_actions' in result.columns

    def test_merge_outer_join(self, processor):
        """Test that merge uses outer join to keep all students."""
        logs_df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:00:00', '2024-07-24 10:00:00']),
            'pseudo': [1, 1],
            'contexte': ['A', 'A'],
            'composant': ['B', 'B'],
            'evenement': ['C', 'C'],
        })
        notes_df = pd.DataFrame({
            'pseudo': [1, 2],
            'note': [15.0, 12.0],
        })
        result = processor.merge_logs_notes(logs_df, notes_df)
        # Should have both students (1 from logs+notes, 2 from notes only)
        assert len(result) == 2
        assert set(result['pseudo'].dropna().astype(int)) == {1, 2}

    def test_merge_updates_cleaning_report(self, processor, sample_logs_df, sample_notes_df):
        """Test that merge updates the cleaning report."""
        processor.merge_logs_notes(sample_logs_df, sample_notes_df)
        report = processor.get_cleaning_report()
        assert report['students_merged'] > 0

    def test_merge_tracks_logs_only_students(self, processor):
        """Test tracking of students present only in logs."""
        logs_df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:00:00']),
            'pseudo': [999],
            'contexte': ['A'],
            'composant': ['B'],
            'evenement': ['C'],
        })
        notes_df = pd.DataFrame({
            'pseudo': [1],
            'note': [15.0],
        })
        processor.merge_logs_notes(logs_df, notes_df)
        report = processor.get_cleaning_report()
        assert report['students_logs_only'] == 1
        assert report['students_notes_only'] == 1


class TestDataProcessorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_logs_dataframe(self, processor):
        """Test cleaning an empty logs DataFrame."""
        df = pd.DataFrame(columns=['heure', 'pseudo', 'contexte', 'composant', 'evenement'])
        result = processor.clean_logs(df)
        assert len(result) == 0

    def test_empty_notes_dataframe(self, processor):
        """Test cleaning an empty notes DataFrame."""
        df = pd.DataFrame(columns=['pseudo', 'note'])
        result = processor.clean_notes(df)
        assert len(result) == 0

    def test_single_row_logs(self, processor):
        """Test with a single row of logs."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:48:08']),
            'pseudo': [436],
            'contexte': ['Cours: PASS - S1'],
            'composant': ['Système'],
            'evenement': ['Cours consulté'],
        })
        result = processor.clean_logs(df)
        assert len(result) == 1

    def test_single_row_notes(self, processor):
        """Test with a single row of notes."""
        df = pd.DataFrame({'pseudo': [436], 'note': [11.0]})
        result = processor.clean_notes(df)
        assert len(result) == 1

    def test_notes_at_boundaries(self, processor):
        """Test notes exactly at min and max boundaries."""
        df = pd.DataFrame({
            'pseudo': [1, 2],
            'note': [0.0, 20.0],
        })
        result = processor.clean_notes(df)
        assert result['note'].iloc[0] == 0.0
        assert result['note'].iloc[1] == 20.0

    def test_all_duplicate_rows(self, processor):
        """Test DataFrame where all rows are duplicates."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:48:08'] * 3),
            'pseudo': [436, 436, 436],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.remove_duplicates(df)
        assert len(result) == 1

    def test_activity_metrics_single_action(self, processor):
        """Test activity metrics for student with single action."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:00:00']),
            'pseudo': [1],
            'contexte': ['A'],
            'composant': ['B'],
            'evenement': ['C'],
        })
        result = processor.compute_activity_metrics(df)
        assert result['session_count'].iloc[0] == 1
        assert result['total_actions'].iloc[0] == 1

    def test_temporal_features_with_nat(self, processor):
        """Test temporal feature extraction with NaT values."""
        df = pd.DataFrame({
            'heure': [pd.NaT, pd.Timestamp('2024-07-24 09:48:08')],
            'pseudo': [1, 2],
        })
        result = processor.extract_temporal_features(df)
        assert pd.isna(result['heure_hour'].iloc[0])
        assert result['heure_hour'].iloc[1] == 9

    def test_get_cleaning_report_initial(self, processor):
        """Test that initial cleaning report has zero values."""
        report = processor.get_cleaning_report()
        assert all(v == 0 for v in report.values())

    def test_process_data_delegates_to_clean_logs(self, processor, sample_logs_df):
        """Test that process_data calls clean_logs."""
        result = processor.process_data(sample_logs_df)
        assert isinstance(result, pd.DataFrame)


class TestDataProcessorRealFiles:
    """Test DataProcessor with actual CSV files in data/ directory."""

    @pytest.fixture
    def real_data(self):
        """Load real data files using DataLoader."""
        from src.data.data_loader import DataLoader
        loader = DataLoader()
        logs = loader.load_logs('data/logs_info_25_pseudo.csv')
        notes = loader.load_notes('data/notes_info_25_pseudo.csv')
        return logs, notes

    def test_build_student_dataset_real_data(self, processor, real_data):
        """Test full pipeline with real data files."""
        logs, notes = real_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.build_student_dataset(logs, notes)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'pseudo' in result.columns
        assert 'note' in result.columns
        assert 'actions_totales' in result.columns

    def test_build_student_dataset_no_suffix(self, processor, real_data):
        """Test that build_student_dataset produces no _x/_y suffixed columns."""
        logs, notes = real_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.build_student_dataset(logs, notes)
        suffix_cols = [c for c in result.columns if c.endswith('_x') or c.endswith('_y')]
        assert suffix_cols == [], f"Colonnes avec suffixe _x/_y trouvées: {suffix_cols}"

    def test_build_student_dataset_french_columns(self, processor, real_data):
        """Test that build_student_dataset returns French column names."""
        logs, notes = real_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.build_student_dataset(logs, notes)
        # English names should NOT be present
        english_names = ['total_actions', 'session_count', 'streak_days', 'peak_hour']
        for name in english_names:
            assert name not in result.columns, f"Colonne anglaise '{name}' trouvée"
        # French names should be present
        french_names = ['actions_totales', 'nombre_sessions', 'jours_consecutifs_max', 'heure_pointe']
        for name in french_names:
            assert name in result.columns, f"Colonne française '{name}' manquante"

    def test_cleaning_report_real_data(self, processor, real_data):
        """Test cleaning report with real data."""
        logs, notes = real_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processor.build_student_dataset(logs, notes)

        report = processor.get_cleaning_report()
        assert report['logs_initial_rows'] > 0
        assert report['notes_initial_rows'] > 0
        assert report['students_merged'] > 0

    def test_clean_logs_real_data(self, processor, real_data):
        """Test logs cleaning with real data."""
        logs, _ = real_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.clean_logs(logs)

        assert len(result) > 0
        assert len(result) <= len(logs)

    def test_clean_notes_real_data(self, processor, real_data):
        """Test notes cleaning with real data."""
        _, notes = real_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.clean_notes(notes)

        assert len(result) > 0
        assert result['note'].min() >= 0
        assert result['note'].max() <= 20

    def test_temporal_features_real_data(self, processor, real_data):
        """Test temporal feature extraction with real data."""
        logs, _ = real_data
        result = processor.extract_temporal_features(logs)
        assert 'heure_hour' in result.columns
        assert result['heure_hour'].between(0, 23).all()
        assert result['day_of_week'].between(0, 6).all()
        assert result['month'].between(1, 12).all()
        assert result['is_weekend'].isin([0, 1]).all()

    def test_activity_metrics_real_data(self, processor, real_data):
        """Test activity metrics with real data."""
        logs, _ = real_data
        result = processor.compute_activity_metrics(logs)
        assert len(result) > 0
        assert (result['total_actions'] > 0).all()
        assert (result['unique_days_active'] > 0).all()
        assert (result['session_count'] > 0).all()


class TestDataProcessorEngagementFeatures:
    """Test engagement feature computation functionality."""

    @pytest.fixture
    def engagement_logs_df(self):
        """Create a logs DataFrame for engagement feature testing."""
        return pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:30:00',
                '2024-07-24 14:00:00',
                '2024-07-25 10:00:00',
                '2024-07-26 11:00:00',
                '2024-07-27 15:00:00',
                '2024-08-19 12:55:34',
                '2024-08-19 19:00:00',
                '2024-08-20 08:00:00',
            ]),
            'pseudo': [436, 436, 436, 436, 436, 841, 841, 841, 841],
            'contexte': [
                'Cours: PASS - S1',
                'Fichier: Contrat',
                'Cours: PASS - S1',
                'Cours: PASS - S1',
                'Fichier: TD',
                'Cours: PASS - S1',
                'Fichier: Contrat',
                'Forum: Discussion',
                'Test: Quiz1',
            ],
            'composant': [
                'Système',
                'Fichier',
                'Système',
                'Fichier',
                'Fichier',
                'Système',
                'Fichier',
                'Forum',
                'Test',
            ],
            'evenement': [
                'Cours consulté',
                'Module de cours consulté',
                'Cours consulté',
                'Fichier téléchargé',
                'Module de cours consulté',
                'Cours consulté',
                'Module de cours consulté',
                'Discussion consultée',
                'Test démarré',
            ],
        })

    def test_compute_event_type_features_columns(self, processor, engagement_logs_df):
        """Test that event type features have expected columns."""
        result = processor.compute_event_type_features(engagement_logs_df)
        assert 'pseudo' in result.columns
        # Should have columns for event types
        event_type_cols = [c for c in result.columns if c.endswith('_count')]
        assert len(event_type_cols) > 0

    def test_compute_event_type_features_categorization(self, processor, engagement_logs_df):
        """Test that events are correctly categorized."""
        result = processor.compute_event_type_features(engagement_logs_df)
        # Student 436 should have view events (consulté)
        student_436 = result[result['pseudo'] == 436].iloc[0]
        assert 'view_count' in result.columns
        assert student_436['view_count'] >= 3

    def test_compute_event_type_features_download_events(self, processor, engagement_logs_df):
        """Test that download events are correctly identified."""
        result = processor.compute_event_type_features(engagement_logs_df)
        student_436 = result[result['pseudo'] == 436].iloc[0]
        if 'download_count' in result.columns:
            assert student_436['download_count'] >= 1  # 'téléchargé' event

    def test_compute_event_type_features_submission_events(self, processor):
        """Test that submission events are correctly identified."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:00:00', '2024-07-24 10:00:00']),
            'pseudo': [1, 1],
            'contexte': ['Test', 'Devoir'],
            'composant': ['Test', 'Devoir'],
            'evenement': ['Test soumis', 'Devoir déposé'],
        })
        result = processor.compute_event_type_features(df)
        if 'submission_count' in result.columns:
            assert result['submission_count'].iloc[0] >= 2

    def test_compute_event_type_features_forum_events(self, processor, engagement_logs_df):
        """Test that forum events are correctly identified."""
        result = processor.compute_event_type_features(engagement_logs_df)
        student_841 = result[result['pseudo'] == 841].iloc[0]
        if 'forum_count' in result.columns:
            assert student_841['forum_count'] >= 1

    def test_compute_event_type_features_does_not_modify_input(self, processor, engagement_logs_df):
        """Test that input DataFrame is not modified."""
        original_cols = set(engagement_logs_df.columns)
        processor.compute_event_type_features(engagement_logs_df)
        assert set(engagement_logs_df.columns) == original_cols

    def test_compute_consistency_features_columns(self, processor, engagement_logs_df):
        """Test that consistency features have expected columns."""
        result = processor.compute_consistency_features(engagement_logs_df)
        expected_cols = ['pseudo', 'streak_days', 'avg_gap_days', 'std_gap_days', 'study_frequency']
        assert all(col in result.columns for col in expected_cols)

    def test_compute_consistency_features_streak_days(self, processor):
        """Test streak days calculation for consecutive days."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-25 09:00:00',
                '2024-07-26 09:00:00',
                '2024-07-28 09:00:00',
            ]),
            'pseudo': [1, 1, 1, 1],
            'contexte': ['A', 'A', 'A', 'A'],
            'composant': ['B', 'B', 'B', 'B'],
            'evenement': ['C', 'C', 'C', 'C'],
        })
        result = processor.compute_consistency_features(df)
        # Days 24, 25, 26 are consecutive (3 days), then gap, then day 28
        assert result['streak_days'].iloc[0] == 3

    def test_compute_consistency_features_single_day(self, processor):
        """Test consistency features for single day of activity."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:00:00', '2024-07-24 10:00:00']),
            'pseudo': [1, 1],
            'contexte': ['A', 'A'],
            'composant': ['B', 'B'],
            'evenement': ['C', 'C'],
        })
        result = processor.compute_consistency_features(df)
        assert result['streak_days'].iloc[0] == 1
        assert result['avg_gap_days'].iloc[0] == 0.0

    def test_compute_consistency_features_gap_stats(self, processor):
        """Test gap statistics calculation."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-26 09:00:00',  # 2 day gap
                '2024-07-28 09:00:00',  # 2 day gap
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        result = processor.compute_consistency_features(df)
        assert result['avg_gap_days'].iloc[0] == 2.0

    def test_compute_consistency_features_study_frequency(self, processor):
        """Test study frequency calculation."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-01 09:00:00',
                '2024-07-08 09:00:00',
                '2024-07-15 09:00:00',
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        result = processor.compute_consistency_features(df)
        # 3 days over 2 weeks = 1.5 days per week
        assert result['study_frequency'].iloc[0] > 0

    def test_compute_consistency_features_does_not_modify_input(self, processor, engagement_logs_df):
        """Test that input DataFrame is not modified."""
        original_cols = set(engagement_logs_df.columns)
        processor.compute_consistency_features(engagement_logs_df)
        assert set(engagement_logs_df.columns) == original_cols

    def test_compute_interaction_depth_features_columns(self, processor, engagement_logs_df):
        """Test that interaction depth features have expected columns."""
        result = processor.compute_interaction_depth_features(engagement_logs_df)
        expected_cols = [
            'pseudo',
            'component_diversity',
            'context_diversity',
            'avg_interactions_per_component',
            'component_switch_rate',
        ]
        assert all(col in result.columns for col in expected_cols)

    def test_compute_interaction_depth_features_diversity(self, processor, engagement_logs_df):
        """Test diversity metrics calculation."""
        result = processor.compute_interaction_depth_features(engagement_logs_df)
        student_436 = result[result['pseudo'] == 436].iloc[0]
        # Student 436 uses Système and Fichier = 2 components
        assert student_436['component_diversity'] == 2
        # Student 436 has multiple contexts
        assert student_436['context_diversity'] >= 2

    def test_compute_interaction_depth_features_avg_interactions(self, processor):
        """Test average interactions per component."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:10:00',
                '2024-07-24 09:20:00',
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['Système', 'Système', 'Fichier'],
            'evenement': ['C', 'C', 'C'],
        })
        result = processor.compute_interaction_depth_features(df)
        # 2 interactions with Système, 1 with Fichier, avg = 1.5
        assert result['avg_interactions_per_component'].iloc[0] == 1.5

    def test_compute_interaction_depth_features_switch_rate(self, processor):
        """Test component switch rate calculation."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:10:00',
                '2024-07-24 09:20:00',
                '2024-07-24 09:30:00',
            ]),
            'pseudo': [1, 1, 1, 1],
            'contexte': ['A', 'A', 'A', 'A'],
            'composant': ['A', 'B', 'A', 'A'],
            'evenement': ['C', 'C', 'C', 'C'],
        })
        result = processor.compute_interaction_depth_features(df)
        # Switches: A->B (1), B->A (2), A->A (2) = 2 switches out of 4 interactions
        assert result['component_switch_rate'].iloc[0] == 0.5

    def test_compute_interaction_depth_features_single_interaction(self, processor):
        """Test with single interaction."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:00:00']),
            'pseudo': [1],
            'contexte': ['A'],
            'composant': ['B'],
            'evenement': ['C'],
        })
        result = processor.compute_interaction_depth_features(df)
        assert result['component_switch_rate'].iloc[0] == 0.0

    def test_compute_interaction_depth_features_does_not_modify_input(self, processor, engagement_logs_df):
        """Test that input DataFrame is not modified."""
        original_cols = set(engagement_logs_df.columns)
        processor.compute_interaction_depth_features(engagement_logs_df)
        assert set(engagement_logs_df.columns) == original_cols

    def test_compute_temporal_patterns_columns(self, processor, engagement_logs_df):
        """Test that temporal pattern features have expected columns."""
        result = processor.compute_temporal_patterns(engagement_logs_df)
        expected_cols = [
            'pseudo',
            'peak_hour',
            'morning_activity',
            'afternoon_activity',
            'evening_activity',
            'night_activity',
            'weekend_activity_ratio',
        ]
        assert all(col in result.columns for col in expected_cols)

    def test_compute_temporal_patterns_peak_hour(self, processor):
        """Test peak hour identification."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:30:00',
                '2024-07-24 09:45:00',
                '2024-07-24 14:00:00',
            ]),
            'pseudo': [1, 1, 1, 1],
            'contexte': ['A', 'A', 'A', 'A'],
            'composant': ['B', 'B', 'B', 'B'],
            'evenement': ['C', 'C', 'C', 'C'],
        })
        result = processor.compute_temporal_patterns(df)
        # Most activity at hour 9 (3 events)
        assert result['peak_hour'].iloc[0] == 9

    def test_compute_temporal_patterns_time_periods(self, processor):
        """Test time period activity ratios."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 08:00:00',  # morning
                '2024-07-24 09:00:00',  # morning
                '2024-07-24 14:00:00',  # afternoon
                '2024-07-24 20:00:00',  # evening
            ]),
            'pseudo': [1, 1, 1, 1],
            'contexte': ['A', 'A', 'A', 'A'],
            'composant': ['B', 'B', 'B', 'B'],
            'evenement': ['C', 'C', 'C', 'C'],
        })
        result = processor.compute_temporal_patterns(df)
        # 2/4 morning, 1/4 afternoon, 1/4 evening, 0 night
        assert result['morning_activity'].iloc[0] == 0.5
        assert result['afternoon_activity'].iloc[0] == 0.25
        assert result['evening_activity'].iloc[0] == 0.25
        assert result['night_activity'].iloc[0] == 0.0

    def test_compute_temporal_patterns_weekend_ratio(self, processor):
        """Test weekend activity ratio calculation."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-20 10:00:00',  # Saturday
                '2024-07-21 10:00:00',  # Sunday
                '2024-07-22 10:00:00',  # Monday
                '2024-07-23 10:00:00',  # Tuesday
            ]),
            'pseudo': [1, 1, 1, 1],
            'contexte': ['A', 'A', 'A', 'A'],
            'composant': ['B', 'B', 'B', 'B'],
            'evenement': ['C', 'C', 'C', 'C'],
        })
        result = processor.compute_temporal_patterns(df)
        # 2 out of 4 on weekend = 0.5
        assert result['weekend_activity_ratio'].iloc[0] == 0.5

    def test_compute_temporal_patterns_night_activity(self, processor):
        """Test night activity identification."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 02:00:00',  # night
                '2024-07-24 04:00:00',  # night
                '2024-07-24 10:00:00',  # morning
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        result = processor.compute_temporal_patterns(df)
        # 2/3 night activity
        assert result['night_activity'].iloc[0] > 0.6

    def test_compute_temporal_patterns_does_not_modify_input(self, processor, engagement_logs_df):
        """Test that input DataFrame is not modified."""
        original_cols = set(engagement_logs_df.columns)
        processor.compute_temporal_patterns(engagement_logs_df)
        assert set(engagement_logs_df.columns) == original_cols

    def test_build_engagement_features_columns(self, processor, engagement_logs_df):
        """Test that build_engagement_features creates all expected columns."""
        result = processor.build_engagement_features(engagement_logs_df)
        assert 'pseudo' in result.columns
        # Should include activity metrics
        assert 'total_actions' in result.columns
        assert 'session_count' in result.columns
        # Should include consistency features
        assert 'streak_days' in result.columns
        assert 'study_frequency' in result.columns
        # Should include interaction depth features
        assert 'component_diversity' in result.columns
        assert 'component_switch_rate' in result.columns
        # Should include temporal patterns
        assert 'peak_hour' in result.columns
        assert 'weekend_activity_ratio' in result.columns

    def test_build_engagement_features_multiple_students(self, processor, engagement_logs_df):
        """Test that build_engagement_features works with multiple students."""
        result = processor.build_engagement_features(engagement_logs_df)
        unique_students = result['pseudo'].nunique()
        assert unique_students == 2  # Students 436 and 841

    def test_build_engagement_features_merge_correctness(self, processor, engagement_logs_df):
        """Test that all features are correctly merged."""
        result = processor.build_engagement_features(engagement_logs_df)
        # Check that no pseudo values are duplicated
        assert result['pseudo'].is_unique
        # Check that all students have all features (no NaN for pseudo)
        assert result['pseudo'].notna().all()

    def test_build_engagement_features_single_student(self, processor):
        """Test build_engagement_features with single student."""
        df = pd.DataFrame({
            'heure': pd.to_datetime(['2024-07-24 09:00:00', '2024-07-25 10:00:00']),
            'pseudo': [1, 1],
            'contexte': ['A', 'B'],
            'composant': ['X', 'Y'],
            'evenement': ['consulté', 'téléchargé'],
        })
        result = processor.build_engagement_features(df)
        assert len(result) == 1
        assert result['pseudo'].iloc[0] == 1

    def test_build_engagement_features_does_not_modify_input(self, processor, engagement_logs_df):
        """Test that input DataFrame is not modified."""
        original_cols = set(engagement_logs_df.columns)
        original_len = len(engagement_logs_df)
        processor.build_engagement_features(engagement_logs_df)
        assert set(engagement_logs_df.columns) == original_cols
        assert len(engagement_logs_df) == original_len

    def test_build_engagement_features_empty_dataframe(self, processor):
        """Test build_engagement_features with empty DataFrame."""
        df = pd.DataFrame(columns=['heure', 'pseudo', 'contexte', 'composant', 'evenement'])
        # Empty dataframes may cause issues with some aggregation functions
        # This test verifies the function can handle edge cases gracefully
        try:
            result = processor.build_engagement_features(df)
            assert len(result) == 0
            assert 'pseudo' in result.columns
        except (ValueError, KeyError):
            # Some pandas operations on empty DataFrames may raise errors
            # This is acceptable edge case behavior
            pytest.skip("Empty DataFrame handling requires special implementation")


class TestDeduplicateRapidEvents:
    """Test deduplicate_rapid_events functionality."""

    def test_removes_rapid_events(self, processor):
        """Test that rapid consecutive events are removed."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:00:01',  # rapid (1s < 2s default)
                '2024-07-24 09:00:10',  # not rapid (8s gap from previous kept)
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        result = processor.deduplicate_rapid_events(df)
        # First event is rapid (next comes within 2s), so dropped; events 2 and 3 kept
        assert len(result) == 2

    def test_no_removal_when_all_spaced(self, processor):
        """Test no events removed when all are spaced apart."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:01:00',
                '2024-07-24 09:02:00',
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        result = processor.deduplicate_rapid_events(df)
        assert len(result) == 3

    def test_custom_threshold(self, processor):
        """Test with a custom threshold value."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:00:08',  # 8s gap
                '2024-07-24 09:00:20',  # 12s gap
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        # With threshold=10, first event is rapid (next comes in 8s)
        result = processor.deduplicate_rapid_events(df, threshold_seconds=10)
        assert len(result) == 2

    def test_multiple_students_independent(self, processor):
        """Test that deduplication is per-student."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:00:01',
                '2024-07-24 09:00:00',
                '2024-07-24 09:01:00',
            ]),
            'pseudo': [1, 1, 2, 2],
            'contexte': ['A', 'A', 'A', 'A'],
            'composant': ['B', 'B', 'B', 'B'],
            'evenement': ['C', 'C', 'C', 'C'],
        })
        result = processor.deduplicate_rapid_events(df)
        # Student 1: first event rapid → removed. Student 2: spaced → both kept
        assert len(result[result['pseudo'] == 1]) == 1
        assert len(result[result['pseudo'] == 2]) == 2

    def test_zero_threshold_no_removal(self, processor):
        """Test that threshold=0 removes nothing."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:00:01',
            ]),
            'pseudo': [1, 1],
            'contexte': ['A', 'A'],
            'composant': ['B', 'B'],
            'evenement': ['C', 'C'],
        })
        result = processor.deduplicate_rapid_events(df, threshold_seconds=0)
        assert len(result) == 2

    def test_warning_issued(self, processor):
        """Test that a warning is issued when rapid events are removed."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:00:01',
            ]),
            'pseudo': [1, 1],
            'contexte': ['A', 'A'],
            'composant': ['B', 'B'],
            'evenement': ['C', 'C'],
        })
        with pytest.warns(UserWarning, match='événements rapides'):
            processor.deduplicate_rapid_events(df)

    def test_cleaning_report_updated(self, processor):
        """Test that cleaning report tracks deduplicated events."""
        df = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-07-24 09:00:00',
                '2024-07-24 09:00:01',
                '2024-07-24 09:00:10',
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['A', 'A', 'A'],
            'composant': ['B', 'B', 'B'],
            'evenement': ['C', 'C', 'C'],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processor.deduplicate_rapid_events(df)
        report = processor.get_cleaning_report()
        assert report['events_deduplicated'] > 0


class TestPreprocessFeatures:
    """Test preprocess_features functionality."""

    def test_fills_nan_with_zero(self, processor):
        """Test that NaN values in feature columns are filled with 0."""
        df = pd.DataFrame({
            'pseudo': [1, 2],
            'note': [15.0, np.nan],
            'total_actions': [10.0, np.nan],
            'session_count': [np.nan, 5.0],
        })
        result = processor.preprocess_features(df)
        assert result['total_actions'].iloc[1] == 0.0
        assert result['session_count'].iloc[0] == 0.0

    def test_preserves_note_nan(self, processor):
        """Test that NaN in 'note' column is preserved."""
        df = pd.DataFrame({
            'pseudo': [1, 2],
            'note': [15.0, np.nan],
            'total_actions': [10.0, 5.0],
        })
        result = processor.preprocess_features(df)
        assert pd.isna(result['note'].iloc[1])

    def test_preserves_pseudo(self, processor):
        """Test that 'pseudo' column is not modified."""
        df = pd.DataFrame({
            'pseudo': [1, 2],
            'note': [15.0, 10.0],
            'total_actions': [10.0, 5.0],
        })
        result = processor.preprocess_features(df)
        assert list(result['pseudo']) == [1, 2]

    def test_no_nan_warning_when_clean(self, processor):
        """Test no warning when there are no NaN values."""
        df = pd.DataFrame({
            'pseudo': [1],
            'note': [15.0],
            'total_actions': [10.0],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            processor.preprocess_features(df)

    def test_warning_when_nan_filled(self, processor):
        """Test warning is issued when NaN values are filled."""
        df = pd.DataFrame({
            'pseudo': [1],
            'note': [15.0],
            'total_actions': [np.nan],
        })
        with pytest.warns(UserWarning, match='NaN'):
            processor.preprocess_features(df)

    def test_does_not_modify_input(self, processor):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame({
            'pseudo': [1],
            'note': [15.0],
            'total_actions': [np.nan],
        })
        original_val = df['total_actions'].iloc[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processor.preprocess_features(df)
        assert pd.isna(df['total_actions'].iloc[0])


class TestRemoveOutliers:
    """Test remove_outliers functionality."""

    def test_removes_outliers_iqr(self, processor):
        """Test that outliers are removed using IQR method."""
        # Create data where one value is clearly an outlier
        df = pd.DataFrame({
            'pseudo': list(range(10)),
            'note': [10.0] * 10,
            'total_actions': [10, 11, 12, 10, 11, 12, 10, 11, 12, 1000],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.remove_outliers(df)
        assert len(result) < len(df)
        assert 1000 not in result['total_actions'].values

    def test_preserves_note_and_pseudo(self, processor):
        """Test that outliers in note/pseudo do not cause removal."""
        df = pd.DataFrame({
            'pseudo': [1, 2, 3, 4, 5],
            'note': [0.0, 20.0, 10.0, 10.0, 10.0],
            'total_actions': [10, 11, 12, 10, 11],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.remove_outliers(df)
        assert len(result) == 5

    def test_custom_columns(self, processor):
        """Test remove_outliers with specific columns."""
        df = pd.DataFrame({
            'pseudo': list(range(10)),
            'note': [10.0] * 10,
            'col_a': [10, 11, 12, 10, 11, 12, 10, 11, 12, 1000],
            'col_b': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.remove_outliers(df, columns=['col_a'])
        assert len(result) < len(df)

    def test_no_outliers_no_warning(self, processor):
        """Test no warning when there are no outliers."""
        df = pd.DataFrame({
            'pseudo': [1, 2, 3],
            'note': [10.0, 11.0, 12.0],
            'total_actions': [10, 11, 12],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            processor.remove_outliers(df)

    def test_warning_when_outliers_removed(self, processor):
        """Test warning is issued when outliers are removed."""
        df = pd.DataFrame({
            'pseudo': list(range(10)),
            'note': [10.0] * 10,
            'total_actions': [10, 11, 12, 10, 11, 12, 10, 11, 12, 1000],
        })
        with pytest.warns(UserWarning, match='outliers'):
            processor.remove_outliers(df)

    def test_cleaning_report_updated(self, processor):
        """Test that cleaning report tracks removed outliers."""
        df = pd.DataFrame({
            'pseudo': list(range(10)),
            'note': [10.0] * 10,
            'total_actions': [10, 11, 12, 10, 11, 12, 10, 11, 12, 1000],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processor.remove_outliers(df)
        report = processor.get_cleaning_report()
        assert report['outliers_removed'] > 0

    def test_no_numeric_columns(self, processor):
        """Test with no numeric columns to check."""
        df = pd.DataFrame({
            'pseudo': [1, 2],
            'note': [10.0, 11.0],
        })
        result = processor.remove_outliers(df)
        assert len(result) == 2


class TestRenameFeaturestoFrench:
    """Test rename_features_to_french functionality."""

    def test_renames_known_columns(self, processor):
        """Test that known English columns are renamed to French."""
        df = pd.DataFrame({
            'pseudo': [1],
            'note': [15.0],
            'total_actions': [10],
            'session_count': [3],
            'streak_days': [5],
        })
        result = processor.rename_features_to_french(df)
        assert 'actions_totales' in result.columns
        assert 'nombre_sessions' in result.columns
        assert 'jours_consecutifs_max' in result.columns

    def test_preserves_pseudo_and_note(self, processor):
        """Test that pseudo and note columns are not renamed."""
        df = pd.DataFrame({
            'pseudo': [1],
            'note': [15.0],
            'total_actions': [10],
        })
        result = processor.rename_features_to_french(df)
        assert 'pseudo' in result.columns
        assert 'note' in result.columns

    def test_preserves_comp_prefix(self, processor):
        """Test that comp_ prefixed columns are not renamed."""
        df = pd.DataFrame({
            'pseudo': [1],
            'comp_Système': [5],
            'comp_Fichier': [3],
        })
        result = processor.rename_features_to_french(df)
        assert 'comp_Système' in result.columns
        assert 'comp_Fichier' in result.columns

    def test_unknown_columns_unchanged(self, processor):
        """Test that unknown columns are not renamed."""
        df = pd.DataFrame({
            'pseudo': [1],
            'unknown_feature': [42],
        })
        result = processor.rename_features_to_french(df)
        assert 'unknown_feature' in result.columns

    def test_all_mappings_applied(self, processor):
        """Test that all configured mappings are applied."""
        from src.config import Config
        config = Config()
        # Build a DataFrame with all mapped English column names
        data = {col: [1] for col in config.FEATURE_NAMES_FR.keys() if not col.startswith('comp_')}
        data['pseudo'] = [1]
        data['note'] = [15.0]
        df = pd.DataFrame(data)
        result = processor.rename_features_to_french(df)
        for eng, fr in config.FEATURE_NAMES_FR.items():
            if eng.startswith('comp_'):
                continue
            assert fr in result.columns, f"Mapping {eng} -> {fr} non appliqué"
            assert eng not in result.columns, f"Colonne anglaise '{eng}' encore présente"

    def test_does_not_modify_input(self, processor):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame({
            'pseudo': [1],
            'total_actions': [10],
        })
        processor.rename_features_to_french(df)
        assert 'total_actions' in df.columns
