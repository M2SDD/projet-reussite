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
        from src.data_loader import DataLoader
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
        assert 'total_actions' in result.columns

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
