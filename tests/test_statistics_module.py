"""
Unit tests for the StatisticsModule class.

Tests cover:
- Summary statistics computation
- Outlier detection using IQR method
- Distribution characterization (skewness, kurtosis)
- Data summary metadata
- Report generation
- Edge cases and warnings
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from src.statistics_module import StatisticsModule


@pytest.fixture
def stats_module():
    """Create a StatisticsModule instance for testing."""
    return StatisticsModule()


@pytest.fixture
def sample_numeric_df():
    """Create a sample DataFrame with numeric columns for testing."""
    return pd.DataFrame({
        'score': [85, 90, 78, 92, 88, 76, 95, 82, 89, 91],
        'age': [20, 21, 19, 22, 20, 21, 23, 19, 20, 22],
        'hours_studied': [5, 8, 3, 9, 6, 4, 10, 5, 7, 8],
    })


@pytest.fixture
def sample_mixed_df():
    """Create a sample DataFrame with mixed data types for testing."""
    return pd.DataFrame({
        'pseudo': [101, 102, 103, 104, 105],
        'note': [12.5, 15.0, 10.5, 18.0, 14.5],
        'nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'heure': pd.to_datetime([
            '2024-01-15 10:00:00',
            '2024-01-16 11:30:00',
            '2024-01-17 14:00:00',
            '2024-01-18 09:15:00',
            '2024-01-19 16:45:00',
        ]),
    })


@pytest.fixture
def df_with_outliers():
    """Create a DataFrame with known outliers for testing."""
    return pd.DataFrame({
        'values': [10, 12, 11, 13, 12, 14, 11, 10, 100, 13, 12, 11],
        'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    })


@pytest.fixture
def df_without_outliers():
    """Create a DataFrame without outliers for testing."""
    return pd.DataFrame({
        'values': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    })


@pytest.fixture
def skewed_df():
    """Create a DataFrame with skewed distribution for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'right_skewed': np.concatenate([
            np.random.normal(10, 2, 80),
            np.random.normal(25, 3, 20)
        ]),
    })


@pytest.fixture
def normal_df():
    """Create a DataFrame with normal distribution for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'normal_dist': np.random.normal(50, 10, 1000),
    })


@pytest.fixture
def empty_df():
    """Create an empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def single_row_df():
    """Create a DataFrame with a single row for testing edge cases."""
    return pd.DataFrame({
        'value': [42],
    })


@pytest.fixture
def df_with_missing():
    """Create a DataFrame with missing values for testing."""
    return pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [10, np.nan, 30, 40, np.nan],
    })


@pytest.fixture
def non_numeric_df():
    """Create a DataFrame with only non-numeric columns for testing."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'city': ['Paris', 'Lyon', 'Marseille'],
    })


class TestStatisticsModuleInit:
    """Test StatisticsModule initialization."""

    def test_init_without_config(self, stats_module):
        """Test that StatisticsModule initializes without config."""
        assert stats_module is not None
        assert stats_module.config is not None

    def test_init_with_config(self):
        """Test that StatisticsModule can be initialized with a custom config."""
        from src.config import Config
        custom_config = Config()
        sm = StatisticsModule(config=custom_config)
        assert sm.config is custom_config


class TestComputeSummaryStatistics:
    """Test compute_summary_statistics functionality."""

    def test_compute_summary_statistics_basic(self, stats_module, sample_numeric_df):
        """Test that summary statistics are computed correctly for numeric columns."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        # Verify structure
        assert isinstance(result, dict)
        expected_keys = ['count', 'mean', 'median', 'std', 'variance', 'min', 'max', 'range', 'q25', 'q50', 'q75']
        assert set(result.keys()) == set(expected_keys)

        # Verify all numeric columns are present
        for key in expected_keys:
            assert 'score' in result[key]
            assert 'age' in result[key]
            assert 'hours_studied' in result[key]

    def test_compute_summary_statistics_count(self, stats_module, sample_numeric_df):
        """Test that count statistic is computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        assert result['count']['score'] == 10
        assert result['count']['age'] == 10
        assert result['count']['hours_studied'] == 10

    def test_compute_summary_statistics_mean(self, stats_module, sample_numeric_df):
        """Test that mean is computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        expected_mean_score = np.mean([85, 90, 78, 92, 88, 76, 95, 82, 89, 91])
        assert result['mean']['score'] == pytest.approx(expected_mean_score)

    def test_compute_summary_statistics_median(self, stats_module, sample_numeric_df):
        """Test that median is computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        expected_median_score = np.median([85, 90, 78, 92, 88, 76, 95, 82, 89, 91])
        assert result['median']['score'] == pytest.approx(expected_median_score)

    def test_compute_summary_statistics_std(self, stats_module, sample_numeric_df):
        """Test that standard deviation is computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        expected_std_score = np.std([85, 90, 78, 92, 88, 76, 95, 82, 89, 91], ddof=1)
        assert result['std']['score'] == pytest.approx(expected_std_score)

    def test_compute_summary_statistics_variance(self, stats_module, sample_numeric_df):
        """Test that variance is computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        expected_var_score = np.var([85, 90, 78, 92, 88, 76, 95, 82, 89, 91], ddof=1)
        assert result['variance']['score'] == pytest.approx(expected_var_score)

    def test_compute_summary_statistics_min_max(self, stats_module, sample_numeric_df):
        """Test that min and max are computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        assert result['min']['score'] == 76
        assert result['max']['score'] == 95

    def test_compute_summary_statistics_range(self, stats_module, sample_numeric_df):
        """Test that range is computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        assert result['range']['score'] == 19  # 95 - 76

    def test_compute_summary_statistics_quantiles(self, stats_module, sample_numeric_df):
        """Test that quantiles are computed correctly."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        scores = [85, 90, 78, 92, 88, 76, 95, 82, 89, 91]
        expected_q25 = np.percentile(scores, 25)
        expected_q50 = np.percentile(scores, 50)
        expected_q75 = np.percentile(scores, 75)

        assert result['q25']['score'] == pytest.approx(expected_q25)
        assert result['q50']['score'] == pytest.approx(expected_q50)
        assert result['q75']['score'] == pytest.approx(expected_q75)

    def test_compute_summary_statistics_mixed_df(self, stats_module, sample_mixed_df):
        """Test that only numeric columns are processed in mixed DataFrame."""
        result = stats_module.compute_summary_statistics(sample_mixed_df)

        # Only numeric columns should be present
        assert 'pseudo' in result['mean']
        assert 'note' in result['mean']
        assert 'nom' not in result['mean']
        assert 'heure' not in result['mean']

    def test_compute_summary_statistics_empty_df(self, stats_module, empty_df):
        """Test that empty DataFrame returns empty dict with warning."""
        with pytest.warns(UserWarning, match='Aucune colonne numérique'):
            result = stats_module.compute_summary_statistics(empty_df)

        assert result == {}

    def test_compute_summary_statistics_non_numeric_df(self, stats_module, non_numeric_df):
        """Test that DataFrame with no numeric columns returns empty dict with warning."""
        with pytest.warns(UserWarning, match='Aucune colonne numérique'):
            result = stats_module.compute_summary_statistics(non_numeric_df)

        assert result == {}

    def test_compute_summary_statistics_single_row(self, stats_module, single_row_df):
        """Test that single row DataFrame computes statistics correctly."""
        result = stats_module.compute_summary_statistics(single_row_df)

        assert result['count']['value'] == 1
        assert result['mean']['value'] == 42
        assert result['median']['value'] == 42
        assert result['min']['value'] == 42
        assert result['max']['value'] == 42
        assert result['range']['value'] == 0
        # Std and variance should be NaN for single value
        assert np.isnan(result['std']['value'])

    def test_compute_summary_statistics_with_missing(self, stats_module, df_with_missing):
        """Test that missing values are handled correctly (excluded from calculations)."""
        result = stats_module.compute_summary_statistics(df_with_missing)

        # col1 has 4 non-NaN values: [1, 2, 4, 5]
        assert result['count']['col1'] == 4
        assert result['mean']['col1'] == pytest.approx(3.0)

        # col2 has 3 non-NaN values: [10, 30, 40]
        assert result['count']['col2'] == 3
        assert result['mean']['col2'] == pytest.approx(26.666667, rel=1e-5)

    def test_compute_summary_statistics_preserves_column_names(self, stats_module, sample_numeric_df):
        """Test that original column names are preserved in results."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        expected_columns = set(sample_numeric_df.columns)
        actual_columns = set(result['mean'].keys())

        assert expected_columns == actual_columns

    def test_compute_summary_statistics_no_side_effects(self, stats_module, sample_numeric_df):
        """Test that computing statistics does not modify the original DataFrame."""
        original_df = sample_numeric_df.copy()
        stats_module.compute_summary_statistics(sample_numeric_df)

        pd.testing.assert_frame_equal(sample_numeric_df, original_df)

    def test_compute_summary_statistics_returns_dict(self, stats_module, sample_numeric_df):
        """Test that the function returns a dictionary."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        assert isinstance(result, dict)

    def test_compute_summary_statistics_all_stats_same_length(self, stats_module, sample_numeric_df):
        """Test that all statistics have the same number of columns."""
        result = stats_module.compute_summary_statistics(sample_numeric_df)

        stat_lengths = [len(result[stat]) for stat in result.keys()]
        assert len(set(stat_lengths)) == 1  # All should be the same length


class TestDetectOutliers:
    """Test detect_outliers functionality."""

    def test_detect_outliers_basic(self, stats_module, df_with_outliers):
        """Test that outliers are detected correctly using IQR method."""
        result = stats_module.detect_outliers(df_with_outliers)

        # Verify structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df_with_outliers.shape
        assert list(result.columns) == list(df_with_outliers.columns)

        # Verify that the value 100 in 'values' column is detected as outlier
        outlier_indices = result[result['values'] == True].index
        assert len(outlier_indices) > 0
        assert 100 in df_with_outliers.loc[outlier_indices, 'values'].values

    def test_detect_outliers_known_outlier(self, stats_module, df_with_outliers):
        """Test that the known outlier (100) is correctly identified."""
        result = stats_module.detect_outliers(df_with_outliers)

        # The value 100 should be identified as an outlier
        values_column = df_with_outliers['values']
        outlier_mask = result['values']

        # Find where the value is 100
        idx_100 = values_column[values_column == 100].index[0]
        assert outlier_mask.loc[idx_100] == True

    def test_detect_outliers_no_outliers(self, stats_module, df_without_outliers):
        """Test that DataFrame without outliers returns all False."""
        result = stats_module.detect_outliers(df_without_outliers)

        # All values should be False
        assert result['values'].sum() == 0

    def test_detect_outliers_normal_column(self, stats_module, df_with_outliers):
        """Test that column without outliers returns all False."""
        result = stats_module.detect_outliers(df_with_outliers)

        # The 'normal' column has sequential values 1-12, should have no outliers
        assert result['normal'].sum() == 0

    def test_detect_outliers_mixed_df(self, stats_module, sample_mixed_df):
        """Test that only numeric columns are processed in mixed DataFrame."""
        result = stats_module.detect_outliers(sample_mixed_df)

        # Result should have the same shape
        assert result.shape == sample_mixed_df.shape

        # Non-numeric columns should be all False
        assert result['nom'].sum() == 0
        assert result['heure'].sum() == 0

        # Numeric columns should be analyzed
        assert 'pseudo' in result.columns
        assert 'note' in result.columns

    def test_detect_outliers_empty_df(self, stats_module, empty_df):
        """Test that empty DataFrame returns empty DataFrame with warning."""
        with pytest.warns(UserWarning, match='Aucune colonne numérique'):
            result = stats_module.detect_outliers(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_detect_outliers_non_numeric_df(self, stats_module, non_numeric_df):
        """Test that DataFrame with no numeric columns returns all False with warning."""
        with pytest.warns(UserWarning, match='Aucune colonne numérique'):
            result = stats_module.detect_outliers(non_numeric_df)

        # Should return DataFrame of same shape with all False
        assert result.shape == non_numeric_df.shape
        assert result.all(axis=None) == False

    def test_detect_outliers_single_row(self, stats_module, single_row_df):
        """Test that single row DataFrame returns no outliers."""
        result = stats_module.detect_outliers(single_row_df)

        # Single value cannot be an outlier (Q1 = Q3 = value, IQR = 0)
        assert result['value'].sum() == 0

    def test_detect_outliers_with_missing(self, stats_module, df_with_missing):
        """Test that missing values are handled correctly (not flagged as outliers)."""
        result = stats_module.detect_outliers(df_with_missing)

        # Result should have same shape
        assert result.shape == df_with_missing.shape

        # NaN values should not be flagged as outliers (should be False or NaN)
        # Check that the result has boolean or NaN values
        for col in result.columns:
            assert result[col].dtype == bool or pd.api.types.is_bool_dtype(result[col])

    def test_detect_outliers_no_side_effects(self, stats_module, df_with_outliers):
        """Test that detecting outliers does not modify the original DataFrame."""
        original_df = df_with_outliers.copy()
        stats_module.detect_outliers(df_with_outliers)

        pd.testing.assert_frame_equal(df_with_outliers, original_df)

    def test_detect_outliers_returns_dataframe(self, stats_module, sample_numeric_df):
        """Test that the function returns a DataFrame."""
        result = stats_module.detect_outliers(sample_numeric_df)

        assert isinstance(result, pd.DataFrame)

    def test_detect_outliers_boolean_values(self, stats_module, df_with_outliers):
        """Test that result contains only boolean values."""
        result = stats_module.detect_outliers(df_with_outliers)

        # All columns should be boolean type
        for col in result.columns:
            assert result[col].dtype == bool

    def test_detect_outliers_same_index(self, stats_module, df_with_outliers):
        """Test that result preserves the same index as input."""
        result = stats_module.detect_outliers(df_with_outliers)

        pd.testing.assert_index_equal(result.index, df_with_outliers.index)

    def test_detect_outliers_same_columns(self, stats_module, sample_numeric_df):
        """Test that result preserves the same columns as input."""
        result = stats_module.detect_outliers(sample_numeric_df)

        pd.testing.assert_index_equal(result.columns, sample_numeric_df.columns)

    def test_detect_outliers_iqr_calculation(self, stats_module):
        """Test that IQR method is correctly applied."""
        # Create a DataFrame where we know the IQR bounds
        # Values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        # Q1 = 2.75, Q3 = 7.25, IQR = 4.5
        # Lower bound = 2.75 - 1.5*4.5 = -4.0
        # Upper bound = 7.25 + 1.5*4.5 = 14.0
        # So 100 should be an outlier
        df = pd.DataFrame({
            'test_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })

        result = stats_module.detect_outliers(df)

        # Only the value 100 should be flagged as outlier
        assert result['test_col'].sum() == 1
        assert result['test_col'].iloc[-1] == True  # Last value (100) is outlier

    def test_detect_outliers_multiple_outliers(self, stats_module):
        """Test detection of multiple outliers."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 100, 200],
        })

        result = stats_module.detect_outliers(df)

        # Both 100 and 200 should be outliers
        assert result['values'].sum() == 2

    def test_detect_outliers_negative_outlier(self, stats_module):
        """Test detection of negative outliers (below lower bound)."""
        df = pd.DataFrame({
            'values': [-100, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        })

        result = stats_module.detect_outliers(df)

        # -100 should be detected as outlier
        assert result['values'].iloc[0] == True
        assert result['values'].sum() >= 1

    def test_detect_outliers_uniform_distribution(self, stats_module):
        """Test that uniform distribution has no outliers."""
        df = pd.DataFrame({
            'values': list(range(1, 21)),  # 1 to 20
        })

        result = stats_module.detect_outliers(df)

        # Uniform distribution should have no outliers
        assert result['values'].sum() == 0

    def test_detect_outliers_all_same_values(self, stats_module):
        """Test that all identical values produce no outliers."""
        df = pd.DataFrame({
            'values': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        })

        result = stats_module.detect_outliers(df)

        # IQR = 0, so no values can be outliers
        assert result['values'].sum() == 0


class TestCharacterizeDistribution:
    """Test characterize_distribution functionality."""
    # Tests will be implemented in subtask-2-4
    pass


class TestComputeDataSummary:
    """Test compute_data_summary functionality."""
    # Tests will be implemented in subtask-2-5
    pass


class TestGenerateReport:
    """Test generate_report functionality."""
    # Tests will be implemented in subtask-2-6
    pass
