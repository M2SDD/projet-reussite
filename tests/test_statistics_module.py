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
    # Tests will be implemented in subtask-2-2
    pass


class TestDetectOutliers:
    """Test detect_outliers functionality."""
    # Tests will be implemented in subtask-2-3
    pass


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
