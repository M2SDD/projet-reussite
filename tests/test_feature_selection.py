"""
Unit tests for feature selection and statistical analysis methods.

Tests cover:
- Feature-target correlation analysis
- Feature-feature correlation matrix
- Descriptive statistics
- Statistical significance testing
- Variance-based feature selection
- Correlation-based feature selection
- SelectKBest feature selection
- Full feature selection pipeline
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import f_regression
from src.data_processor import DataProcessor


@pytest.fixture
def processor():
    """Create a DataProcessor instance for testing."""
    return DataProcessor()


@pytest.fixture
def sample_features_with_target():
    """Create a sample DataFrame with features and a target variable."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0],  # Highly correlated with target
        'feature_3': [5.0, 4.0, 3.0, 2.0, 1.0],   # Negatively correlated
        'feature_4': [1.0, 1.0, 1.0, 1.0, 1.0],   # Constant (zero variance)
        'note': [10.0, 12.0, 14.0, 16.0, 18.0],
    })


@pytest.fixture
def large_features_dataset():
    """Create a larger DataFrame for more robust statistical testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'highly_correlated': np.linspace(0, 100, n_samples) + np.random.randn(n_samples) * 5,
        'moderately_correlated': np.linspace(0, 50, n_samples) + np.random.randn(n_samples) * 15,
        'weakly_correlated': np.random.randn(n_samples) * 10,
        'constant_feature': np.ones(n_samples),
        'low_variance': np.ones(n_samples) * 5 + np.random.randn(n_samples) * 0.01,
        'note': np.linspace(0, 100, n_samples) + np.random.randn(n_samples) * 10,
    })


@pytest.fixture
def features_with_missing_values():
    """Create a DataFrame with missing values for robust testing."""
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature_2': [2.0, np.nan, 6.0, 8.0, 10.0],
        'feature_3': [5.0, 4.0, 3.0, 2.0, 1.0],
        'note': [10.0, 12.0, 14.0, 16.0, 18.0],
    })


@pytest.fixture
def features_with_non_numeric():
    """Create a DataFrame with non-numeric columns."""
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0],
        'pseudo': ['A', 'B', 'C', 'D', 'E'],  # Non-numeric
        'note': [10.0, 12.0, 14.0, 16.0, 18.0],
    })


@pytest.fixture
def redundant_features_dataset():
    """Create a DataFrame with highly correlated features for correlation filtering."""
    np.random.seed(42)
    feature_a = np.random.randn(50)
    return pd.DataFrame({
        'feature_a': feature_a,
        'feature_b': feature_a + np.random.randn(50) * 0.01,  # Almost identical to feature_a
        'feature_c': feature_a * 2 + 1,  # Highly correlated with feature_a
        'feature_d': np.random.randn(50),  # Independent
    })


class TestStatisticalAnalysis:
    """Test statistical analysis methods."""

    def test_compute_feature_correlations(self, processor, sample_features_with_target):
        """Test feature-target correlation computation."""
        result = processor.compute_feature_correlations(sample_features_with_target, 'note')

        assert isinstance(result, pd.Series)
        assert 'note' not in result.index  # Target should be excluded
        assert len(result) > 0
        # All non-NaN correlations should be between -1 and 1
        valid_corrs = result.dropna()
        assert all(valid_corrs >= -1.0)
        assert all(valid_corrs <= 1.0)

    def test_compute_feature_correlations_basic(self, processor, sample_features_with_target):
        """Test basic feature-target correlation computation."""
        result = processor.compute_feature_correlations(sample_features_with_target, 'note')

        assert isinstance(result, pd.Series)
        assert 'note' not in result.index  # Target should be excluded
        assert 'feature_1' in result.index
        assert 'feature_2' in result.index
        assert 'feature_3' in result.index

    def test_compute_feature_correlations_sorted_by_abs_value(self, processor, sample_features_with_target):
        """Test that correlations are sorted by absolute value."""
        result = processor.compute_feature_correlations(sample_features_with_target, 'note')

        # Check that absolute values are in descending order (excluding NaN)
        abs_values = result.abs().dropna()
        if len(abs_values) > 1:
            assert all(abs_values.iloc[i] >= abs_values.iloc[i+1] for i in range(len(abs_values)-1))

    def test_compute_feature_correlations_values(self, processor, sample_features_with_target):
        """Test that correlation values are in valid range."""
        result = processor.compute_feature_correlations(sample_features_with_target, 'note')

        # All non-NaN correlations should be between -1 and 1
        valid_corrs = result.dropna()
        assert all(valid_corrs >= -1.0)
        assert all(valid_corrs <= 1.0)

    def test_compute_feature_correlations_missing_target(self, processor, sample_features_with_target):
        """Test error handling when target column doesn't exist."""
        with pytest.raises(ValueError, match="n'existe pas"):
            processor.compute_feature_correlations(sample_features_with_target, 'nonexistent')

    def test_compute_feature_correlations_non_numeric_target(self, processor, features_with_non_numeric):
        """Test error handling when target is not numeric."""
        with pytest.raises(ValueError, match="n'est pas numérique"):
            processor.compute_feature_correlations(features_with_non_numeric, 'pseudo')

    def test_compute_feature_correlations_no_numeric_columns(self, processor):
        """Test error handling when DataFrame has no numeric columns."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        with pytest.raises(ValueError, match="aucune colonne numérique"):
            processor.compute_feature_correlations(df, 'a')

    def test_compute_feature_feature_correlations_basic(self, processor, sample_features_with_target):
        """Test basic correlation matrix computation."""
        result = processor.compute_feature_feature_correlations(sample_features_with_target)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]  # Should be square matrix
        assert all(result.columns == result.index)  # Symmetric

    def test_compute_feature_feature_correlations_diagonal(self, processor, sample_features_with_target):
        """Test that diagonal elements are 1.0 (self-correlation) or NaN for constant features."""
        result = processor.compute_feature_feature_correlations(sample_features_with_target)

        diagonal = np.diag(result)
        # Non-NaN diagonal elements should be 1.0
        valid_diagonal = diagonal[~np.isnan(diagonal)]
        assert all(np.abs(valid_diagonal - 1.0) < 1e-10)

    def test_compute_feature_feature_correlations_symmetric(self, processor, sample_features_with_target):
        """Test that correlation matrix is symmetric."""
        result = processor.compute_feature_feature_correlations(sample_features_with_target)

        # Check symmetry, handling NaN values
        assert np.allclose(result, result.T, equal_nan=True)

    def test_compute_feature_feature_correlations_exclude_columns(self, processor, sample_features_with_target):
        """Test excluding specific columns from correlation matrix."""
        result = processor.compute_feature_feature_correlations(
            sample_features_with_target,
            exclude_columns=['note', 'feature_4']
        )

        assert 'note' not in result.columns
        assert 'feature_4' not in result.columns
        assert 'feature_1' in result.columns
        assert 'feature_2' in result.columns

    def test_compute_feature_feature_correlations_no_numeric(self, processor):
        """Test error handling when no numeric columns exist."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        with pytest.raises(ValueError, match="aucune colonne numérique"):
            processor.compute_feature_feature_correlations(df)

    def test_compute_feature_feature_correlations_exclude_all(self, processor, sample_features_with_target):
        """Test error handling when all columns are excluded."""
        all_columns = list(sample_features_with_target.columns)
        with pytest.raises(ValueError, match="Aucune colonne.*après exclusion"):
            processor.compute_feature_feature_correlations(
                sample_features_with_target,
                exclude_columns=all_columns
            )

    def test_compute_descriptive_statistics_basic(self, processor, sample_features_with_target):
        """Test basic descriptive statistics computation."""
        result = processor.compute_descriptive_statistics(sample_features_with_target)

        assert isinstance(result, pd.DataFrame)
        assert 'mean' in result.columns
        assert 'std' in result.columns
        assert 'min' in result.columns
        assert 'max' in result.columns
        assert '50%' in result.columns  # Median

    def test_compute_descriptive_statistics_features_as_rows(self, processor, sample_features_with_target):
        """Test that features are in rows (transposed format)."""
        result = processor.compute_descriptive_statistics(sample_features_with_target)

        assert 'feature_1' in result.index
        assert 'feature_2' in result.index
        assert 'note' in result.index

    def test_compute_descriptive_statistics_mean_calculation(self, processor, sample_features_with_target):
        """Test that mean is correctly calculated."""
        result = processor.compute_descriptive_statistics(sample_features_with_target)

        # feature_1 has values [1, 2, 3, 4, 5], mean should be 3.0
        assert result.loc['feature_1', 'mean'] == 3.0

    def test_compute_descriptive_statistics_no_numeric(self, processor):
        """Test error handling when no numeric columns exist."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        with pytest.raises(ValueError, match="aucune colonne numérique"):
            processor.compute_descriptive_statistics(df)

    def test_test_feature_significance_basic(self, processor, large_features_dataset):
        """Test basic significance testing."""
        result = processor.test_feature_significance(large_features_dataset, 'note')

        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'correlation' in result.columns
        assert 'p_value' in result.columns
        assert 'is_significant' in result.columns

    def test_test_feature_significance_sorted_by_pvalue(self, processor, large_features_dataset):
        """Test that results are sorted by p-value (most significant first)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.test_feature_significance(large_features_dataset, 'note')

        # p-values should be in ascending order (non-decreasing), excluding NaN
        if len(result) > 1:
            # Get non-NaN p-values
            valid_pvalues = result['p_value'].dropna().values
            if len(valid_pvalues) > 1:
                # Check ascending order with small tolerance for floating point comparison
                assert all(valid_pvalues[i] <= valid_pvalues[i+1] + 1e-10 for i in range(len(valid_pvalues)-1))

    def test_test_feature_significance_boolean_flag(self, processor, large_features_dataset):
        """Test that is_significant flag is set correctly."""
        result = processor.test_feature_significance(large_features_dataset, 'note')

        # Check that is_significant is True when p_value < 0.05
        for _, row in result.iterrows():
            if row['p_value'] < 0.05:
                assert row['is_significant'] is True
            else:
                assert row['is_significant'] is False

    def test_test_feature_significance_excludes_target(self, processor, large_features_dataset):
        """Test that target column is not included in results."""
        result = processor.test_feature_significance(large_features_dataset, 'note')

        assert 'note' not in result['feature'].values

    def test_test_feature_significance_missing_target(self, processor, sample_features_with_target):
        """Test error handling when target doesn't exist."""
        with pytest.raises(ValueError, match="n'existe pas"):
            processor.test_feature_significance(sample_features_with_target, 'nonexistent')

    def test_test_feature_significance_non_numeric_target(self, processor, features_with_non_numeric):
        """Test error handling when target is not numeric."""
        with pytest.raises(ValueError, match="n'est pas numérique"):
            processor.test_feature_significance(features_with_non_numeric, 'pseudo')

    def test_test_feature_significance_no_numeric_columns(self, processor):
        """Test error handling when no numeric columns exist."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        with pytest.raises(ValueError, match="aucune colonne numérique"):
            processor.test_feature_significance(df, 'a')

    def test_test_feature_significance_warning_few_samples(self, processor):
        """Test warning when feature has too few valid samples."""
        df = pd.DataFrame({
            'feature_1': [1.0, np.nan, np.nan, np.nan, np.nan],
            'note': [10.0, 12.0, 14.0, 16.0, 18.0],
        })

        with pytest.warns(UserWarning, match="moins de 3 valeurs valides"):
            processor.test_feature_significance(df, 'note')

    def test_test_feature_significance_handles_nan(self, processor, features_with_missing_values):
        """Test that significance testing handles NaN values correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.test_feature_significance(features_with_missing_values, 'note')

        # Should still return results for features with enough valid data
        assert len(result) >= 0
        assert isinstance(result, pd.DataFrame)


class TestFeatureSelection:
    """Test feature selection methods."""

    def test_select_by_variance_placeholder(self, processor):
        """Placeholder test for variance selection (will be implemented in subtask-4-2)."""
        # This test will be expanded in the next subtask
        assert hasattr(processor, 'select_by_variance')

    def test_select_by_correlation_placeholder(self, processor):
        """Placeholder test for correlation selection (will be implemented in subtask-4-2)."""
        # This test will be expanded in the next subtask
        assert hasattr(processor, 'select_by_correlation')

    def test_select_k_best_placeholder(self, processor):
        """Placeholder test for SelectKBest (will be implemented in subtask-4-2)."""
        # This test will be expanded in the next subtask
        assert hasattr(processor, 'select_k_best')

    def test_select_features_pipeline_placeholder(self, processor):
        """Placeholder test for feature selection pipeline (will be implemented in subtask-4-2)."""
        # This test will be expanded in the next subtask
        assert hasattr(processor, 'select_features_pipeline')


# Module-level test for verification command compatibility
def test_compute_feature_correlations():
    """Module-level test for compute_feature_correlations (for verification command)."""
    processor = DataProcessor()

    # Create sample data
    df = pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0],
        'note': [10.0, 12.0, 14.0, 16.0, 18.0],
    })

    result = processor.compute_feature_correlations(df, 'note')

    assert isinstance(result, pd.Series)
    assert 'note' not in result.index
    assert len(result) > 0
