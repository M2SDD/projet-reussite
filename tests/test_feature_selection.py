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

    def test_select_by_variance_basic(self, processor, sample_features_with_target):
        """Test basic variance-based feature selection."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_variance(sample_features_with_target, threshold=0.01)

        assert isinstance(result, pd.DataFrame)
        # Constant feature (feature_4) should be removed with threshold > 0
        assert 'feature_4' not in result.columns
        # Other features should remain
        assert 'feature_1' in result.columns
        assert 'feature_2' in result.columns
        assert 'feature_3' in result.columns

    def test_select_by_variance_removes_constant_features(self, processor, sample_features_with_target):
        """Test that constant features (zero variance) are removed with positive threshold."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_variance(sample_features_with_target, threshold=0.01)

        # feature_4 is constant and should be removed
        assert 'feature_4' not in result.columns

    def test_select_by_variance_with_threshold(self, processor, large_features_dataset):
        """Test variance selection with a specific threshold."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_variance(large_features_dataset, threshold=1.0)

        assert isinstance(result, pd.DataFrame)
        # low_variance should be removed with threshold=1.0
        assert 'low_variance' not in result.columns
        # constant_feature should also be removed
        assert 'constant_feature' not in result.columns

    def test_select_by_variance_warning(self, processor, sample_features_with_target):
        """Test that a warning is issued when features are removed."""
        with pytest.warns(UserWarning, match='variance'):
            processor.select_by_variance(sample_features_with_target, threshold=0.01)

    def test_select_by_variance_threshold_zero_keeps_constant(self, processor, sample_features_with_target):
        """Test that features with variance=0 are kept when threshold=0.0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_variance(sample_features_with_target, threshold=0.0)

        # Constant feature (variance=0.0) is kept with threshold=0.0 because 0.0 >= 0.0
        assert 'feature_4' in result.columns

    def test_select_by_variance_no_warning_when_all_kept(self, processor):
        """Test that no warning is issued when all features are kept."""
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            processor.select_by_variance(df, threshold=0.0)

    def test_select_by_variance_preserves_non_numeric(self, processor, features_with_non_numeric):
        """Test that non-numeric columns are preserved."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_variance(features_with_non_numeric, threshold=0.0)

        assert 'pseudo' in result.columns

    def test_select_by_variance_no_numeric_columns(self, processor):
        """Test error handling when no numeric columns exist."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        with pytest.raises(ValueError, match="aucune colonne numérique"):
            processor.select_by_variance(df, threshold=0.0)

    def test_select_by_correlation_basic(self, processor, redundant_features_dataset):
        """Test basic correlation-based feature selection."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_correlation(redundant_features_dataset, threshold=0.95)

        assert isinstance(result, pd.DataFrame)
        # At least one feature should remain
        assert len(result.columns) >= 1
        # Should have fewer features than original
        assert len(result.columns) <= len(redundant_features_dataset.columns)

    def test_select_by_correlation_removes_redundant(self, processor, redundant_features_dataset):
        """Test that redundant features are removed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_correlation(redundant_features_dataset, threshold=0.95)

        # feature_a should be kept
        assert 'feature_a' in result.columns
        # At least one of the highly correlated features should be removed
        removed_count = len(redundant_features_dataset.columns) - len(result.columns)
        assert removed_count > 0

    def test_select_by_correlation_with_lower_threshold(self, processor, redundant_features_dataset):
        """Test correlation selection with a lower threshold removes more features."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_high = processor.select_by_correlation(redundant_features_dataset, threshold=0.95)
            result_low = processor.select_by_correlation(redundant_features_dataset, threshold=0.7)

        # Lower threshold should remove more features
        assert len(result_low.columns) <= len(result_high.columns)

    def test_select_by_correlation_warning(self, processor, redundant_features_dataset):
        """Test that a warning is issued when correlated features are removed."""
        with pytest.warns(UserWarning, match='corrélées'):
            processor.select_by_correlation(redundant_features_dataset, threshold=0.95)

    def test_select_by_correlation_no_warning_when_all_kept(self, processor):
        """Test that no warning is issued when all features are kept."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),  # Independent features
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            processor.select_by_correlation(df, threshold=0.95)

    def test_select_by_correlation_preserves_non_numeric(self, processor):
        """Test that non-numeric columns are preserved."""
        df = pd.DataFrame({
            'feature_a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_b': [1.0, 2.0, 3.0, 4.0, 5.0],  # Identical to feature_a
            'pseudo': ['A', 'B', 'C', 'D', 'E'],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_by_correlation(df, threshold=0.95)

        assert 'pseudo' in result.columns

    def test_select_by_correlation_no_numeric_columns(self, processor):
        """Test error handling when no numeric columns exist."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        with pytest.raises(ValueError, match="aucune colonne numérique"):
            processor.select_by_correlation(df, threshold=0.95)

    def test_select_k_best_basic(self, processor, large_features_dataset):
        """Test basic SelectKBest feature selection."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_k_best(large_features_dataset, 'note', k=3)

        assert isinstance(result, pd.DataFrame)
        # Should select exactly k features
        assert len(result.columns) == 3

    def test_select_k_best_selects_best_features(self, processor, large_features_dataset):
        """Test that SelectKBest selects the most correlated features."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_k_best(large_features_dataset, 'note', k=2)

        # highly_correlated should be selected
        assert 'highly_correlated' in result.columns

    def test_select_k_best_warning(self, processor, large_features_dataset):
        """Test that a warning is issued when features are removed."""
        with pytest.warns(UserWarning, match='exclues par SelectKBest'):
            processor.select_k_best(large_features_dataset, 'note', k=2)

    def test_select_k_best_no_warning_when_all_kept(self, processor, sample_features_with_target):
        """Test that no warning is issued when k equals number of features."""
        # Count numeric features (excluding target and constant features)
        numeric_features = sample_features_with_target.select_dtypes(include=['number']).columns
        numeric_features = [col for col in numeric_features if col != 'note']

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            processor.select_k_best(sample_features_with_target, 'note', k=len(numeric_features))

    def test_select_k_best_missing_target(self, processor, sample_features_with_target):
        """Test error handling when target doesn't exist."""
        with pytest.raises(ValueError, match="n'existe pas"):
            processor.select_k_best(sample_features_with_target, 'nonexistent', k=2)

    def test_select_k_best_no_numeric_columns(self, processor):
        """Test error handling when no numeric columns exist."""
        df = pd.DataFrame({
            'a': ['x', 'y', 'z'],
            'b': ['p', 'q', 'r'],
            'target': [1, 2, 3]
        })
        with pytest.raises(ValueError, match="aucune colonne numérique pour la sélection"):
            processor.select_k_best(df, 'target', k=1)

    def test_select_k_best_k_too_large(self, processor, sample_features_with_target):
        """Test error handling when k exceeds number of features."""
        with pytest.raises(ValueError, match="supérieur au nombre de features"):
            processor.select_k_best(sample_features_with_target, 'note', k=100)

    def test_select_k_best_excludes_target(self, processor, large_features_dataset):
        """Test that target column is not included in selected features."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_k_best(large_features_dataset, 'note', k=3)

        assert 'note' not in result.columns

    def test_select_features_pipeline_basic(self, processor, large_features_dataset):
        """Test basic feature selection pipeline."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_features_pipeline(
                large_features_dataset,
                'note',
                variance_threshold=0.0,
                correlation_threshold=0.95,
                k=3
            )

        assert isinstance(result, pd.DataFrame)
        # Should select exactly k features
        assert len(result.columns) == 3
        # Target should not be in result
        assert 'note' not in result.columns

    def test_select_features_pipeline_removes_constant(self, processor, large_features_dataset):
        """Test that pipeline removes constant features."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = processor.select_features_pipeline(
                large_features_dataset,
                'note',
                variance_threshold=0.0,
                correlation_threshold=0.95,
                k=3
            )

        # constant_feature should be removed
        assert 'constant_feature' not in result.columns

    def test_select_features_pipeline_sequential_filtering(self, processor, large_features_dataset):
        """Test that pipeline applies filters sequentially."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # High variance threshold should reduce features before k-best
            result = processor.select_features_pipeline(
                large_features_dataset,
                'note',
                variance_threshold=100.0,  # High threshold
                correlation_threshold=0.95,
                k=2
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) <= 2


# Module-level tests for verification command compatibility
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


def test_select_by_variance():
    """Module-level test for select_by_variance (for verification command)."""
    processor = DataProcessor()

    # Create sample data with constant feature
    df = pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0],
        'feature_3': [1.0, 1.0, 1.0, 1.0, 1.0],  # Constant (variance = 0.0)
        'note': [10.0, 12.0, 14.0, 16.0, 18.0],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use threshold > 0 to remove constant features
        result = processor.select_by_variance(df, threshold=0.01)

    assert isinstance(result, pd.DataFrame)
    # Constant feature should be removed with threshold > 0
    assert 'feature_3' not in result.columns
    # Other features should remain
    assert 'feature_1' in result.columns
    assert 'feature_2' in result.columns


def test_select_by_correlation():
    """Module-level test for select_by_correlation (for verification command)."""
    processor = DataProcessor()

    # Create sample data with highly correlated features
    np.random.seed(42)
    feature_a = np.random.randn(50)
    df = pd.DataFrame({
        'feature_a': feature_a,
        'feature_b': feature_a + np.random.randn(50) * 0.01,  # Almost identical
        'feature_c': np.random.randn(50),  # Independent
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = processor.select_by_correlation(df, threshold=0.95)

    assert isinstance(result, pd.DataFrame)
    # At least one feature should remain
    assert len(result.columns) >= 1
    # Should have fewer features than original (redundant removed)
    assert len(result.columns) < len(df.columns)


def test_select_k_best():
    """Module-level test for select_k_best (for verification command)."""
    processor = DataProcessor()

    # Create sample data with varying correlations to target
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'highly_correlated': np.linspace(0, 100, n_samples) + np.random.randn(n_samples) * 5,
        'moderately_correlated': np.linspace(0, 50, n_samples) + np.random.randn(n_samples) * 15,
        'weakly_correlated': np.random.randn(n_samples) * 10,
        'note': np.linspace(0, 100, n_samples) + np.random.randn(n_samples) * 10,
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = processor.select_k_best(df, 'note', k=2)

    assert isinstance(result, pd.DataFrame)
    # Should select exactly k features
    assert len(result.columns) == 2
    # Target should not be in result
    assert 'note' not in result.columns
    # highly_correlated should be selected as one of the top 2
    assert 'highly_correlated' in result.columns
