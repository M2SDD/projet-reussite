"""
Unit tests for the RegressionModel class.

Tests cover:
- Model initialization
- Train-test split functionality
- Model training (fit)
- Prediction capabilities
- Coefficient extraction
- R² score computation
- RMSE computation
- MAE computation
- Adjusted R² computation
- Residuals computation
- Full model evaluation
- Residuals visualization
- Normality testing (Shapiro-Wilk)
- Q-Q plot generation
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.regression_model import RegressionModel
from src.config import Config


@pytest.fixture
def model():
    """Create a RegressionModel instance for testing."""
    return RegressionModel()


@pytest.fixture
def model_with_config():
    """Create a RegressionModel instance with custom config."""
    config = Config()
    return RegressionModel(config=config)


@pytest.fixture
def sample_data():
    """Create a sample DataFrame with features and target."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        'feature_3': [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'note': [10.0, 12.0, 14.0, 16.0, 18.0, 15.0, 13.0, 11.0, 9.0, 7.0],
    })


@pytest.fixture
def large_sample_data():
    """Create a larger sample dataset for more robust testing."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    # Create target with known relationship: y = 2*X1 + 3*X2 - 1*X3 + 5 + noise
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1 * X[:, 2] + 5 + np.random.randn(n_samples) * 0.5

    return pd.DataFrame({
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'feature_3': X[:, 2],
        'note': y
    })


@pytest.fixture
def perfect_linear_data():
    """Create perfectly linear data with no noise."""
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2 * X + 5  # Perfect linear relationship

    return pd.DataFrame({
        'feature_1': X,
        'note': y
    })


@pytest.fixture
def trained_model(model, sample_data):
    """Create a pre-trained model for testing."""
    X = sample_data.drop(columns=['note'])
    y = sample_data['note']
    model.fit(X, y)
    return model


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def single_feature_data():
    """Create a dataset with a single feature."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'note': [2.0, 4.0, 6.0, 8.0, 10.0]
    })


class TestModelInitialization:
    """Test model initialization."""

    def test_init_default_config(self, model):
        """Test initialization with default config."""
        assert model.model is None
        assert model.config is not None
        assert isinstance(model.config, Config)

    def test_init_custom_config(self, model_with_config):
        """Test initialization with custom config."""
        assert model_with_config.model is None
        assert model_with_config.config is not None
        assert isinstance(model_with_config.config, Config)

    def test_init_none_config(self):
        """Test initialization with None config creates default config."""
        model = RegressionModel(config=None)
        assert model.config is not None
        assert isinstance(model.config, Config)


class TestTrainTestSplit:
    """Test train-test split functionality."""

    def test_train_test_split_basic(self, model, sample_data):
        """Test basic train-test split."""
        X_train, X_test, y_train, y_test = model.train_test_split(
            sample_data, 'note', test_size=0.2, random_state=42
        )

        assert len(X_train) == 8  # 80% of 10
        assert len(X_test) == 2   # 20% of 10
        assert len(y_train) == 8
        assert len(y_test) == 2
        assert 'note' not in X_train.columns
        assert 'note' not in X_test.columns

    def test_train_test_split_proportions(self, model, large_sample_data):
        """Test train-test split proportions with larger dataset."""
        X_train, X_test, y_train, y_test = model.train_test_split(
            large_sample_data, 'note', test_size=0.3, random_state=42
        )

        total_samples = len(large_sample_data)
        expected_test = int(total_samples * 0.3)

        assert len(X_test) == expected_test
        assert len(X_train) == total_samples - expected_test

    def test_train_test_split_missing_target(self, model, sample_data):
        """Test error when target column doesn't exist."""
        with pytest.raises(ValueError, match="n'existe pas"):
            model.train_test_split(sample_data, 'nonexistent_column')

    def test_train_test_split_random_state(self, model, sample_data):
        """Test that random_state ensures reproducibility."""
        X_train1, X_test1, y_train1, y_test1 = model.train_test_split(
            sample_data, 'note', test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = model.train_test_split(
            sample_data, 'note', test_size=0.2, random_state=42
        )

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)

    def test_train_test_split_different_test_sizes(self, model, sample_data):
        """Test different test_size values."""
        for test_size in [0.1, 0.2, 0.3, 0.5]:
            X_train, X_test, y_train, y_test = model.train_test_split(
                sample_data, 'note', test_size=test_size, random_state=42
            )
            assert len(X_test) == int(len(sample_data) * test_size)


class TestModelFit:
    """Test model training functionality."""

    def test_fit_basic(self, model, sample_data):
        """Test basic model fitting."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        result = model.fit(X, y)

        assert model.model is not None
        assert isinstance(model.model, LinearRegression)
        assert result is model  # Test method chaining

    def test_fit_with_numpy_arrays(self, model):
        """Test fitting with numpy arrays."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model.fit(X, y)

        assert model.model is not None

    def test_fit_empty_X(self, model):
        """Test error handling with empty X."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            model.fit(X, y)

    def test_fit_empty_y(self, model, sample_data):
        """Test error handling with empty y."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series([])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            model.fit(X, y)

    def test_fit_dimension_mismatch(self, model, sample_data):
        """Test error handling with mismatched dimensions."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series([1, 2, 3])  # Different length

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            model.fit(X, y)

    def test_fit_single_feature(self, model, single_feature_data):
        """Test fitting with a single feature."""
        X = single_feature_data.drop(columns=['note'])
        y = single_feature_data['note']

        model.fit(X, y)

        assert model.model is not None


class TestModelPredict:
    """Test model prediction functionality."""

    def test_predict_basic(self, trained_model, sample_data):
        """Test basic prediction."""
        X = sample_data.drop(columns=['note'])
        predictions = trained_model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_before_training(self, model, sample_data):
        """Test error when predicting before training."""
        X = sample_data.drop(columns=['note'])

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.predict(X)

    def test_predict_empty_X(self, trained_model):
        """Test error handling with empty X."""
        X = pd.DataFrame()

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.predict(X)

    def test_predict_returns_array(self, trained_model, sample_data):
        """Test that predict returns numpy array."""
        X = sample_data.drop(columns=['note'])
        predictions = trained_model.predict(X)

        assert isinstance(predictions, np.ndarray)


class TestGetCoefficients:
    """Test coefficient extraction."""

    def test_get_coefficients_basic(self, trained_model):
        """Test basic coefficient extraction."""
        coeffs = trained_model.get_coefficients()

        assert isinstance(coeffs, dict)
        assert 'coefficients' in coeffs
        assert 'intercept' in coeffs
        assert isinstance(coeffs['coefficients'], np.ndarray)
        assert isinstance(coeffs['intercept'], (float, np.floating))

    def test_get_coefficients_before_training(self, model):
        """Test error when getting coefficients before training."""
        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.get_coefficients()

    def test_get_coefficients_shape(self, model, sample_data):
        """Test that coefficients have correct shape."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']
        model.fit(X, y)

        coeffs = model.get_coefficients()

        assert len(coeffs['coefficients']) == X.shape[1]

    def test_get_coefficients_perfect_fit(self, model, perfect_linear_data):
        """Test coefficients with perfect linear data."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']
        model.fit(X, y)

        coeffs = model.get_coefficients()

        # Should be close to y = 2*X + 5
        assert np.isclose(coeffs['coefficients'][0], 2.0, rtol=1e-10)
        assert np.isclose(coeffs['intercept'], 5.0, rtol=1e-10)


class TestR2Score:
    """Test R² score computation."""

    def test_compute_r2_score_basic(self, trained_model, sample_data):
        """Test basic R² computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        r2 = trained_model.compute_r2_score(X, y)

        assert isinstance(r2, (float, np.floating))
        assert r2 <= 1.0

    def test_compute_r2_score_perfect_fit(self, model, perfect_linear_data):
        """Test R² with perfect linear fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']
        model.fit(X, y)

        r2 = model.compute_r2_score(X, y)

        assert np.isclose(r2, 1.0, rtol=1e-10)

    def test_compute_r2_score_before_training(self, model, sample_data):
        """Test error when computing R² before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_r2_score(X, y)

    def test_compute_r2_score_empty_X(self, trained_model):
        """Test error handling with empty X."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.compute_r2_score(X, y)

    def test_compute_r2_score_dimension_mismatch(self, trained_model, sample_data):
        """Test error handling with mismatched dimensions."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            trained_model.compute_r2_score(X, y)


class TestRMSE:
    """Test RMSE computation."""

    def test_compute_rmse_basic(self, trained_model, sample_data):
        """Test basic RMSE computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        rmse = trained_model.compute_rmse(X, y)

        assert isinstance(rmse, (float, np.floating))
        assert rmse >= 0.0

    def test_compute_rmse_perfect_fit(self, model, perfect_linear_data):
        """Test RMSE with perfect linear fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']
        model.fit(X, y)

        rmse = model.compute_rmse(X, y)

        assert np.isclose(rmse, 0.0, atol=1e-10)

    def test_compute_rmse_before_training(self, model, sample_data):
        """Test error when computing RMSE before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_rmse(X, y)

    def test_compute_rmse_positive(self, trained_model, sample_data):
        """Test that RMSE is always positive."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        rmse = trained_model.compute_rmse(X, y)

        assert rmse >= 0.0


class TestMAE:
    """Test MAE computation."""

    def test_compute_mae_basic(self, trained_model, sample_data):
        """Test basic MAE computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        mae = trained_model.compute_mae(X, y)

        assert isinstance(mae, (float, np.floating))
        assert mae >= 0.0

    def test_compute_mae_perfect_fit(self, model, perfect_linear_data):
        """Test MAE with perfect linear fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']
        model.fit(X, y)

        mae = model.compute_mae(X, y)

        assert np.isclose(mae, 0.0, atol=1e-10)

    def test_compute_mae_before_training(self, model, sample_data):
        """Test error when computing MAE before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_mae(X, y)

    def test_compute_mae_positive(self, trained_model, sample_data):
        """Test that MAE is always positive."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        mae = trained_model.compute_mae(X, y)

        assert mae >= 0.0


class TestAdjustedR2:
    """Test adjusted R² computation."""

    def test_compute_adjusted_r2_basic(self, trained_model, sample_data):
        """Test basic adjusted R² computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        adj_r2 = trained_model.compute_adjusted_r2(X, y)

        assert isinstance(adj_r2, (float, np.floating))

    def test_compute_adjusted_r2_vs_r2(self, trained_model, large_sample_data):
        """Test that adjusted R² is less than or equal to R²."""
        X = large_sample_data.drop(columns=['note'])
        y = large_sample_data['note']

        # Train on large dataset
        trained_model.fit(X, y)

        r2 = trained_model.compute_r2_score(X, y)
        adj_r2 = trained_model.compute_adjusted_r2(X, y)

        # Adjusted R² should be <= R²
        assert adj_r2 <= r2

    def test_compute_adjusted_r2_before_training(self, model, sample_data):
        """Test error when computing adjusted R² before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_adjusted_r2(X, y)

    def test_compute_adjusted_r2_insufficient_samples(self, model):
        """Test error when n_samples <= n_features + 1."""
        # Create data with 3 samples and 3 features
        X = pd.DataFrame({
            'f1': [1, 2, 3],
            'f2': [4, 5, 6],
            'f3': [7, 8, 9]
        })
        y = pd.Series([1, 2, 3])

        model.fit(X, y)

        with pytest.raises(ValueError, match="doit être supérieur"):
            model.compute_adjusted_r2(X, y)

    def test_compute_adjusted_r2_single_feature(self, model, single_feature_data):
        """Test adjusted R² with single feature."""
        X = single_feature_data.drop(columns=['note'])
        y = single_feature_data['note']

        model.fit(X, y)
        adj_r2 = model.compute_adjusted_r2(X, y)

        assert isinstance(adj_r2, (float, np.floating))


class TestResiduals:
    """Test residuals computation."""

    def test_compute_residuals_basic(self, trained_model, sample_data):
        """Test basic residuals computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        residuals = trained_model.compute_residuals(X, y)

        assert isinstance(residuals, np.ndarray)
        assert len(residuals) == len(y)

    def test_compute_residuals_perfect_fit(self, model, perfect_linear_data):
        """Test residuals with perfect fit are near zero."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']
        model.fit(X, y)

        residuals = model.compute_residuals(X, y)

        assert np.allclose(residuals, 0.0, atol=1e-10)

    def test_compute_residuals_before_training(self, model, sample_data):
        """Test error when computing residuals before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_residuals(X, y)

    def test_compute_residuals_sum_near_zero(self, trained_model, large_sample_data):
        """Test that sum of residuals is near zero for large datasets."""
        X = large_sample_data.drop(columns=['note'])
        y = large_sample_data['note']

        # Retrain on large dataset
        trained_model.fit(X, y)

        residuals = trained_model.compute_residuals(X, y)

        # Sum of residuals should be close to zero
        assert np.abs(np.sum(residuals)) < 1.0

    def test_compute_residuals_converts_series(self, trained_model, sample_data):
        """Test that pandas Series residuals are converted to numpy array."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        residuals = trained_model.compute_residuals(X, y)

        assert isinstance(residuals, np.ndarray)


class TestEvaluate:
    """Test full model evaluation."""

    def test_evaluate_basic(self, trained_model, sample_data):
        """Test basic model evaluation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        metrics = trained_model.evaluate(X, y)

        assert isinstance(metrics, dict)
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'adjusted_r2' in metrics

    def test_evaluate_all_metrics_present(self, trained_model, sample_data):
        """Test that all metrics are present in evaluation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        metrics = trained_model.evaluate(X, y)

        required_metrics = ['r2', 'rmse', 'mae', 'adjusted_r2']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (float, np.floating))

    def test_evaluate_perfect_fit(self, model, perfect_linear_data):
        """Test evaluation metrics with perfect fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']
        model.fit(X, y)

        metrics = model.evaluate(X, y)

        assert np.isclose(metrics['r2'], 1.0, rtol=1e-10)
        assert np.isclose(metrics['rmse'], 0.0, atol=1e-10)
        assert np.isclose(metrics['mae'], 0.0, atol=1e-10)
        assert np.isclose(metrics['adjusted_r2'], 1.0, rtol=1e-10)

    def test_evaluate_before_training(self, model, sample_data):
        """Test error when evaluating before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.evaluate(X, y)


class TestPlotResiduals:
    """Test residuals plotting functionality."""

    def test_plot_residuals_basic(self, trained_model, sample_data):
        """Test basic residuals plot creation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        fig = trained_model.plot_residuals(X, y)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_residuals_before_training(self, model, sample_data):
        """Test error when plotting residuals before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.plot_residuals(X, y)

    def test_plot_residuals_has_axes(self, trained_model, sample_data):
        """Test that plot has axes configured."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        fig = trained_model.plot_residuals(X, y)

        axes = fig.get_axes()
        assert len(axes) > 0
        plt.close(fig)

    def test_plot_residuals_empty_X(self, trained_model):
        """Test error handling with empty X."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.plot_residuals(X, y)

    def test_plot_residuals_dimension_mismatch(self, trained_model, sample_data):
        """Test error handling with mismatched dimensions."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            trained_model.plot_residuals(X, y)


class TestResidualsNormality:
    """Test residuals normality testing."""

    def test_check_residuals_normality_basic(self, trained_model, sample_data):
        """Test basic normality test."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        result = trained_model.check_residuals_normality(X, y)

        assert isinstance(result, dict)
        assert 'test_statistic' in result
        assert 'p_value' in result

    def test_check_residuals_normality_values(self, trained_model, large_sample_data):
        """Test that normality test returns valid values."""
        X = large_sample_data.drop(columns=['note'])
        y = large_sample_data['note']

        # Retrain on large dataset
        trained_model.fit(X, y)

        result = trained_model.check_residuals_normality(X, y)

        # Test statistic should be between 0 and 1
        assert 0 <= result['test_statistic'] <= 1
        # p-value should be between 0 and 1
        assert 0 <= result['p_value'] <= 1

    def test_check_residuals_normality_before_training(self, model, sample_data):
        """Test error when checking normality before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.check_residuals_normality(X, y)

    def test_check_residuals_normality_perfect_fit(self, model, perfect_linear_data):
        """Test normality test with perfect fit (residuals should be zero)."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']
        model.fit(X, y)

        result = model.check_residuals_normality(X, y)

        # With perfect fit, residuals are all zero, test should still run
        assert 'test_statistic' in result
        assert 'p_value' in result


class TestQQPlot:
    """Test Q-Q plot generation."""

    def test_plot_qq_plot_basic(self, trained_model, sample_data):
        """Test basic Q-Q plot creation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        fig = trained_model.plot_qq_plot(X, y)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_qq_plot_before_training(self, model, sample_data):
        """Test error when plotting Q-Q before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.plot_qq_plot(X, y)

    def test_plot_qq_plot_has_axes(self, trained_model, sample_data):
        """Test that Q-Q plot has axes configured."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        fig = trained_model.plot_qq_plot(X, y)

        axes = fig.get_axes()
        assert len(axes) > 0
        plt.close(fig)

    def test_plot_qq_plot_empty_X(self, trained_model):
        """Test error handling with empty X."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.plot_qq_plot(X, y)

    def test_plot_qq_plot_dimension_mismatch(self, trained_model, sample_data):
        """Test error handling with mismatched dimensions."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            trained_model.plot_qq_plot(X, y)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_model_retrain(self, model, sample_data, large_sample_data):
        """Test that model can be retrained with different data."""
        # Train on sample data
        X1 = sample_data.drop(columns=['note'])
        y1 = sample_data['note']
        model.fit(X1, y1)

        coeffs1 = model.get_coefficients()

        # Retrain on large sample data
        X2 = large_sample_data.drop(columns=['note'])
        y2 = large_sample_data['note']
        model.fit(X2, y2)

        coeffs2 = model.get_coefficients()

        # Coefficients should be different
        assert not np.array_equal(coeffs1['coefficients'], coeffs2['coefficients'])

    def test_model_chaining(self, model, sample_data):
        """Test method chaining with fit."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        # Should be able to chain methods
        result = model.fit(X, y)

        assert result is model
        assert model.model is not None

    def test_numpy_array_input(self, model):
        """Test that model works with numpy arrays."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)

    def test_mixed_input_types(self, model, sample_data):
        """Test training with DataFrame and predicting with array."""
        X_df = sample_data.drop(columns=['note'])
        y_series = sample_data['note']

        model.fit(X_df, y_series)

        X_array = X_df.values
        predictions = model.predict(X_array)

        assert len(predictions) == len(X_array)

    def test_metrics_consistency(self, trained_model, sample_data):
        """Test that metrics are consistent across multiple calls."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        r2_1 = trained_model.compute_r2_score(X, y)
        r2_2 = trained_model.compute_r2_score(X, y)

        assert r2_1 == r2_2

    def test_large_dataset_performance(self, model):
        """Test model with larger dataset."""
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(n_samples) * 0.1

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        # With low noise, R² should be high
        assert metrics['r2'] > 0.95
