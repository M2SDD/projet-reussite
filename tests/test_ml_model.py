"""
Unit tests for the MLModel class.

Tests cover:
- Model initialization
- Model training (fit)
- Prediction capabilities
- Feature importance extraction
- R² score computation
- RMSE computation
- MAE computation
- Full model evaluation
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from sklearn.ensemble import RandomForestRegressor
from src.models.ensemble_regressor import EnsembleRegressor
from src.models.linear_regressor import LinearRegressor
from src.config import Config


@pytest.fixture
def model():
    """Create a MLModel instance for testing."""
    return EnsembleRegressor()


@pytest.fixture
def model_with_config():
    """Create a MLModel instance with custom config."""
    config = Config()
    return EnsembleRegressor(config=config)


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
        model = EnsembleRegressor(config=None)
        assert model.config is not None
        assert isinstance(model.config, Config)

class TestModelFit:
    """Test model training functionality."""

    def test_fit_basic(self, model, sample_data):
        """Test basic model fitting."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        result = model.fit(X, y)

        assert model.model is not None
        assert isinstance(model.model, RandomForestRegressor)
        assert result is model  # Check method chaining

    def test_fit_with_dataframe(self, model, sample_data):
        """Test fitting with DataFrame creates RandomForest model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        model.fit(X, y)

        assert model.model is not None
        assert isinstance(model.model, RandomForestRegressor)
        assert hasattr(model.model, 'feature_importances_')
        assert len(model.model.feature_importances_) == X.shape[1]

    def test_fit_with_numpy_arrays(self, model, sample_data):
        """Test fitting with numpy arrays."""
        X = sample_data.drop(columns=['note']).values
        y = sample_data['note'].values

        model.fit(X, y)

        assert model.model is not None
        assert hasattr(model.model, 'feature_importances_')

    def test_fit_empty_X(self, model):
        """Test error when X is empty."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            model.fit(X, y)

    def test_fit_empty_y(self, model, sample_data):
        """Test error when y is empty."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            model.fit(X, y)

    def test_fit_mismatched_dimensions(self, model, sample_data):
        """Test error when X and y have different number of samples."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note'][:5]  # Only first 5 samples

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            model.fit(X, y)

    def test_fit_single_feature(self, model, single_feature_data):
        """Test fitting with a single feature."""
        X = single_feature_data.drop(columns=['note'])
        y = single_feature_data['note']

        model.fit(X, y)

        assert model.model is not None
        assert len(model.model.feature_importances_) == 1


class TestModelPredict:
    """Test model prediction functionality."""

    def test_predict_basic(self, trained_model, sample_data):
        """Test basic prediction."""
        X = sample_data.drop(columns=['note'])

        predictions = trained_model.predict(X)

        assert predictions is not None
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

    def test_predict_untrained_model(self, model, sample_data):
        """Test error when predicting with untrained model."""
        X = sample_data.drop(columns=['note'])

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.predict(X)

    def test_predict_empty_X(self, trained_model):
        """Test error when X is empty."""
        X = pd.DataFrame()

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.predict(X)

    def test_predict_returns_array(self, trained_model, sample_data):
        """Test that predict returns numpy array."""
        X = sample_data.drop(columns=['note'])

        predictions = trained_model.predict(X)

        assert isinstance(predictions, np.ndarray)

    def test_predict_with_numpy_array(self, trained_model, sample_data):
        """Test prediction with numpy array input."""
        X = sample_data.drop(columns=['note']).values

        predictions = trained_model.predict(X)

        assert predictions is not None
        assert len(predictions) == len(X)


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_get_feature_importance_basic(self, trained_model, sample_data):
        """Test basic feature importance extraction."""
        feature_names = sample_data.drop(columns=['note']).columns

        importance = trained_model.get_feature_importance(feature_names)

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == len(feature_names)

    def test_get_feature_importance_sorted(self, trained_model, sample_data):
        """Test that feature importance is sorted in descending order."""
        feature_names = sample_data.drop(columns=['note']).columns

        importance = trained_model.get_feature_importance(feature_names)

        # Check that importance values are sorted in descending order
        importance_values = importance['importance'].values
        assert all(importance_values[i] >= importance_values[i + 1]
                   for i in range(len(importance_values) - 1))

    def test_get_feature_importance_untrained_model(self, model):
        """Test error when getting feature importance from untrained model."""
        feature_names = ['feature_1', 'feature_2']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.get_feature_importance(feature_names)

    def test_get_feature_importance_empty_names(self, trained_model):
        """Test error when feature_names is empty."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.get_feature_importance([])

    def test_get_feature_importance_mismatched_count(self, trained_model):
        """Test error when number of names doesn't match number of features."""
        feature_names = ['feature_1', 'feature_2']  # Only 2 names but model has 3 features

        with pytest.raises(ValueError, match="ne correspond pas"):
            trained_model.get_feature_importance(feature_names)

    def test_get_feature_importance_values_absolute(self, model, sample_data):
        """Test that feature importance uses absolute values of coefficients."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']
        model.fit(X, y)

        importance = model.get_feature_importance(X.columns)

        # All importance values should be non-negative (absolute values)
        assert all(importance['importance'] >= 0)


class TestR2Score:
    """Test R² score computation."""

    def test_compute_r2_score_basic(self, trained_model, sample_data):
        """Test basic R² score computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        r2 = trained_model.compute_r2_score(X, y)

        assert isinstance(r2, (int, float))
        assert r2 <= 1.0  # R² is typically <= 1

    def test_compute_r2_score_perfect_fit(self, model, perfect_linear_data):
        """Test R² score with perfect linear fit.

        Note: RandomForest may not achieve perfect R²=1.0 on linear data due to
        ensemble averaging, but should be very high (>0.99).
        """
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        r2 = model.compute_r2_score(X, y)

        assert r2 > 0.99  # RandomForest should achieve very high R² on this data

    def test_compute_r2_score_untrained_model(self, model, sample_data):
        """Test error when computing R² with untrained model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_r2_score(X, y)

    def test_compute_r2_score_empty_X(self, trained_model):
        """Test error when X is empty."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.compute_r2_score(X, y)

    def test_compute_r2_score_empty_y(self, trained_model, sample_data):
        """Test error when y is empty."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.compute_r2_score(X, y)

    def test_compute_r2_score_mismatched_dimensions(self, trained_model, sample_data):
        """Test error when X and y have different number of samples."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note'][:5]

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            trained_model.compute_r2_score(X, y)


class TestRMSE:
    """Test RMSE computation."""

    def test_compute_rmse_basic(self, trained_model, sample_data):
        """Test basic RMSE computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        rmse = trained_model.compute_rmse(X, y)

        assert isinstance(rmse, (int, float))
        assert rmse >= 0  # RMSE is always non-negative

    def test_compute_rmse_perfect_fit(self, model, perfect_linear_data):
        """Test RMSE with perfect linear fit.

        Note: RandomForest may have small non-zero RMSE even on linear data
        due to ensemble averaging.
        """
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        rmse = model.compute_rmse(X, y)

        assert rmse < 0.5  # RandomForest should achieve very low RMSE

    def test_compute_rmse_untrained_model(self, model, sample_data):
        """Test error when computing RMSE with untrained model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_rmse(X, y)

    def test_compute_rmse_empty_X(self, trained_model):
        """Test error when X is empty."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.compute_rmse(X, y)

    def test_compute_rmse_empty_y(self, trained_model, sample_data):
        """Test error when y is empty."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.compute_rmse(X, y)

    def test_compute_rmse_mismatched_dimensions(self, trained_model, sample_data):
        """Test error when X and y have different number of samples."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note'][:5]

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            trained_model.compute_rmse(X, y)


class TestMAE:
    """Test MAE computation."""

    def test_compute_mae_basic(self, trained_model, sample_data):
        """Test basic MAE computation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        mae = trained_model.compute_mae(X, y)

        assert isinstance(mae, (int, float))
        assert mae >= 0  # MAE is always non-negative

    def test_compute_mae_perfect_fit(self, model, perfect_linear_data):
        """Test MAE with perfect linear fit.

        Note: RandomForest may have small non-zero MAE even on linear data
        due to ensemble averaging.
        """
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        mae = model.compute_mae(X, y)

        assert mae < 0.5  # RandomForest should achieve very low MAE

    def test_compute_mae_untrained_model(self, model, sample_data):
        """Test error when computing MAE with untrained model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.compute_mae(X, y)

    def test_compute_mae_empty_X(self, trained_model):
        """Test error when X is empty."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.compute_mae(X, y)

    def test_compute_mae_empty_y(self, trained_model, sample_data):
        """Test error when y is empty."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            trained_model.compute_mae(X, y)

    def test_compute_mae_mismatched_dimensions(self, trained_model, sample_data):
        """Test error when X and y have different number of samples."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note'][:5]

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            trained_model.compute_mae(X, y)


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

    def test_evaluate_all_metrics_present(self, trained_model, sample_data):
        """Test that all expected metrics are present."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        metrics = trained_model.evaluate(X, y, include_adjusted_r2=True)

        expected_keys = {'r2', 'rmse', 'mae', 'adjusted_r2'}
        assert set(metrics.keys()) == expected_keys

    def test_evaluate_metric_types(self, trained_model, sample_data):
        """Test that all metrics are numeric."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        metrics = trained_model.evaluate(X, y)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"

    def test_evaluate_untrained_model(self, model, sample_data):
        """Test error when evaluating untrained model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            model.evaluate(X, y)

    def test_evaluate_perfect_fit(self, model, perfect_linear_data):
        """Test evaluation with perfect linear fit.

        Note: RandomForest may not achieve perfect metrics on linear data
        due to ensemble averaging, but should be very good.
        """
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        assert metrics['r2'] > 0.99  # Very high R² expected
        assert metrics['rmse'] < 0.5  # Very low RMSE expected
        assert metrics['mae'] < 0.5  # Very low MAE expected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_fit_with_single_sample(self, model):
        """Test that fitting with single sample works (not ideal but allowed)."""
        X = pd.DataFrame({'feature_1': [1.0]})
        y = pd.Series([2.0])

        # RandomForest should work with single sample (though not meaningful)
        model.fit(X, y)
        assert model.model is not None

    def test_predict_single_sample(self, trained_model):
        """Test prediction on single sample."""
        X = pd.DataFrame({
            'feature_1': [5.0],
            'feature_2': [10.0],
            'feature_3': [3.0]
        })

        predictions = trained_model.predict(X)

        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.number))

    def test_large_dataset(self, model, large_sample_data):
        """Test with larger dataset."""
        X = large_sample_data.drop(columns=['note'])
        y = large_sample_data['note']

        model.fit(X, y)
        r2 = model.compute_r2_score(X, y)

        # With known relationship, R² should be quite high
        assert r2 > 0.8

    def test_none_values_in_config(self):
        """Test that None config is handled properly."""
        model = EnsembleRegressor(config=None)

        assert model.config is not None
        assert isinstance(model.config, Config)


class TestMLModelVsRegressionModelComparison:
    """Integration tests comparing MLModel and RegressionModel on same data."""

    def test_mlmodel_vs_regression_comparison(self, large_sample_data):
        """Test comprehensive comparison between MLModel and RegressionModel.

        This test validates that both models can be trained on the same data
        and produce valid predictions and metrics. The models use different
        algorithms (RandomForest vs LinearRegression), so results will differ.
        """
        # Prepare data
        X = large_sample_data.drop(columns=['note'])
        y = large_sample_data['note']

        # Create and train both models
        ml_model = EnsembleRegressor(model_type='random_forest')  # Explicit model type
        regression_model = LinearRegressor()

        ml_model.fit(X, y)
        regression_model.fit(X, y)

        # Both should produce predictions
        ml_predictions = ml_model.predict(X)
        regression_predictions = regression_model.predict(X)

        # Predictions should have same shape
        assert ml_predictions.shape == regression_predictions.shape

        # IMPORTANT: Predictions should NOT necessarily be identical!
        # Random Forest and Linear Regression are different algorithms.
        # This is the whole point of the feature - comparing two approaches.

        # Both should compute metrics
        ml_r2 = ml_model.compute_r2_score(X, y)
        regression_r2 = regression_model.compute_r2_score(X, y)

        ml_rmse = ml_model.compute_rmse(X, y)
        regression_rmse = regression_model.compute_rmse(X, y)

        ml_mae = ml_model.compute_mae(X, y)
        regression_mae = regression_model.compute_mae(X, y)

        # All metrics should be valid numbers
        assert isinstance(ml_r2, (int, float))
        assert isinstance(regression_r2, (int, float))
        assert isinstance(ml_rmse, (int, float))
        assert isinstance(regression_rmse, (int, float))
        assert isinstance(ml_mae, (int, float))
        assert isinstance(regression_mae, (int, float))

        # MLModel should support feature importance
        ml_fi = ml_model.get_feature_importance(X.columns)

        # ML feature importances should sum to 1.0 (Gini importance)
        assert abs(ml_fi['importance'].sum() - 1.0) < 1e-6

        # Should return DataFrame with correct structure
        assert list(ml_fi.columns) == ['feature', 'importance']
        assert len(ml_fi) == X.shape[1]  # One importance per feature

        # RegressionModel has get_coefficients() instead (different approach)
        reg_coefs = regression_model.get_coefficients()
        assert 'coefficients' in reg_coefs
        assert 'intercept' in reg_coefs

    def test_mlmodel_vs_regression_evaluation(self, sample_data):
        """Test that both models' evaluate methods work correctly."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        # Create and train both models
        ml_model = EnsembleRegressor()
        regression_model = LinearRegressor()

        ml_model.fit(X, y)
        regression_model.fit(X, y)

        # Get evaluation metrics from both
        ml_metrics = ml_model.evaluate(X, y)
        regression_metrics = regression_model.evaluate(X, y)

        # Both should have all metrics
        assert 'r2' in ml_metrics and 'r2' in regression_metrics
        assert 'rmse' in ml_metrics and 'rmse' in regression_metrics
        assert 'mae' in ml_metrics and 'mae' in regression_metrics

        # Metrics should be valid numbers
        for metric in ['r2', 'rmse', 'mae']:
            assert isinstance(ml_metrics[metric], (int, float))
            assert isinstance(regression_metrics[metric], (int, float))

    def test_mlmodel_vs_regression_perfect_fit(self, perfect_linear_data):
        """Test that both models achieve good fit on perfect linear data."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        # Create and train both models
        ml_model = EnsembleRegressor()
        regression_model = LinearRegressor()

        ml_model.fit(X, y)
        regression_model.fit(X, y)

        # Linear regression should achieve perfect R² on linear data
        regression_r2 = regression_model.compute_r2_score(X, y)
        assert np.isclose(regression_r2, 1.0, atol=1e-10)

        # Random Forest should achieve very high R² (but may not be exactly 1.0)
        ml_r2 = ml_model.compute_r2_score(X, y)
        assert ml_r2 > 0.99  # Very high R² expected

        # Both should have low RMSE
        ml_rmse = ml_model.compute_rmse(X, y)
        regression_rmse = regression_model.compute_rmse(X, y)

        # Linear regression achieves near-zero RMSE on perfect linear data
        assert np.isclose(regression_rmse, 0.0, atol=1e-10)
        # Random Forest should have low RMSE but not necessarily zero
        assert ml_rmse < 0.5
