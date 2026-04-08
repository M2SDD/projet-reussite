"""
Unit tests for the MLModel class.

Tests cover:
- Model initialization
- Train-test split functionality
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
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from src.ml_model import MLModel
from src.regression_model import RegressionModel
from src.config import Config


@pytest.fixture
def model():
    """Create a MLModel instance for testing."""
    return MLModel()


@pytest.fixture
def model_with_config():
    """Create a MLModel instance with custom config."""
    config = Config()
    return MLModel(config=config)


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
        model = MLModel(config=None)
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
        assert isinstance(model.model, RandomForestRegressor)
        assert result is model  # Check method chaining

    def test_fit_with_dataframe(self, model, sample_data):
        """Test fitting with DataFrame."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        model.fit(X, y)

        assert model.model is not None
        assert hasattr(model.model, 'coef_')
        assert hasattr(model.model, 'intercept_')

    def test_fit_with_numpy_arrays(self, model, sample_data):
        """Test fitting with numpy arrays."""
        X = sample_data.drop(columns=['note']).values
        y = sample_data['note'].values

        model.fit(X, y)

        assert model.model is not None
        assert hasattr(model.model, 'coef_')

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
        assert len(model.model.coef_) == 1


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
        """Test R² score with perfect linear fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        r2 = model.compute_r2_score(X, y)

        assert np.isclose(r2, 1.0, atol=1e-10)  # Should be very close to 1

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
        """Test RMSE with perfect linear fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        rmse = model.compute_rmse(X, y)

        assert np.isclose(rmse, 0.0, atol=1e-10)  # Should be very close to 0

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
        """Test MAE with perfect linear fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        mae = model.compute_mae(X, y)

        assert np.isclose(mae, 0.0, atol=1e-10)  # Should be very close to 0

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

        metrics = trained_model.evaluate(X, y)

        expected_keys = {'r2', 'rmse', 'mae'}
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
        """Test evaluation with perfect linear fit."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        assert np.isclose(metrics['r2'], 1.0, atol=1e-10)
        assert np.isclose(metrics['rmse'], 0.0, atol=1e-10)
        assert np.isclose(metrics['mae'], 0.0, atol=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_fit_with_single_sample(self, model):
        """Test that fitting with single sample raises appropriate error."""
        X = pd.DataFrame({'feature_1': [1.0]})
        y = pd.Series([2.0])

        # LinearRegression should work with single sample, but it's not meaningful
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
        model = MLModel(config=None)

        assert model.config is not None
        assert isinstance(model.config, Config)


class TestMLModelVsRegressionModelComparison:
    """Integration tests comparing MLModel and RegressionModel on same data."""

    def test_mlmodel_vs_regression_comparison(self, large_sample_data):
        """Test that MLModel and RegressionModel produce similar results on same data."""
        # Prepare data
        X = large_sample_data.drop(columns=['note'])
        y = large_sample_data['note']

        # Create and train both models
        ml_model = MLModel()
        regression_model = RegressionModel()

        ml_model.fit(X, y)
        regression_model.fit(X, y)

        # Compare predictions
        ml_predictions = ml_model.predict(X)
        regression_predictions = regression_model.predict(X)

        # Predictions should be very close (both use LinearRegression)
        np.testing.assert_allclose(ml_predictions, regression_predictions, rtol=1e-10)

        # Compare R² scores
        ml_r2 = ml_model.compute_r2_score(X, y)
        regression_r2 = regression_model.compute_r2_score(X, y)

        assert np.isclose(ml_r2, regression_r2, rtol=1e-10)

        # Compare RMSE
        ml_rmse = ml_model.compute_rmse(X, y)
        regression_rmse = regression_model.compute_rmse(X, y)

        assert np.isclose(ml_rmse, regression_rmse, rtol=1e-10)

        # Compare MAE
        ml_mae = ml_model.compute_mae(X, y)
        regression_mae = regression_model.compute_mae(X, y)

        assert np.isclose(ml_mae, regression_mae, rtol=1e-10)

        # Compare coefficients
        ml_coefs = ml_model.model.coef_
        regression_coefs_dict = regression_model.get_coefficients()

        # Extract coefficients from regression model dictionary
        regression_coefs = regression_coefs_dict['coefficients']

        np.testing.assert_allclose(ml_coefs, regression_coefs, rtol=1e-10)

        # Compare intercepts
        ml_intercept = ml_model.model.intercept_
        regression_intercept = regression_coefs_dict['intercept']

        assert np.isclose(ml_intercept, regression_intercept, rtol=1e-10)

    def test_mlmodel_vs_regression_evaluation(self, sample_data):
        """Test that both models' evaluate methods return similar metrics."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        # Create and train both models
        ml_model = MLModel()
        regression_model = RegressionModel()

        ml_model.fit(X, y)
        regression_model.fit(X, y)

        # Get evaluation metrics from both
        ml_metrics = ml_model.evaluate(X, y)
        regression_metrics = regression_model.evaluate(X, y)

        # Compare common metrics
        assert np.isclose(ml_metrics['r2'], regression_metrics['r2'], rtol=1e-10)
        assert np.isclose(ml_metrics['rmse'], regression_metrics['rmse'], rtol=1e-10)
        assert np.isclose(ml_metrics['mae'], regression_metrics['mae'], rtol=1e-10)

    def test_mlmodel_vs_regression_perfect_fit(self, perfect_linear_data):
        """Test that both models achieve perfect fit on perfect linear data."""
        X = perfect_linear_data.drop(columns=['note'])
        y = perfect_linear_data['note']

        # Create and train both models
        ml_model = MLModel()
        regression_model = RegressionModel()

        ml_model.fit(X, y)
        regression_model.fit(X, y)

        # Both should achieve near-perfect R²
        ml_r2 = ml_model.compute_r2_score(X, y)
        regression_r2 = regression_model.compute_r2_score(X, y)

        assert np.isclose(ml_r2, 1.0, atol=1e-10)
        assert np.isclose(regression_r2, 1.0, atol=1e-10)
        assert np.isclose(ml_r2, regression_r2, rtol=1e-10)

        # Both should have near-zero RMSE
        ml_rmse = ml_model.compute_rmse(X, y)
        regression_rmse = regression_model.compute_rmse(X, y)

        assert np.isclose(ml_rmse, 0.0, atol=1e-10)
        assert np.isclose(regression_rmse, 0.0, atol=1e-10)

    def test_mlmodel_vs_regression_train_test_split(self, large_sample_data):
        """Test that both models handle train-test splits consistently."""
        # Use the same random_state for both models
        ml_model = MLModel()
        regression_model = RegressionModel()

        # Perform train-test split with both models
        ml_X_train, ml_X_test, ml_y_train, ml_y_test = ml_model.train_test_split(
            large_sample_data, 'note', test_size=0.2, random_state=42
        )

        regression_X_train, regression_X_test, regression_y_train, regression_y_test = regression_model.train_test_split(
            large_sample_data, 'note', test_size=0.2, random_state=42
        )

        # Both should produce identical splits
        pd.testing.assert_frame_equal(ml_X_train, regression_X_train)
        pd.testing.assert_frame_equal(ml_X_test, regression_X_test)
        pd.testing.assert_series_equal(ml_y_train, regression_y_train)
        pd.testing.assert_series_equal(ml_y_test, regression_y_test)

        # Train both models
        ml_model.fit(ml_X_train, ml_y_train)
        regression_model.fit(regression_X_train, regression_y_train)

        # Evaluate on test set
        ml_test_r2 = ml_model.compute_r2_score(ml_X_test, ml_y_test)
        regression_test_r2 = regression_model.compute_r2_score(regression_X_test, regression_y_test)

        assert np.isclose(ml_test_r2, regression_test_r2, rtol=1e-10)
