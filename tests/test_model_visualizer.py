"""
Unit tests for the ModelVisualizer class.

Tests cover:
- Residuals visualization
- Q-Q plot generation
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from src.models.linear_regressor import LinearRegressor
from src.visualization.model_visualizer import ModelVisualizer
from src.config import Config


@pytest.fixture
def model():
    """Create a RegressionModel instance for testing."""
    return LinearRegressor()


@pytest.fixture
def model_with_config():
    """Create a RegressionModel instance with custom config."""
    config = Config()
    return LinearRegressor(config=config)


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


class TestPlotResiduals:
    """Test residuals plotting functionality."""

    def test_plot_residuals_basic(self, trained_model, sample_data):
        """Test basic residuals plot creation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        ax = ModelVisualizer.plot_residuals(trained_model, X, y)

        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_plot_residuals_before_training(self, model, sample_data):
        """Test error when plotting residuals before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            ModelVisualizer.plot_residuals(model, X, y)

    def test_plot_residuals_empty_X(self, trained_model):
        """Test error handling with empty X."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            ModelVisualizer.plot_residuals(trained_model, X, y)

    def test_plot_residuals_dimension_mismatch(self, trained_model, sample_data):
        """Test error handling with mismatched dimensions."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            ModelVisualizer.plot_residuals(trained_model, X, y)


class TestQQPlot:
    """Test Q-Q plot generation."""

    def test_plot_qq_plot_basic(self, trained_model, sample_data):
        """Test basic Q-Q plot creation."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        ax = ModelVisualizer.plot_qq_plot(trained_model, X, y)

        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_plot_qq_plot_before_training(self, model, sample_data):
        """Test error when plotting Q-Q before training."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="n'a pas encore été entraîné"):
            ModelVisualizer.plot_qq_plot(model, X, y)

    def test_plot_qq_plot_empty_X(self, trained_model):
        """Test error handling with empty X."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            ModelVisualizer.plot_qq_plot(trained_model, X, y)

    def test_plot_qq_plot_dimension_mismatch(self, trained_model, sample_data):
        """Test error handling with mismatched dimensions."""
        X = sample_data.drop(columns=['note'])
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="même nombre d'échantillons"):
            ModelVisualizer.plot_qq_plot(trained_model, X, y)