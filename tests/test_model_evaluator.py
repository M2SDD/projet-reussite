"""
Unit tests for the ModelEvaluator class.

Tests cover:
- ModelEvaluator initialization
- Model registration
- Model addition with evaluation data
- Model evaluation (evaluate_all)
- Comparison table generation
- Predictions visualization
- Residuals visualization
- Metrics comparison visualization
- Model recommendation
- Results export functionality
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import os
from src.model_evaluator import ModelEvaluator
from src.regression_model import RegressionModel
from src.config import Config


@pytest.fixture
def evaluator():
    """Create a ModelEvaluator instance for testing."""
    return ModelEvaluator()


@pytest.fixture
def evaluator_with_config():
    """Create a ModelEvaluator instance with custom config."""
    config = Config()
    return ModelEvaluator(config=config)


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
def trained_model(sample_data):
    """Create a pre-trained RegressionModel for testing."""
    X = sample_data.drop(columns=['note'])
    y = sample_data['note']
    model = RegressionModel()
    model.fit(X, y)
    return model


@pytest.fixture
def second_trained_model(sample_data):
    """Create a second pre-trained RegressionModel for comparison testing."""
    # Use only first two features for a different model
    X = sample_data[['feature_1', 'feature_2']]
    y = sample_data['note']
    model = RegressionModel()
    model.fit(X, y)
    return model


@pytest.fixture
def evaluator_with_models(evaluator, trained_model, second_trained_model, sample_data):
    """Create a ModelEvaluator with two registered models."""
    X = sample_data.drop(columns=['note'])
    y = sample_data['note']
    X_reduced = sample_data[['feature_1', 'feature_2']]

    evaluator.add_model('model_full', trained_model, X, y)
    evaluator.add_model('model_reduced', second_trained_model, X_reduced, y)

    return evaluator


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame."""
    return pd.DataFrame()


class TestModelEvaluatorInitialization:
    """Test ModelEvaluator initialization."""

    def test_initialization(self, evaluator):
        """Test initialization with default config."""
        assert evaluator.config is not None
        assert isinstance(evaluator.config, Config)
        assert evaluator.models == {}
        assert len(evaluator.models) == 0

    def test_initialization_custom_config(self, evaluator_with_config):
        """Test initialization with custom config."""
        assert evaluator_with_config.config is not None
        assert isinstance(evaluator_with_config.config, Config)
        assert evaluator_with_config.models == {}

    def test_initialization_none_config(self):
        """Test initialization with None config creates default config."""
        evaluator = ModelEvaluator(config=None)
        assert evaluator.config is not None
        assert isinstance(evaluator.config, Config)
        assert evaluator.models == {}


class TestModelRegistration:
    """Test model registration functionality."""

    def test_register_model_basic(self, evaluator, trained_model):
        """Test basic model registration."""
        evaluator.register_model('test_model', trained_model)

        assert 'test_model' in evaluator.models
        assert evaluator.models['test_model']['model'] == trained_model
        assert evaluator.models['test_model']['X'] is None
        assert evaluator.models['test_model']['y'] is None

    def test_register_model_multiple(self, evaluator, trained_model, second_trained_model):
        """Test registering multiple models."""
        evaluator.register_model('model1', trained_model)
        evaluator.register_model('model2', second_trained_model)

        assert len(evaluator.models) == 2
        assert 'model1' in evaluator.models
        assert 'model2' in evaluator.models

    def test_register_model_empty_name(self, evaluator, trained_model):
        """Test error when registering with empty name."""
        with pytest.raises(ValueError, match="nom du modèle doit être une chaîne non vide"):
            evaluator.register_model('', trained_model)

    def test_register_model_none_name(self, evaluator, trained_model):
        """Test error when registering with None name."""
        with pytest.raises(ValueError, match="nom du modèle doit être une chaîne non vide"):
            evaluator.register_model(None, trained_model)

    def test_register_model_none_model(self, evaluator):
        """Test error when registering None model."""
        with pytest.raises(ValueError, match="modèle ne peut pas être None"):
            evaluator.register_model('test_model', None)

    def test_register_model_duplicate_name(self, evaluator, trained_model):
        """Test error when registering duplicate model name."""
        evaluator.register_model('test_model', trained_model)

        with pytest.raises(ValueError, match="modèle avec le nom 'test_model' existe déjà"):
            evaluator.register_model('test_model', trained_model)


class TestModelAddition:
    """Test model addition with evaluation data."""

    def test_add_model_basic(self, evaluator, trained_model, sample_data):
        """Test basic model addition with data."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)

        assert 'test_model' in evaluator.models
        assert evaluator.models['test_model']['model'] == trained_model
        pd.testing.assert_frame_equal(evaluator.models['test_model']['X'], X)
        pd.testing.assert_series_equal(evaluator.models['test_model']['y'], y)

    def test_add_model_multiple(self, evaluator, trained_model, second_trained_model, sample_data):
        """Test adding multiple models with data."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']
        X_reduced = sample_data[['feature_1', 'feature_2']]

        evaluator.add_model('model1', trained_model, X, y)
        evaluator.add_model('model2', second_trained_model, X_reduced, y)

        assert len(evaluator.models) == 2

    def test_add_model_empty_name(self, evaluator, trained_model, sample_data):
        """Test error when adding with empty name."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="nom du modèle doit être une chaîne non vide"):
            evaluator.add_model('', trained_model, X, y)

    def test_add_model_none_model(self, evaluator, sample_data):
        """Test error when adding None model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        with pytest.raises(ValueError, match="modèle ne peut pas être None"):
            evaluator.add_model('test_model', None, X, y)

    def test_add_model_empty_x(self, evaluator, trained_model, sample_data):
        """Test error when X is empty."""
        y = sample_data['note']
        X_empty = pd.DataFrame()

        with pytest.raises(ValueError, match="X ne peut pas être vide"):
            evaluator.add_model('test_model', trained_model, X_empty, y)

    def test_add_model_none_x(self, evaluator, trained_model, sample_data):
        """Test error when X is None."""
        y = sample_data['note']

        with pytest.raises(ValueError, match="X ne peut pas être vide"):
            evaluator.add_model('test_model', trained_model, None, y)

    def test_add_model_empty_y(self, evaluator, trained_model, sample_data):
        """Test error when y is empty."""
        X = sample_data.drop(columns=['note'])
        y_empty = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="y ne peut pas être vide"):
            evaluator.add_model('test_model', trained_model, X, y_empty)

    def test_add_model_none_y(self, evaluator, trained_model, sample_data):
        """Test error when y is None."""
        X = sample_data.drop(columns=['note'])

        with pytest.raises(ValueError, match="y ne peut pas être vide"):
            evaluator.add_model('test_model', trained_model, X, None)

    def test_add_model_mismatched_dimensions(self, evaluator, trained_model, sample_data):
        """Test error when X and y have different lengths."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note'].iloc[:5]  # Only first 5 samples

        with pytest.raises(ValueError, match="X et y doivent avoir le même nombre d'échantillons"):
            evaluator.add_model('test_model', trained_model, X, y)

    def test_add_model_duplicate_name(self, evaluator, trained_model, sample_data):
        """Test error when adding duplicate model name."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)

        with pytest.raises(ValueError, match="modèle avec le nom 'test_model' existe déjà"):
            evaluator.add_model('test_model', trained_model, X, y)


class TestEvaluateAll:
    """Test evaluate_all functionality."""

    def test_evaluate_all_single_model(self, evaluator, trained_model, sample_data):
        """Test evaluation of a single model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)
        results = evaluator.evaluate_all()

        assert 'test_model' in results
        assert 'r2' in results['test_model']
        assert 'rmse' in results['test_model']
        assert 'mae' in results['test_model']
        assert 'adjusted_r2' in results['test_model']

    def test_evaluate_all_multiple_models(self, evaluator_with_models):
        """Test evaluation of multiple models."""
        results = evaluator_with_models.evaluate_all()

        assert len(results) == 2
        assert 'model_full' in results
        assert 'model_reduced' in results

    def test_evaluate_all_no_models(self, evaluator):
        """Test error when no models have evaluation data."""
        with pytest.raises(ValueError, match="Aucun modèle avec données d'évaluation"):
            evaluator.evaluate_all()

    def test_evaluate_all_only_registered_models(self, evaluator, trained_model, sample_data):
        """Test that only models with data are evaluated."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        # Register one model without data
        evaluator.register_model('registered_only', trained_model)
        # Add one model with data
        evaluator.add_model('with_data', trained_model, X, y)

        results = evaluator.evaluate_all()

        assert 'with_data' in results
        assert 'registered_only' not in results

    def test_evaluate_all_metrics_validity(self, evaluator, trained_model, sample_data):
        """Test that metrics are valid numbers."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)
        results = evaluator.evaluate_all()

        metrics = results['test_model']

        # R² should be between -inf and 1
        assert metrics['r2'] <= 1.0

        # RMSE and MAE should be non-negative
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0

        # Adjusted R² should be between -inf and 1
        assert metrics['adjusted_r2'] <= 1.0


class TestComparisonTable:
    """Test get_comparison_table functionality."""

    def test_get_comparison_table_single_model(self, evaluator, trained_model, sample_data):
        """Test comparison table with single model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)
        table = evaluator.get_comparison_table()

        assert isinstance(table, pd.DataFrame)
        assert 'test_model' in table.index
        assert 'r2' in table.columns
        assert 'rmse' in table.columns
        assert 'mae' in table.columns
        assert 'adjusted_r2' in table.columns

    def test_get_comparison_table_multiple_models(self, evaluator_with_models):
        """Test comparison table with multiple models."""
        table = evaluator_with_models.get_comparison_table()

        assert len(table) == 2
        assert 'model_full' in table.index
        assert 'model_reduced' in table.index

    def test_get_comparison_table_no_models(self, evaluator):
        """Test error when no models have evaluation data."""
        with pytest.raises(ValueError, match="Aucun modèle avec données d'évaluation"):
            evaluator.get_comparison_table()

    def test_get_comparison_table_rounding(self, evaluator, trained_model, sample_data):
        """Test that table values are rounded to 4 decimals."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)
        table = evaluator.get_comparison_table()

        # Check that values have at most 4 decimal places
        for col in table.columns:
            for val in table[col]:
                # Convert to string and check decimal places
                val_str = f"{val:.4f}"
                assert len(val_str.split('.')[-1]) <= 4


class TestPlotPredictions:
    """Test plot_predictions functionality."""

    def test_plot_predictions_single_model(self, evaluator, trained_model, sample_data):
        """Test predictions plot with single model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)
        fig = evaluator.plot_predictions()

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_predictions_multiple_models(self, evaluator_with_models):
        """Test predictions plot with multiple models."""
        fig = evaluator_with_models.plot_predictions()

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_predictions_no_models(self, evaluator):
        """Test error when no models have evaluation data."""
        with pytest.raises(ValueError, match="Aucun modèle avec données d'évaluation"):
            evaluator.plot_predictions()


class TestPlotResiduals:
    """Test plot_residuals functionality."""

    def test_plot_residuals_single_model(self, evaluator, trained_model, sample_data):
        """Test residuals plot with single model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)
        fig = evaluator.plot_residuals()

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_residuals_multiple_models(self, evaluator_with_models):
        """Test residuals plot with multiple models."""
        fig = evaluator_with_models.plot_residuals()

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_residuals_no_models(self, evaluator):
        """Test error when no models have evaluation data."""
        with pytest.raises(ValueError, match="Aucun modèle avec données d'évaluation"):
            evaluator.plot_residuals()


class TestPlotMetricsComparison:
    """Test plot_metrics_comparison functionality."""

    def test_plot_metrics_comparison_basic(self, evaluator_with_models):
        """Test metrics comparison plot."""
        fig = evaluator_with_models.plot_metrics_comparison()

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_metrics_comparison_without_adjusted_r2(self, evaluator_with_models):
        """Test metrics comparison plot without adjusted R²."""
        fig = evaluator_with_models.plot_metrics_comparison(include_adjusted_r2=False)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_metrics_comparison_no_models(self, evaluator):
        """Test error when no models have evaluation data."""
        with pytest.raises(ValueError, match="Aucun modèle avec données d'évaluation"):
            evaluator.plot_metrics_comparison()


class TestGetRecommendation:
    """Test get_recommendation functionality."""

    def test_get_recommendation_basic(self, evaluator_with_models):
        """Test model recommendation."""
        recommendation = evaluator_with_models.get_recommendation()

        assert 'best_model' in recommendation
        assert 'reason' in recommendation
        assert 'metrics' in recommendation
        assert 'all_metrics' in recommendation

        assert recommendation['best_model'] in ['model_full', 'model_reduced']

    def test_get_recommendation_no_models(self, evaluator):
        """Test error when no models have evaluation data."""
        with pytest.raises(ValueError, match="Aucun modèle avec données d'évaluation"):
            evaluator.get_recommendation()

    def test_get_recommendation_single_model(self, evaluator, trained_model, sample_data):
        """Test recommendation with single model."""
        X = sample_data.drop(columns=['note'])
        y = sample_data['note']

        evaluator.add_model('test_model', trained_model, X, y)
        recommendation = evaluator.get_recommendation()

        assert recommendation['best_model'] == 'test_model'


class TestExportResults:
    """Test export_results functionality."""

    def test_export_results_basic(self, evaluator_with_models):
        """Test exporting results to directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator_with_models.export_results(tmpdir)

            # Check that files were created
            assert os.path.exists(os.path.join(tmpdir, 'comparison_table.csv'))
            assert os.path.exists(os.path.join(tmpdir, 'predictions_comparison.png'))
            assert os.path.exists(os.path.join(tmpdir, 'residuals_comparison.png'))
            assert os.path.exists(os.path.join(tmpdir, 'metrics_comparison.png'))

    def test_export_results_empty_dir(self, evaluator):
        """Test error when output_dir is empty."""
        with pytest.raises(ValueError, match="répertoire de sortie doit être une chaîne non vide"):
            evaluator.export_results('')

    def test_export_results_none_dir(self, evaluator):
        """Test error when output_dir is None."""
        with pytest.raises(ValueError, match="répertoire de sortie doit être une chaîne non vide"):
            evaluator.export_results(None)

    def test_export_results_no_models(self, evaluator):
        """Test error when no models have evaluation data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Aucun modèle avec données d'évaluation"):
                evaluator.export_results(tmpdir)

    def test_export_results_creates_directory(self, evaluator_with_models):
        """Test that export creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'new_subdir')
            evaluator_with_models.export_results(output_dir)

            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, 'comparison_table.csv'))
