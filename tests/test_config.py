"""
Unit tests for the Config class.

Tests cover:
- Default initialization
- Parameter types verification
- Configuration file loading (JSON and YAML)
- Validation logic
- Invalid parameter handling
- Export functionality
- Integration with other modules
- Edge cases
"""

import pytest
import json
import tempfile
import os
import warnings
from src.config import Config


def test_default_initialization():
    """Test that Config initializes with sensible defaults (standalone version)."""
    config = Config()

    # Verify all key parameters exist and have correct default values
    assert hasattr(config, 'LOGS_FILE_PATH')
    assert hasattr(config, 'NOTES_FILE_PATH')
    assert hasattr(config, 'OUTPUT_DIR')
    assert hasattr(config, 'TEST_SPLIT_RATIO')
    assert hasattr(config, 'CV_FOLDS')
    assert hasattr(config, 'RANDOM_STATE')
    assert hasattr(config, 'PLOT_DPI')
    assert hasattr(config, 'PLOT_FIGSIZE')
    assert hasattr(config, 'RISK_THRESHOLD_HIGH')
    assert hasattr(config, 'RISK_THRESHOLD_MEDIUM')

    # Verify default values
    assert config.LOGS_FILE_PATH == 'data/logs.csv'
    assert config.NOTES_FILE_PATH == 'data/notes.csv'
    assert config.TEST_SPLIT_RATIO == 0.2
    assert config.CV_FOLDS == 5
    assert config.PLOT_DPI == 300


def test_load_json_config():
    """Test loading configuration from JSON file."""
    config_data = {
        'PLOT_DPI': 600,
        'TEST_SPLIT_RATIO': 0.7,
        'CV_FOLDS': 10
    }
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_data, f)
    f.close()

    try:
        config = Config(config_file=f.name)

        # Verify that values from JSON file override defaults
        assert config.PLOT_DPI == 600
        assert config.TEST_SPLIT_RATIO == 0.7
        assert config.CV_FOLDS == 10

        # Verify that non-overridden values remain at defaults
        assert config.RANDOM_STATE == 42
        assert config.OUTPUT_DIR == 'output'
    finally:
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)


@pytest.fixture
def config():
    """Create a Config instance with default settings for testing."""
    return Config()


@pytest.fixture
def temp_json_config():
    """Create a temporary JSON config file for testing."""
    config_data = {
        'PLOT_DPI': 600,
        'TEST_SPLIT_RATIO': 0.7,
        'CV_FOLDS': 10
    }
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_data, f)
    f.close()
    yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_invalid_json_config():
    """Create a temporary JSON config file with invalid values."""
    config_data = {
        'TEST_SPLIT_RATIO': 1.5,  # Invalid: must be < 1
        'CV_FOLDS': 1,  # Invalid: must be >= 2
    }
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_data, f)
    f.close()
    yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


class TestConfigDefaultInitialization:
    """Test default initialization of Config class."""

    def test_default_initialization(self, config):
        """Test that Config initializes with sensible defaults."""
        # Data file paths
        assert hasattr(config, 'LOGS_FILE_PATH')
        assert hasattr(config, 'NOTES_FILE_PATH')
        assert hasattr(config, 'OUTPUT_DIR')
        assert config.LOGS_FILE_PATH == 'data/logs.csv'
        assert config.NOTES_FILE_PATH == 'data/notes.csv'
        assert config.OUTPUT_DIR == 'output'

    def test_default_note_validation_bounds(self, config):
        """Test that note validation bounds are set correctly."""
        assert hasattr(config, 'NOTE_MIN')
        assert hasattr(config, 'NOTE_MAX')
        assert config.NOTE_MIN == 0
        assert config.NOTE_MAX == 20
        assert config.NOTE_MIN < config.NOTE_MAX

    def test_default_risk_thresholds(self, config):
        """Test that risk thresholds are set with sensible defaults."""
        assert hasattr(config, 'RISK_THRESHOLD_HIGH')
        assert hasattr(config, 'RISK_THRESHOLD_MEDIUM')
        assert config.RISK_THRESHOLD_HIGH == 10
        assert config.RISK_THRESHOLD_MEDIUM == 12
        assert config.RISK_THRESHOLD_HIGH < config.RISK_THRESHOLD_MEDIUM

    def test_default_ml_parameters(self, config):
        """Test that ML parameters are initialized correctly."""
        assert hasattr(config, 'TEST_SPLIT_RATIO')
        assert hasattr(config, 'CV_FOLDS')
        assert hasattr(config, 'RANDOM_STATE')
        assert config.TEST_SPLIT_RATIO == 0.2
        assert config.CV_FOLDS == 5
        assert config.RANDOM_STATE == 42

    def test_default_visualization_parameters(self, config):
        """Test that visualization parameters are initialized correctly."""
        assert hasattr(config, 'PLOT_DPI')
        assert hasattr(config, 'PLOT_FIGSIZE')
        assert hasattr(config, 'PLOT_STYLE')
        assert hasattr(config, 'PLOT_COLOR_PALETTE')
        assert hasattr(config, 'PLOT_SAVE_FORMAT')
        assert hasattr(config, 'PLOT_FONT_SIZE')
        assert config.PLOT_DPI == 300
        assert config.PLOT_FIGSIZE == (10, 6)
        assert config.PLOT_STYLE == 'seaborn-v0_8'
        assert config.PLOT_COLOR_PALETTE == 'Set2'
        assert config.PLOT_SAVE_FORMAT == 'png'
        assert config.PLOT_FONT_SIZE == 12

    def test_default_feature_engineering_toggles(self, config):
        """Test that feature engineering toggles are set correctly."""
        assert hasattr(config, 'FEATURE_HOUR_OF_DAY')
        assert hasattr(config, 'FEATURE_DAY_OF_WEEK')
        assert hasattr(config, 'FEATURE_SESSION_COUNT')
        assert hasattr(config, 'FEATURE_TOTAL_EVENTS')
        assert config.FEATURE_HOUR_OF_DAY is True
        assert config.FEATURE_DAY_OF_WEEK is True
        assert config.FEATURE_SESSION_COUNT is True
        assert config.FEATURE_TOTAL_EVENTS is True

    def test_default_datetime_format(self, config):
        """Test that datetime format is set correctly."""
        assert hasattr(config, 'DATETIME_FORMAT')
        assert config.DATETIME_FORMAT == '%Y-%m-%d %H:%M:%S'

    def test_default_duplicate_removal_settings(self, config):
        """Test that duplicate removal settings are initialized."""
        assert hasattr(config, 'DUPLICATE_KEEP')
        assert hasattr(config, 'DUPLICATE_SUBSET')
        assert config.DUPLICATE_KEEP == 'first'
        assert config.DUPLICATE_SUBSET is None

    def test_default_session_gap_minutes(self, config):
        """Test that session gap setting is initialized."""
        assert hasattr(config, 'SESSION_GAP_MINUTES')
        assert config.SESSION_GAP_MINUTES == 30


class TestConfigParameterTypes:
    """Test that all configuration parameters have correct types."""

    def test_numeric_parameters_types(self, config):
        """Test that numeric parameters are correct types."""
        assert isinstance(config.TEST_SPLIT_RATIO, (int, float))
        assert isinstance(config.CV_FOLDS, int)
        assert isinstance(config.RANDOM_STATE, int)
        assert isinstance(config.NOTE_MIN, (int, float))
        assert isinstance(config.NOTE_MAX, (int, float))
        assert isinstance(config.RISK_THRESHOLD_HIGH, (int, float))
        assert isinstance(config.RISK_THRESHOLD_MEDIUM, (int, float))
        assert isinstance(config.PLOT_DPI, int)
        assert isinstance(config.PLOT_FONT_SIZE, int)
        assert isinstance(config.SESSION_GAP_MINUTES, int)

    def test_string_parameters_types(self, config):
        """Test that string parameters are correct types."""
        assert isinstance(config.LOGS_FILE_PATH, str)
        assert isinstance(config.NOTES_FILE_PATH, str)
        assert isinstance(config.OUTPUT_DIR, str)
        assert isinstance(config.DATETIME_FORMAT, str)
        assert isinstance(config.DUPLICATE_KEEP, str)
        assert isinstance(config.PLOT_STYLE, str)
        assert isinstance(config.PLOT_COLOR_PALETTE, str)
        assert isinstance(config.PLOT_SAVE_FORMAT, str)

    def test_boolean_parameters_types(self, config):
        """Test that boolean parameters are correct types."""
        assert isinstance(config.FEATURE_HOUR_OF_DAY, bool)
        assert isinstance(config.FEATURE_DAY_OF_WEEK, bool)
        assert isinstance(config.FEATURE_SESSION_COUNT, bool)
        assert isinstance(config.FEATURE_TOTAL_EVENTS, bool)

    def test_collection_parameters_types(self, config):
        """Test that collection parameters are correct types."""
        assert isinstance(config.PLOT_FIGSIZE, tuple)

    def test_plot_figsize_tuple_elements(self, config):
        """Test that PLOT_FIGSIZE tuple has correct structure."""
        assert len(config.PLOT_FIGSIZE) == 2
        assert isinstance(config.PLOT_FIGSIZE[0], (int, float))
        assert isinstance(config.PLOT_FIGSIZE[1], (int, float))


class TestConfigValidation:
    """Test configuration parameter validation and error handling."""

    def test_validation_passes_with_defaults(self, config):
        """Test that default values pass validation."""
        assert config is not None

    def test_validation_test_split_ratio_too_high(self):
        """Test that TEST_SPLIT_RATIO >= 1 raises ValueError."""
        config_data = {'TEST_SPLIT_RATIO': 1.5}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='TEST_SPLIT_RATIO doit être strictement entre 0 et 1'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_test_split_ratio_too_low(self):
        """Test that TEST_SPLIT_RATIO <= 0 raises ValueError."""
        config_data = {'TEST_SPLIT_RATIO': 0.0}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='TEST_SPLIT_RATIO doit être strictement entre 0 et 1'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_test_split_ratio_negative(self):
        """Test that negative TEST_SPLIT_RATIO raises ValueError."""
        config_data = {'TEST_SPLIT_RATIO': -0.5}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='TEST_SPLIT_RATIO doit être strictement entre 0 et 1'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_test_split_ratio_equals_one(self):
        """Test that TEST_SPLIT_RATIO == 1 raises ValueError."""
        config_data = {'TEST_SPLIT_RATIO': 1.0}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='TEST_SPLIT_RATIO doit être strictement entre 0 et 1'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_cv_folds_too_low(self):
        """Test that CV_FOLDS < 2 raises ValueError."""
        config_data = {'CV_FOLDS': 1}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='CV_FOLDS doit être au moins 2'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_cv_folds_zero(self):
        """Test that CV_FOLDS = 0 raises ValueError."""
        config_data = {'CV_FOLDS': 0}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='CV_FOLDS doit être au moins 2'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_cv_folds_negative(self):
        """Test that negative CV_FOLDS raises ValueError."""
        config_data = {'CV_FOLDS': -5}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='CV_FOLDS doit être au moins 2'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_note_min_greater_than_max(self):
        """Test that NOTE_MIN >= NOTE_MAX raises ValueError."""
        config_data = {'NOTE_MIN': 20, 'NOTE_MAX': 10}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='NOTE_MIN .* doit être inférieur à NOTE_MAX'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_note_min_equal_to_max(self):
        """Test that NOTE_MIN == NOTE_MAX raises ValueError."""
        config_data = {'NOTE_MIN': 15, 'NOTE_MAX': 15}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='NOTE_MIN .* doit être inférieur à NOTE_MAX'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_plot_dpi_zero(self):
        """Test that PLOT_DPI = 0 raises ValueError."""
        config_data = {'PLOT_DPI': 0}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='PLOT_DPI doit être strictement positif'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_plot_dpi_negative(self):
        """Test that negative PLOT_DPI raises ValueError."""
        config_data = {'PLOT_DPI': -100}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='PLOT_DPI doit être strictement positif'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_risk_threshold_high_out_of_range_low(self):
        """Test that RISK_THRESHOLD_HIGH below NOTE_MIN raises ValueError."""
        config_data = {'RISK_THRESHOLD_HIGH': -5}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='RISK_THRESHOLD_HIGH .* doit être dans la plage'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_risk_threshold_high_out_of_range_high(self):
        """Test that RISK_THRESHOLD_HIGH above NOTE_MAX raises ValueError."""
        config_data = {'RISK_THRESHOLD_HIGH': 25}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='RISK_THRESHOLD_HIGH .* doit être dans la plage'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_risk_threshold_medium_out_of_range(self):
        """Test that RISK_THRESHOLD_MEDIUM outside range raises ValueError."""
        config_data = {'RISK_THRESHOLD_MEDIUM': 25}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='RISK_THRESHOLD_MEDIUM .* doit être dans la plage'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_risk_thresholds_wrong_order(self):
        """Test that RISK_THRESHOLD_HIGH >= RISK_THRESHOLD_MEDIUM raises ValueError."""
        config_data = {'RISK_THRESHOLD_HIGH': 15, 'RISK_THRESHOLD_MEDIUM': 10}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='RISK_THRESHOLD_HIGH .* doit être inférieur à RISK_THRESHOLD_MEDIUM'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_risk_thresholds_equal(self):
        """Test that RISK_THRESHOLD_HIGH == RISK_THRESHOLD_MEDIUM raises ValueError."""
        config_data = {'RISK_THRESHOLD_HIGH': 12, 'RISK_THRESHOLD_MEDIUM': 12}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='RISK_THRESHOLD_HIGH .* doit être inférieur à RISK_THRESHOLD_MEDIUM'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_session_gap_minutes_zero(self):
        """Test that SESSION_GAP_MINUTES = 0 raises ValueError."""
        config_data = {'SESSION_GAP_MINUTES': 0}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='SESSION_GAP_MINUTES doit être strictement positif'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_session_gap_minutes_negative(self):
        """Test that negative SESSION_GAP_MINUTES raises ValueError."""
        config_data = {'SESSION_GAP_MINUTES': -10}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='SESSION_GAP_MINUTES doit être strictement positif'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_type_error_test_split_ratio(self):
        """Test that non-numeric TEST_SPLIT_RATIO raises TypeError."""
        config_data = {'TEST_SPLIT_RATIO': 'invalid'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='TEST_SPLIT_RATIO doit être un nombre'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_type_error_cv_folds(self):
        """Test that non-integer CV_FOLDS raises TypeError."""
        config_data = {'CV_FOLDS': 5.5}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='CV_FOLDS doit être un entier'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_type_error_random_state(self):
        """Test that non-integer RANDOM_STATE raises TypeError."""
        config_data = {'RANDOM_STATE': '42'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='RANDOM_STATE doit être un entier'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_type_error_plot_dpi(self):
        """Test that non-integer PLOT_DPI raises TypeError."""
        config_data = {'PLOT_DPI': 300.5}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='PLOT_DPI doit être un entier'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_type_error_logs_file_path(self):
        """Test that non-string LOGS_FILE_PATH raises TypeError."""
        config_data = {'LOGS_FILE_PATH': 123}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='LOGS_FILE_PATH doit être une chaîne'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_type_error_feature_hour_of_day(self):
        """Test that non-boolean FEATURE_HOUR_OF_DAY raises TypeError."""
        config_data = {'FEATURE_HOUR_OF_DAY': 'true'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='FEATURE_HOUR_OF_DAY doit être un booléen'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_type_error_plot_figsize(self):
        """Test that non-tuple PLOT_FIGSIZE raises TypeError."""
        config_data = {'PLOT_FIGSIZE': '(10, 6)'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='PLOT_FIGSIZE doit être un tuple'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_edge_case_train_test_split_boundary_low(self):
        """Test boundary value just above 0 for TEST_SPLIT_RATIO."""
        config_data = {'TEST_SPLIT_RATIO': 0.001}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.TEST_SPLIT_RATIO == 0.001
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_edge_case_test_split_boundary_high(self):
        """Test boundary value just below 1 for TEST_SPLIT_RATIO."""
        config_data = {'TEST_SPLIT_RATIO': 0.999}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.TEST_SPLIT_RATIO == 0.999
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_edge_case_cv_folds_minimum(self):
        """Test minimum valid CV_FOLDS value."""
        config_data = {'CV_FOLDS': 2}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.CV_FOLDS == 2
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_edge_case_risk_thresholds_at_boundaries(self):
        """Test risk thresholds at note boundaries."""
        config_data = {
            'NOTE_MIN': 0,
            'NOTE_MAX': 20,
            'RISK_THRESHOLD_HIGH': 0,
            'RISK_THRESHOLD_MEDIUM': 20
        }
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.RISK_THRESHOLD_HIGH == 0
            assert config.RISK_THRESHOLD_MEDIUM == 20
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_edge_case_session_gap_minimum(self):
        """Test minimum valid SESSION_GAP_MINUTES value."""
        config_data = {'SESSION_GAP_MINUTES': 1}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.SESSION_GAP_MINUTES == 1
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_multiple_errors_raises_first(self):
        """Test that first validation error is raised when multiple errors exist."""
        config_data = {
            'TEST_SPLIT_RATIO': 'invalid',  # TypeError (checked first)
            'CV_FOLDS': 1,  # ValueError
            'PLOT_DPI': -100  # ValueError
        }
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_valid_custom_configuration(self):
        """Test that a valid custom configuration passes all validation."""
        config_data = {
            'TEST_SPLIT_RATIO': 0.75,
            'CV_FOLDS': 10,
            'PLOT_DPI': 600,
            'RISK_THRESHOLD_HIGH': 8,
            'RISK_THRESHOLD_MEDIUM': 13,
            'SESSION_GAP_MINUTES': 45
        }
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.TEST_SPLIT_RATIO == 0.75
            assert config.CV_FOLDS == 10
            assert config.PLOT_DPI == 600
            assert config.RISK_THRESHOLD_HIGH == 8
            assert config.RISK_THRESHOLD_MEDIUM == 13
            assert config.SESSION_GAP_MINUTES == 45
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)


class TestConfigFileLoading:
    """Test configuration file loading functionality."""

    def test_load_from_json_file(self, temp_json_config):
        """Test loading configuration from JSON file."""
        config = Config(config_file=temp_json_config)
        assert config.PLOT_DPI == 600
        assert config.TEST_SPLIT_RATIO == 0.7
        assert config.CV_FOLDS == 10

    def test_missing_config_file_uses_defaults(self):
        """Test that missing config file falls back to defaults."""
        config = Config(config_file='nonexistent_file.json')
        assert config.PLOT_DPI == 300
        assert config.TEST_SPLIT_RATIO == 0.2
        assert config.CV_FOLDS == 5

    def test_partial_config_merges_with_defaults(self):
        """Test that partial config file merges with defaults."""
        config_data = {'PLOT_DPI': 600}  # Only override one parameter
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()
        try:
            config = Config(config_file=f.name)
            assert config.PLOT_DPI == 600  # Overridden
            assert config.CV_FOLDS == 5  # Still default
        finally:
            os.unlink(f.name)

    def test_list_to_tuple_conversion(self):
        """Test that JSON arrays are converted to tuples where needed."""
        config_data = {'PLOT_FIGSIZE': [12, 8]}  # JSON array
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()
        try:
            config = Config(config_file=f.name)
            assert isinstance(config.PLOT_FIGSIZE, tuple)
            assert config.PLOT_FIGSIZE == (12, 8)
        finally:
            os.unlink(f.name)


class TestConfigExport:
    """Test configuration export functionality."""

    def test_save_to_file_to_json(self, config):
        """Test exporting configuration to JSON file."""
        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        try:
            config.save_to_file(f.name)
            assert os.path.exists(f.name)

            # Verify the exported file can be loaded
            with open(f.name, 'r', encoding='utf-8') as exported:
                exported_data = json.load(exported)

            assert 'PLOT_DPI' in exported_data
            assert 'TEST_SPLIT_RATIO' in exported_data
            assert exported_data['PLOT_DPI'] == 300
        finally:
            os.unlink(f.name)

    def test_export_creates_parent_directory(self, config):
        """Test that export creates parent directories if they don't exist."""
        temp_dir = tempfile.mkdtemp()
        try:
            nested_path = os.path.join(temp_dir, 'subdir', 'config.json')
            config.save_to_file(nested_path)
            assert os.path.exists(nested_path)
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

    def test_export_and_reload_roundtrip(self, config):
        """Test that exported config can be reloaded correctly."""
        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        try:
            config.save_to_file(f.name)
            config2 = Config(config_file=f.name)

            # Verify key parameters match
            assert config2.PLOT_DPI == config.PLOT_DPI
            assert config2.TEST_SPLIT_RATIO == config.TEST_SPLIT_RATIO
            assert config2.CV_FOLDS == config.CV_FOLDS
        finally:
            os.unlink(f.name)

    def test_export_includes_all_config_attributes(self, config):
        """Test that export includes all non-private, non-callable attributes."""
        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        try:
            config.save_to_file(f.name)

            with open(f.name, 'r', encoding='utf-8') as exported:
                exported_data = json.load(exported)

            # Verify key attributes are present
            assert 'LOGS_FILE_PATH' in exported_data
            assert 'NOTES_FILE_PATH' in exported_data
            assert 'OUTPUT_DIR' in exported_data
            assert 'TEST_SPLIT_RATIO' in exported_data
            assert 'CV_FOLDS' in exported_data
            assert 'RANDOM_STATE' in exported_data
            assert 'NOTE_MIN' in exported_data
            assert 'NOTE_MAX' in exported_data
            assert 'RISK_THRESHOLD_HIGH' in exported_data
            assert 'RISK_THRESHOLD_MEDIUM' in exported_data
            assert 'PLOT_DPI' in exported_data
            assert 'PLOT_FIGSIZE' in exported_data
            assert 'FEATURE_HOUR_OF_DAY' in exported_data
        finally:
            os.unlink(f.name)

    def test_export_with_modified_values(self, config):
        """Test that export correctly saves modified configuration values."""
        # Modify some config values
        config.PLOT_DPI = 600
        config.TEST_SPLIT_RATIO = 0.75
        config.CV_FOLDS = 10
        config.RANDOM_STATE = 123

        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        try:
            config.save_to_file(f.name)

            with open(f.name, 'r', encoding='utf-8') as exported:
                exported_data = json.load(exported)

            # Verify modified values are saved
            assert exported_data['PLOT_DPI'] == 600
            assert exported_data['TEST_SPLIT_RATIO'] == 0.75
            assert exported_data['CV_FOLDS'] == 10
            assert exported_data['RANDOM_STATE'] == 123
        finally:
            os.unlink(f.name)

    def test_export_preserves_data_types(self, config):
        """Test that export preserves different data types correctly."""
        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        try:
            config.save_to_file(f.name)

            with open(f.name, 'r', encoding='utf-8') as exported:
                exported_data = json.load(exported)

            # Verify string types
            assert isinstance(exported_data['LOGS_FILE_PATH'], str)
            assert isinstance(exported_data['OUTPUT_DIR'], str)
            assert isinstance(exported_data['DATETIME_FORMAT'], str)

            # Verify numeric types
            assert isinstance(exported_data['PLOT_DPI'], int)
            assert isinstance(exported_data['CV_FOLDS'], int)
            assert isinstance(exported_data['TEST_SPLIT_RATIO'], float)

            # Verify boolean types
            assert isinstance(exported_data['FEATURE_HOUR_OF_DAY'], bool)
            assert isinstance(exported_data['FEATURE_DAY_OF_WEEK'], bool)

            # Verify collection types
            assert isinstance(exported_data['PLOT_FIGSIZE'], list)  # Tuple exported as list in JSON
        finally:
            os.unlink(f.name)

    def test_export_yaml_format_error_without_pyyaml(self, config):
        """Test that exporting to YAML without PyYAML raises ImportError."""
        f = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        f.close()
        try:
            # Try to import yaml to check if it's available
            try:
                import yaml
                pytest.skip("PyYAML is installed, skipping this test")
            except ImportError:
                # PyYAML not available, test should raise ImportError
                with pytest.raises(ImportError, match='PyYAML'):
                    config.save_to_file(f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_export_default_format_is_json(self, config):
        """Test that export defaults to JSON format when extension is not recognized."""
        f = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        f.close()
        try:
            config.save_to_file(f.name)
            assert os.path.exists(f.name)

            # Should be valid JSON
            with open(f.name, 'r', encoding='utf-8') as exported:
                exported_data = json.load(exported)

            assert 'PLOT_DPI' in exported_data
        finally:
            os.unlink(f.name)


class TestConfigIntegration:
    """Test configuration integration with other modules."""

    def test_config_with_data_processor(self):
        """Test that Config integrates correctly with DataProcessor."""
        from src.data_processor import DataProcessor

        config = Config()
        processor = DataProcessor(config=config)

        # Verify processor uses the config
        assert processor.config is config
        assert processor.config.NOTE_MIN == config.NOTE_MIN
        assert processor.config.NOTE_MAX == config.NOTE_MAX

    def test_config_with_data_processor_custom_values(self):
        """Test DataProcessor with custom Config values."""
        from src.data_processor import DataProcessor

        config = Config()
        config.NOTE_MIN = 5
        config.NOTE_MAX = 15

        processor = DataProcessor(config=config)

        # Verify custom values are used
        assert processor.config.NOTE_MIN == 5
        assert processor.config.NOTE_MAX == 15

    def test_config_with_statistics_module(self):
        """Test that Config integrates correctly with StatisticsModule."""
        from data.statistics_module import StatisticsModule

        config = Config()
        stats_module = StatisticsModule(config=config)

        # Verify stats module uses the config
        assert stats_module.config is config
        assert stats_module.config.RISK_THRESHOLD_HIGH == config.RISK_THRESHOLD_HIGH
        assert stats_module.config.RISK_THRESHOLD_MEDIUM == config.RISK_THRESHOLD_MEDIUM

    def test_config_with_statistics_module_custom_thresholds(self):
        """Test StatisticsModule with custom Config threshold values."""
        from data.statistics_module import StatisticsModule

        config = Config()
        config.RISK_THRESHOLD_HIGH = 8
        config.RISK_THRESHOLD_MEDIUM = 11

        stats_module = StatisticsModule(config=config)

        # Verify custom thresholds are used
        assert stats_module.config.RISK_THRESHOLD_HIGH == 8
        assert stats_module.config.RISK_THRESHOLD_MEDIUM == 11

    def test_multiple_modules_share_config(self):
        """Test that multiple modules can share the same Config instance."""
        from src.data_processor import DataProcessor
        from data.statistics_module import StatisticsModule

        config = Config()
        config.RANDOM_STATE = 999

        processor = DataProcessor(config=config)
        stats_module = StatisticsModule(config=config)

        # Verify both modules use the same config instance
        assert processor.config is config
        assert stats_module.config is config
        assert processor.config.RANDOM_STATE == 999
        assert stats_module.config.RANDOM_STATE == 999

    def test_config_modification_affects_modules(self):
        """Test that modifying Config affects modules that use it."""
        from src.data_processor import DataProcessor

        config = Config()
        processor = DataProcessor(config=config)

        # Modify config after processor creation
        config.NOTE_MIN = 3
        config.NOTE_MAX = 17

        # Verify processor sees the changes
        assert processor.config.NOTE_MIN == 3
        assert processor.config.NOTE_MAX == 17

    def test_config_export_and_module_integration(self):
        """Test full workflow: export config, reload it, and use with modules."""
        from src.data_processor import DataProcessor
        import tempfile

        # Create and modify config
        config1 = Config()
        config1.PLOT_DPI = 500
        config1.TEST_SPLIT_RATIO = 0.85
        config1.NOTE_MIN = 2
        config1.NOTE_MAX = 18

        # Export config
        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        try:
            config1.save_to_file(f.name)

            # Reload config
            config2 = Config(config_file=f.name)

            # Use reloaded config with DataProcessor
            processor = DataProcessor(config=config2)

            # Verify processor uses reloaded values
            assert processor.config.PLOT_DPI == 500
            assert processor.config.TEST_SPLIT_RATIO == 0.85
            assert processor.config.NOTE_MIN == 2
            assert processor.config.NOTE_MAX == 18
        finally:
            os.unlink(f.name)

    def test_config_with_real_data_processing(self):
        """Test Config integration with real data processing workflow."""
        from src.data_processor import DataProcessor
        import pandas as pd

        config = Config()
        config.NOTE_MIN = 0
        config.NOTE_MAX = 20

        processor = DataProcessor(config=config)

        # Create sample notes data
        notes_df = pd.DataFrame({
            'pseudo': [100, 101, 102],
            'note': [15.0, -5.0, 25.0]  # Include out-of-range values
        })

        # Process notes (should clip based on config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = processor.clean_notes(notes_df)

        # Verify clipping used config values
        assert cleaned['note'].min() >= config.NOTE_MIN
        assert cleaned['note'].max() <= config.NOTE_MAX

    def test_config_isolation_between_instances(self):
        """Test that different Config instances are isolated from each other."""
        from src.data_processor import DataProcessor

        config1 = Config()
        config1.NOTE_MIN = 5

        config2 = Config()
        config2.NOTE_MIN = 0

        processor1 = DataProcessor(config=config1)
        processor2 = DataProcessor(config=config2)

        # Verify each processor uses its own config
        assert processor1.config.NOTE_MIN == 5
        assert processor2.config.NOTE_MIN == 0
        assert processor1.config is not processor2.config


class TestConfigNewDataProcessingParameters:
    """Test new data processing parameters added for the refactoring."""

    def test_default_rapid_event_threshold(self, config):
        """Test that RAPID_EVENT_THRESHOLD_SECONDS has correct default."""
        assert hasattr(config, 'RAPID_EVENT_THRESHOLD_SECONDS')
        assert isinstance(config.RAPID_EVENT_THRESHOLD_SECONDS, (int, float))

    def test_default_outlier_removal_enabled(self, config):
        """Test that OUTLIER_REMOVAL_ENABLED has correct default."""
        assert hasattr(config, 'OUTLIER_REMOVAL_ENABLED')
        assert isinstance(config.OUTLIER_REMOVAL_ENABLED, bool)

    def test_default_na_fill_strategy(self, config):
        """Test that NA_FILL_STRATEGY has correct default."""
        assert hasattr(config, 'NA_FILL_STRATEGY')
        assert config.NA_FILL_STRATEGY == 'zero'
        assert isinstance(config.NA_FILL_STRATEGY, str)

    def test_validation_rapid_event_threshold_negative(self):
        """Test that negative RAPID_EVENT_THRESHOLD_SECONDS raises ValueError."""
        config_data = {'RAPID_EVENT_THRESHOLD_SECONDS': -1}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='RAPID_EVENT_THRESHOLD_SECONDS doit être positif ou nul'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_rapid_event_threshold_zero(self):
        """Test that RAPID_EVENT_THRESHOLD_SECONDS = 0 is valid (no filtering)."""
        config_data = {'RAPID_EVENT_THRESHOLD_SECONDS': 0}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.RAPID_EVENT_THRESHOLD_SECONDS == 0
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_rapid_event_threshold_type_error(self):
        """Test that non-numeric RAPID_EVENT_THRESHOLD_SECONDS raises TypeError."""
        config_data = {'RAPID_EVENT_THRESHOLD_SECONDS': 'fast'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='RAPID_EVENT_THRESHOLD_SECONDS doit être un nombre'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_outlier_removal_enabled_type_error(self):
        """Test that non-boolean OUTLIER_REMOVAL_ENABLED raises TypeError."""
        config_data = {'OUTLIER_REMOVAL_ENABLED': 'yes'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='OUTLIER_REMOVAL_ENABLED doit être un booléen'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_na_fill_strategy_invalid(self):
        """Test that invalid NA_FILL_STRATEGY raises ValueError."""
        config_data = {'NA_FILL_STRATEGY': 'interpolate'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(ValueError, match='NA_FILL_STRATEGY doit être l\'un de'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_validation_na_fill_strategy_type_error(self):
        """Test that non-string NA_FILL_STRATEGY raises TypeError."""
        config_data = {'NA_FILL_STRATEGY': 123}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            with pytest.raises(TypeError, match='NA_FILL_STRATEGY doit être une chaîne'):
                Config(config_file=f.name)
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_na_fill_strategy_valid_values(self):
        """Test that all valid NA_FILL_STRATEGY values are accepted."""
        for strategy in ['zero', 'mean', 'median']:
            config_data = {'NA_FILL_STRATEGY': strategy}
            f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(config_data, f)
            f.close()

            try:
                config = Config(config_file=f.name)
                assert config.NA_FILL_STRATEGY == strategy
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_rapid_event_threshold_loaded_from_json(self):
        """Test that RAPID_EVENT_THRESHOLD_SECONDS can be overridden from JSON."""
        config_data = {'RAPID_EVENT_THRESHOLD_SECONDS': 10}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()

        try:
            config = Config(config_file=f.name)
            assert config.RAPID_EVENT_THRESHOLD_SECONDS == 10
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)
