"""
Unit tests for the Config class.

Tests cover:
- Default initialization
- Parameter types verification
- Configuration file loading (JSON and YAML)
- Validation logic
- Invalid parameter handling
- Export functionality
- Edge cases
"""

import pytest
import json
import tempfile
import os
from src.config import Config


def test_default_initialization():
    """Test that Config initializes with sensible defaults (standalone version)."""
    config = Config()

    # Verify all key parameters exist and have correct default values
    assert hasattr(config, 'LOGS_FILE_PATH')
    assert hasattr(config, 'NOTES_FILE_PATH')
    assert hasattr(config, 'OUTPUT_DIR')
    assert hasattr(config, 'TRAIN_TEST_SPLIT_RATIO')
    assert hasattr(config, 'CV_FOLDS')
    assert hasattr(config, 'RANDOM_STATE')
    assert hasattr(config, 'PLOT_DPI')
    assert hasattr(config, 'PLOT_FIGSIZE')
    assert hasattr(config, 'RISK_THRESHOLD_HIGH')
    assert hasattr(config, 'RISK_THRESHOLD_MEDIUM')

    # Verify default values
    assert config.LOGS_FILE_PATH == 'data/logs.csv'
    assert config.NOTES_FILE_PATH == 'data/notes.csv'
    assert config.TRAIN_TEST_SPLIT_RATIO == 0.8
    assert config.CV_FOLDS == 5
    assert config.PLOT_DPI == 300


@pytest.fixture
def config():
    """Create a Config instance with default settings for testing."""
    return Config()


@pytest.fixture
def temp_json_config():
    """Create a temporary JSON config file for testing."""
    config_data = {
        'PLOT_DPI': 600,
        'TRAIN_TEST_SPLIT_RATIO': 0.7,
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
        'TRAIN_TEST_SPLIT_RATIO': 1.5,  # Invalid: must be < 1
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
        assert hasattr(config, 'TRAIN_TEST_SPLIT_RATIO')
        assert hasattr(config, 'CV_FOLDS')
        assert hasattr(config, 'RANDOM_STATE')
        assert config.TRAIN_TEST_SPLIT_RATIO == 0.8
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

    def test_default_column_mappings(self, config):
        """Test that column mappings are initialized correctly."""
        assert hasattr(config, 'LOGS_COLUMN_MAPPING')
        assert hasattr(config, 'NOTES_COLUMN_MAPPING')
        assert isinstance(config.LOGS_COLUMN_MAPPING, dict)
        assert isinstance(config.NOTES_COLUMN_MAPPING, dict)
        assert 'pseudo' in config.LOGS_COLUMN_MAPPING
        assert 'note' in config.NOTES_COLUMN_MAPPING

    def test_default_composant_categories(self, config):
        """Test that composant categories mapping is initialized."""
        assert hasattr(config, 'COMPOSANT_CATEGORIES')
        assert isinstance(config.COMPOSANT_CATEGORIES, dict)
        assert 'Système' in config.COMPOSANT_CATEGORIES

    def test_default_evenement_categories(self, config):
        """Test that evenement categories mapping is initialized."""
        assert hasattr(config, 'EVENEMENT_CATEGORIES')
        assert isinstance(config.EVENEMENT_CATEGORIES, dict)
        assert 'Cours consulté' in config.EVENEMENT_CATEGORIES

    def test_default_session_gap_minutes(self, config):
        """Test that session gap setting is initialized."""
        assert hasattr(config, 'SESSION_GAP_MINUTES')
        assert config.SESSION_GAP_MINUTES == 30

    def test_default_feature_event_types(self, config):
        """Test that feature event types list is initialized."""
        assert hasattr(config, 'FEATURE_EVENT_TYPES')
        assert isinstance(config.FEATURE_EVENT_TYPES, list)
        assert 'view' in config.FEATURE_EVENT_TYPES


class TestConfigParameterTypes:
    """Test that all configuration parameters have correct types."""

    def test_numeric_parameters_types(self, config):
        """Test that numeric parameters are correct types."""
        assert isinstance(config.TRAIN_TEST_SPLIT_RATIO, (int, float))
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
        assert isinstance(config.FEATURE_EVENT_TYPES, list)
        assert isinstance(config.LOGS_COLUMN_MAPPING, dict)
        assert isinstance(config.NOTES_COLUMN_MAPPING, dict)
        assert isinstance(config.COMPOSANT_CATEGORIES, dict)
        assert isinstance(config.EVENEMENT_CATEGORIES, dict)

    def test_plot_figsize_tuple_elements(self, config):
        """Test that PLOT_FIGSIZE tuple has correct structure."""
        assert len(config.PLOT_FIGSIZE) == 2
        assert isinstance(config.PLOT_FIGSIZE[0], (int, float))
        assert isinstance(config.PLOT_FIGSIZE[1], (int, float))


class TestConfigValidation:
    """Test configuration parameter validation."""

    def test_validation_passes_with_defaults(self, config):
        """Test that default values pass validation."""
        # If we got here without exception, validation passed
        assert config is not None

    def test_validation_fails_with_invalid_split_ratio_too_high(self, temp_invalid_json_config):
        """Test that validation fails when split ratio is >= 1."""
        with pytest.raises(ValueError, match="TRAIN_TEST_SPLIT_RATIO"):
            Config(config_file=temp_invalid_json_config)

    def test_validation_fails_with_invalid_cv_folds(self):
        """Test that validation fails when CV_FOLDS < 2."""
        config_data = {'CV_FOLDS': 1}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="CV_FOLDS"):
                Config(config_file=f.name)
        finally:
            os.unlink(f.name)

    def test_validation_fails_with_negative_dpi(self):
        """Test that validation fails when PLOT_DPI <= 0."""
        config_data = {'PLOT_DPI': -100}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="PLOT_DPI"):
                Config(config_file=f.name)
        finally:
            os.unlink(f.name)

    def test_validation_fails_with_invalid_note_range(self):
        """Test that validation fails when NOTE_MIN >= NOTE_MAX."""
        config_data = {'NOTE_MIN': 20, 'NOTE_MAX': 10}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="NOTE_MIN"):
                Config(config_file=f.name)
        finally:
            os.unlink(f.name)

    def test_validation_fails_with_invalid_risk_threshold_ordering(self):
        """Test that validation fails when RISK_THRESHOLD_HIGH >= RISK_THRESHOLD_MEDIUM."""
        config_data = {'RISK_THRESHOLD_HIGH': 15, 'RISK_THRESHOLD_MEDIUM': 10}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="RISK_THRESHOLD_HIGH"):
                Config(config_file=f.name)
        finally:
            os.unlink(f.name)

    def test_validation_fails_with_invalid_session_gap(self):
        """Test that validation fails when SESSION_GAP_MINUTES <= 0."""
        config_data = {'SESSION_GAP_MINUTES': -5}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="SESSION_GAP_MINUTES"):
                Config(config_file=f.name)
        finally:
            os.unlink(f.name)


class TestConfigFileLoading:
    """Test configuration file loading functionality."""

    def test_load_from_json_file(self, temp_json_config):
        """Test loading configuration from JSON file."""
        config = Config(config_file=temp_json_config)
        assert config.PLOT_DPI == 600
        assert config.TRAIN_TEST_SPLIT_RATIO == 0.7
        assert config.CV_FOLDS == 10

    def test_missing_config_file_uses_defaults(self):
        """Test that missing config file falls back to defaults."""
        config = Config(config_file='nonexistent_file.json')
        assert config.PLOT_DPI == 300
        assert config.TRAIN_TEST_SPLIT_RATIO == 0.8
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

    def test_export_defaults_to_json(self, config):
        """Test exporting configuration to JSON file."""
        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        try:
            config.export_defaults(f.name)
            assert os.path.exists(f.name)

            # Verify the exported file can be loaded
            with open(f.name, 'r', encoding='utf-8') as exported:
                exported_data = json.load(exported)

            assert 'PLOT_DPI' in exported_data
            assert 'TRAIN_TEST_SPLIT_RATIO' in exported_data
            assert exported_data['PLOT_DPI'] == 300
        finally:
            os.unlink(f.name)

    def test_export_creates_parent_directory(self, config):
        """Test that export creates parent directories if they don't exist."""
        temp_dir = tempfile.mkdtemp()
        try:
            nested_path = os.path.join(temp_dir, 'subdir', 'config.json')
            config.export_defaults(nested_path)
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
            config.export_defaults(f.name)
            config2 = Config(config_file=f.name)

            # Verify key parameters match
            assert config2.PLOT_DPI == config.PLOT_DPI
            assert config2.TRAIN_TEST_SPLIT_RATIO == config.TRAIN_TEST_SPLIT_RATIO
            assert config2.CV_FOLDS == config.CV_FOLDS
        finally:
            os.unlink(f.name)
