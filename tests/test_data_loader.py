"""
Unit tests for the DataLoader class.

Tests cover:
- Successful loading of logs and notes CSV files
- Column validation
- DateTime parsing
- Data types validation
- Row counts
- Error handling
"""

import pytest
import pandas as pd
import tempfile
import os
from src.data_loader import DataLoader


@pytest.fixture
def data_loader():
    """Create a DataLoader instance for testing."""
    return DataLoader()


@pytest.fixture
def sample_logs_csv():
    """Create a temporary logs CSV file for testing."""
    content = """heure,pseudo,contexte,composant,evenement
2024-07-24 09:48:08,436,Cours: PASS - S1,Système,Cours consulté
2024-07-24 09:48:14,436,Fichier: Contrat,Fichier,Module de cours consulté
2024-08-19 12:55:34,841,Cours: PASS - S1,Système,Cours consulté"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_notes_csv():
    """Create a temporary notes CSV file for testing."""
    content = """pseudo,note
318,11.05
717,11.506
364,10.022"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def invalid_logs_csv():
    """Create a logs CSV with missing required columns."""
    content = """heure,pseudo
2024-07-24 09:48:08,436
2024-07-24 09:48:14,437"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def invalid_notes_csv():
    """Create a notes CSV with missing required columns."""
    content = """pseudo
318
717"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def logs_with_invalid_dates():
    """Create a logs CSV with some invalid datetime values."""
    content = """heure,pseudo,contexte,composant,evenement
2024-07-24 09:48:08,436,Cours: PASS - S1,Système,Cours consulté
invalid-date,437,Fichier: Contrat,Fichier,Module consulté
2024-08-19 12:55:34,841,Cours: PASS - S1,Système,Cours consulté"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestDataLoaderBasicFunctionality:
    """Test basic DataLoader functionality with sample data."""

    def test_load_logs_success(self, data_loader, sample_logs_csv):
        """Test successful loading of logs CSV file."""
        df = data_loader.load_logs(sample_logs_csv)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_notes_success(self, data_loader, sample_notes_csv):
        """Test successful loading of notes CSV file."""
        df = data_loader.load_notes(sample_notes_csv)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_logs_column_names(self, data_loader, sample_logs_csv):
        """Test that logs DataFrame has correct column names."""
        df = data_loader.load_logs(sample_logs_csv)

        expected_columns = ['heure', 'pseudo', 'contexte', 'composant', 'evenement']
        assert all(col in df.columns for col in expected_columns)

    def test_notes_column_names(self, data_loader, sample_notes_csv):
        """Test that notes DataFrame has correct column names."""
        df = data_loader.load_notes(sample_notes_csv)

        expected_columns = ['pseudo', 'note']
        assert all(col in df.columns for col in expected_columns)

    def test_logs_datetime_parsing(self, data_loader, sample_logs_csv):
        """Test that datetime parsing works correctly for logs."""
        df = data_loader.load_logs(sample_logs_csv)

        # Check that heure column is datetime type
        assert df['heure'].dtype == 'datetime64[ns]'

        # Check that values are parsed correctly
        assert pd.notna(df['heure'].iloc[0])
        assert df['heure'].iloc[0] == pd.Timestamp('2024-07-24 09:48:08')

    def test_logs_data_types(self, data_loader, sample_logs_csv):
        """Test that logs DataFrame has correct data types."""
        df = data_loader.load_logs(sample_logs_csv)

        assert df['heure'].dtype == 'datetime64[ns]'
        assert df['pseudo'].dtype in ['int64', 'int32']
        # String columns can be 'object' or StringDtype
        assert df['contexte'].dtype == 'object' or pd.api.types.is_string_dtype(df['contexte'])
        assert df['composant'].dtype == 'object' or pd.api.types.is_string_dtype(df['composant'])
        assert df['evenement'].dtype == 'object' or pd.api.types.is_string_dtype(df['evenement'])

    def test_notes_data_types(self, data_loader, sample_notes_csv):
        """Test that notes DataFrame has correct data types."""
        df = data_loader.load_notes(sample_notes_csv)

        assert df['pseudo'].dtype in ['int64', 'int32']
        assert df['note'].dtype in ['float64', 'float32']


class TestDataLoaderErrorHandling:
    """Test error handling in DataLoader."""

    def test_load_logs_missing_file(self, data_loader):
        """Test that loading a non-existent logs file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            data_loader.load_logs('nonexistent_file.csv')

        assert 'not found' in str(exc_info.value).lower()

    def test_load_notes_missing_file(self, data_loader):
        """Test that loading a non-existent notes file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            data_loader.load_notes('nonexistent_file.csv')

        assert 'not found' in str(exc_info.value).lower()

    def test_load_logs_missing_columns(self, data_loader, invalid_logs_csv):
        """Test that loading logs with missing columns raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            data_loader.load_logs(invalid_logs_csv)

        error_message = str(exc_info.value)
        assert 'Missing required columns' in error_message
        assert 'composant' in error_message or 'contexte' in error_message or 'evenement' in error_message

    def test_load_notes_missing_columns(self, data_loader, invalid_notes_csv):
        """Test that loading notes with missing columns raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            data_loader.load_notes(invalid_notes_csv)

        error_message = str(exc_info.value)
        assert 'Missing required columns' in error_message
        assert 'note' in error_message

    def test_invalid_datetime_handling(self, data_loader, logs_with_invalid_dates):
        """Test that invalid datetime values are handled with warnings."""
        with pytest.warns(UserWarning, match='invalid datetime'):
            df = data_loader.load_logs(logs_with_invalid_dates)

        # Check that invalid dates are converted to NaT
        assert pd.isna(df['heure'].iloc[1])

        # Check that valid dates are still parsed correctly
        assert pd.notna(df['heure'].iloc[0])
        assert pd.notna(df['heure'].iloc[2])


class TestDataLoaderRealFiles:
    """Test DataLoader with actual CSV files in data/ directory."""

    def test_load_real_logs_file(self, data_loader):
        """Test loading the actual logs CSV file."""
        df = data_loader.load_logs('data/logs_info_25_pseudo.csv')

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Verify all required columns are present
        required_columns = ['heure', 'pseudo', 'contexte', 'composant', 'evenement']
        assert all(col in df.columns for col in required_columns)

        # Verify datetime parsing
        assert df['heure'].dtype == 'datetime64[ns]'

    def test_load_real_notes_file(self, data_loader):
        """Test loading the actual notes CSV file."""
        df = data_loader.load_notes('data/notes_info_25_pseudo.csv')

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Verify all required columns are present
        required_columns = ['pseudo', 'note']
        assert all(col in df.columns for col in required_columns)

    def test_real_files_row_counts(self, data_loader):
        """Test that real files have expected row counts."""
        logs_df = data_loader.load_logs('data/logs_info_25_pseudo.csv')
        notes_df = data_loader.load_notes('data/notes_info_25_pseudo.csv')

        # Both files should have data
        assert len(logs_df) > 0, "Logs file should not be empty"
        assert len(notes_df) > 0, "Notes file should not be empty"

        # Notes file should have fewer rows than logs (one grade per student vs many log entries)
        assert len(notes_df) < len(logs_df), "Notes file should have fewer rows than logs"


class TestDataLoaderValidation:
    """Test column validation logic."""

    def test_validate_columns_method(self, data_loader):
        """Test the _validate_columns method directly."""
        # Create a valid DataFrame
        valid_df = pd.DataFrame({
            'heure': ['2024-07-24 09:48:08'],
            'pseudo': [436],
            'contexte': ['Test'],
            'composant': ['Test'],
            'evenement': ['Test']
        })

        # Should not raise an error
        data_loader._validate_columns(valid_df, DataLoader.LOGS_REQUIRED_COLUMNS, 'test file')

    def test_validate_columns_raises_on_missing(self, data_loader):
        """Test that _validate_columns raises ValueError for missing columns."""
        # Create a DataFrame with missing columns
        invalid_df = pd.DataFrame({
            'heure': ['2024-07-24 09:48:08'],
            'pseudo': [436]
        })

        with pytest.raises(ValueError) as exc_info:
            data_loader._validate_columns(invalid_df, DataLoader.LOGS_REQUIRED_COLUMNS, 'test file')

        error_message = str(exc_info.value)
        assert 'Missing required columns' in error_message
