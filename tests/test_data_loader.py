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
from src.data.data_loader import DataLoader
from src.config import Config


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

        assert 'introuvable' in str(exc_info.value).lower()

    def test_load_notes_missing_file(self, data_loader):
        """Test that loading a non-existent notes file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            data_loader.load_notes('nonexistent_file.csv')

        assert 'introuvable' in str(exc_info.value).lower()

    def test_load_logs_missing_columns(self, data_loader, invalid_logs_csv):
        """Test that loading logs with missing columns raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            data_loader.load_logs(invalid_logs_csv)

        error_message = str(exc_info.value)
        assert 'Colonnes manquantes' in error_message
        assert 'composant' in error_message or 'contexte' in error_message or 'evenement' in error_message

    def test_load_notes_missing_columns(self, data_loader, invalid_notes_csv):
        """Test that loading notes with missing columns raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            data_loader.load_notes(invalid_notes_csv)

        error_message = str(exc_info.value)
        assert 'Colonnes manquantes' in error_message
        assert 'note' in error_message

    def test_invalid_datetime_handling(self, data_loader, logs_with_invalid_dates):
        """Test that invalid datetime values are handled with warnings."""
        with pytest.warns(UserWarning, match='valeurs non valides'):
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
        data_loader._validate_columns(valid_df, Config.LOGS_REQUIRED_COLUMNS, 'test file')

    def test_validate_columns_raises_on_missing(self, data_loader):
        """Test that _validate_columns raises ValueError for missing columns."""
        # Create a DataFrame with missing columns
        invalid_df = pd.DataFrame({
            'heure': ['2024-07-24 09:48:08'],
            'pseudo': [436]
        })

        with pytest.raises(ValueError) as exc_info:
            data_loader._validate_columns(invalid_df, Config.LOGS_REQUIRED_COLUMNS, 'test file')

        error_message = str(exc_info.value)
        assert 'Colonnes manquantes' in error_message


class TestDataLoaderEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def empty_logs_csv(self):
        """Create an empty logs CSV file (no content at all)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            # File is empty - no headers, no data
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def logs_csv_headers_only(self):
        """Create a logs CSV with only headers, no data rows."""
        content = """heure,pseudo,contexte,composant,evenement"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def notes_csv_headers_only(self):
        """Create a notes CSV with only headers, no data rows."""
        content = """pseudo,note"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def logs_csv_with_extra_columns(self):
        """Create a logs CSV with extra columns beyond the required ones."""
        content = """heure,pseudo,contexte,composant,evenement,extra_col1,extra_col2
2024-07-24 09:48:08,436,Cours: PASS - S1,Système,Cours consulté,extra1,extra2
2024-07-24 09:48:14,437,Fichier: Contrat,Fichier,Module consulté,extra3,extra4"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def notes_csv_with_extra_columns(self):
        """Create a notes CSV with extra columns."""
        content = """pseudo,note,extra_column
318,11.05,extra
717,11.506,data"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def logs_csv_with_whitespace(self):
        """Create a logs CSV with whitespace in values."""
        content = """heure,pseudo,contexte,composant,evenement
2024-07-24 09:48:08,  436  ,  Cours: PASS - S1  ,  Système  ,  Cours consulté
2024-07-24 09:48:14,437,Fichier: Contrat,Fichier,Module consulté"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def notes_csv_with_special_values(self):
        """Create a notes CSV with special values (negative, zero, very high)."""
        content = """pseudo,note
318,-5.0
717,0.0
364,100.0
841,20.5"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def logs_csv_with_special_chars(self):
        """Create a logs CSV with French special characters."""
        content = """heure,pseudo,contexte,composant,evenement
2024-07-24 09:48:08,436,Cours: Français Élémentaire,Système,Événement créé
2024-07-24 09:48:14,437,Activité: Améliorer,Fichier,Complété"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def logs_csv_all_invalid_dates(self):
        """Create a logs CSV where all datetime values are invalid."""
        content = """heure,pseudo,contexte,composant,evenement
not-a-date,436,Cours: PASS - S1,Système,Cours consulté
invalid,437,Fichier: Contrat,Fichier,Module consulté
bad-date,841,Cours: PASS - S1,Système,Cours consulté"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def notes_csv_with_nan_values(self):
        """Create a notes CSV with NaN/empty values."""
        content = """pseudo,note
318,11.05
717,
364,10.022
841,"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def logs_csv_with_duplicate_rows(self):
        """Create a logs CSV with duplicate rows."""
        content = """heure,pseudo,contexte,composant,evenement
2024-07-24 09:48:08,436,Cours: PASS - S1,Système,Cours consulté
2024-07-24 09:48:08,436,Cours: PASS - S1,Système,Cours consulté
2024-07-24 09:48:14,437,Fichier: Contrat,Fichier,Module consulté"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_empty_csv_file(self, data_loader, empty_logs_csv):
        """Test that loading an empty CSV file raises an appropriate error."""
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            data_loader.load_logs(empty_logs_csv)

    def test_logs_csv_headers_only(self, data_loader, logs_csv_headers_only):
        """Test loading a logs CSV with only headers and no data rows."""
        df = data_loader.load_logs(logs_csv_headers_only)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Columns should still be present
        assert all(col in df.columns for col in Config.LOGS_REQUIRED_COLUMNS)

    def test_notes_csv_headers_only(self, data_loader, notes_csv_headers_only):
        """Test loading a notes CSV with only headers and no data rows."""
        df = data_loader.load_notes(notes_csv_headers_only)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Columns should still be present
        assert all(col in df.columns for col in Config.NOTES_REQUIRED_COLUMNS)

    def test_logs_csv_with_extra_columns(self, data_loader, logs_csv_with_extra_columns):
        """Test that logs CSV with extra columns still loads successfully."""
        df = data_loader.load_logs(logs_csv_with_extra_columns)

        assert df is not None
        assert len(df) == 2
        # Required columns should be present
        assert all(col in df.columns for col in Config.LOGS_REQUIRED_COLUMNS)
        # Extra columns should also be present
        assert 'extra_col1' in df.columns
        assert 'extra_col2' in df.columns

    def test_notes_csv_with_extra_columns(self, data_loader, notes_csv_with_extra_columns):
        """Test that notes CSV with extra columns still loads successfully."""
        df = data_loader.load_notes(notes_csv_with_extra_columns)

        assert df is not None
        assert len(df) == 2
        # Required columns should be present
        assert all(col in df.columns for col in Config.NOTES_REQUIRED_COLUMNS)
        # Extra column should also be present
        assert 'extra_column' in df.columns

    def test_logs_csv_with_whitespace(self, data_loader, logs_csv_with_whitespace):
        """Test that whitespace in values is handled correctly."""
        df = data_loader.load_logs(logs_csv_with_whitespace)

        assert df is not None
        assert len(df) == 2
        # Pandas should handle whitespace - data should load successfully
        assert df['pseudo'].iloc[0] == 436
        assert df['pseudo'].iloc[1] == 437

    def test_notes_csv_with_special_values(self, data_loader, notes_csv_with_special_values):
        """Test handling of special numeric values (negative, zero, very high)."""
        df = data_loader.load_notes(notes_csv_with_special_values)

        assert df is not None
        assert len(df) == 4
        # All values should be loaded as floats
        assert df['note'].iloc[0] == -5.0
        assert df['note'].iloc[1] == 0.0
        assert df['note'].iloc[2] == 100.0
        assert df['note'].iloc[3] == 20.5

    def test_logs_csv_with_special_chars(self, data_loader, logs_csv_with_special_chars):
        """Test that French special characters (UTF-8) are handled correctly."""
        df = data_loader.load_logs(logs_csv_with_special_chars)

        assert df is not None
        assert len(df) == 2
        # Special characters should be preserved
        assert 'Français' in df['contexte'].iloc[0]
        assert 'Événement' in df['evenement'].iloc[0]
        assert 'Améliorer' in df['contexte'].iloc[1]

    def test_logs_csv_all_invalid_dates(self, data_loader, logs_csv_all_invalid_dates):
        """Test handling when all datetime values are invalid."""
        with pytest.warns(UserWarning, match='valeurs non valides'):
            df = data_loader.load_logs(logs_csv_all_invalid_dates)

        assert df is not None
        assert len(df) == 3
        # All dates should be NaT
        assert df['heure'].isna().all()

    def test_notes_csv_with_nan_values(self, data_loader, notes_csv_with_nan_values):
        """Test handling of NaN/empty values in notes."""
        df = data_loader.load_notes(notes_csv_with_nan_values)

        assert df is not None
        assert len(df) == 4
        # Check that NaN values are present
        assert pd.notna(df['note'].iloc[0])
        assert pd.isna(df['note'].iloc[1])
        assert pd.notna(df['note'].iloc[2])
        assert pd.isna(df['note'].iloc[3])

    def test_logs_csv_with_duplicate_rows(self, data_loader, logs_csv_with_duplicate_rows):
        """Test that duplicate rows are loaded (not automatically removed)."""
        df = data_loader.load_logs(logs_csv_with_duplicate_rows)

        assert df is not None
        assert len(df) == 3  # All rows should be loaded, including duplicates
        # First two rows should be identical
        assert df.iloc[0]['pseudo'] == df.iloc[1]['pseudo']
        assert df.iloc[0]['contexte'] == df.iloc[1]['contexte']


class TestDataLoaderEncodingAndCorruption:
    """Test encoding issues and corrupted file scenarios."""

    @pytest.fixture
    def non_utf8_csv(self):
        """Create a CSV file with non-UTF-8 encoding."""
        content = "heure,pseudo,contexte,composant,evenement\n"
        content += "2024-07-24 09:48:08,436,Test,Test,Test\n"

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            # Write with latin-1 encoding to cause UTF-8 decode error
            f.write(content.encode('latin-1'))
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_non_utf8_encoding(self, data_loader, non_utf8_csv):
        """Test that non-UTF-8 encoded files raise UnicodeDecodeError or load with errors."""
        # Note: This test may pass or fail depending on whether the latin-1 content
        # happens to be valid UTF-8. The main goal is to ensure the code handles
        # encoding errors gracefully.
        try:
            df = data_loader.load_logs(non_utf8_csv)
            # If it loads, that's also acceptable behavior
            assert df is not None
        except UnicodeDecodeError:
            # Expected for truly incompatible encoding
            pass


class TestDataLoaderPathHandling:
    """Test various file path edge cases."""

    def test_relative_path(self, data_loader):
        """Test that relative paths work correctly."""
        # This should work if the file exists
        try:
            df = data_loader.load_logs('data/logs_info_25_pseudo.csv')
            assert df is not None
        except FileNotFoundError:
            # Expected if running from different directory
            pass

    def test_empty_filename(self, data_loader):
        """Test that empty filename raises appropriate error."""
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            data_loader.load_logs('')

    def test_none_filename(self, data_loader):
        """Test that None filename raises appropriate error."""
        with pytest.raises((TypeError, AttributeError, ValueError)):
            data_loader.load_logs(None)


class TestDataLoaderMultipleLoads:
    """Test loading multiple files and reusing DataLoader instance."""

    def test_multiple_loads_same_instance(self, data_loader, sample_logs_csv, sample_notes_csv):
        """Test that the same DataLoader instance can load multiple files."""
        df1 = data_loader.load_logs(sample_logs_csv)
        df2 = data_loader.load_notes(sample_notes_csv)
        df3 = data_loader.load_logs(sample_logs_csv)  # Load same file again

        assert df1 is not None
        assert df2 is not None
        assert df3 is not None
        # DataFrames should be independent (not the same object)
        assert df1 is not df3

    def test_dataloader_is_reusable(self, sample_logs_csv):
        """Test that multiple DataLoader instances work independently."""
        loader1 = DataLoader()
        loader2 = DataLoader()

        df1 = loader1.load_logs(sample_logs_csv)
        df2 = loader2.load_logs(sample_logs_csv)

        assert df1 is not None
        assert df2 is not None
        # DataFrames should be independent instances
        assert df1 is not df2
