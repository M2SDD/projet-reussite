"""
DataLoader module for loading ARCHE CSV exports.

This module provides the DataLoader class that handles reading CSV files
from ARCHE (logs and notes) with proper encoding and column validation.
"""

import pandas as pd
import warnings


class DataLoader:
    """
    DataLoader class for loading and validating ARCHE CSV exports.

    This class handles loading of two types of CSV files:
    - logs_info_25_pseudo.csv: Student activity logs
    - notes_info_25_pseudo.csv: Student grades

    The loader validates that required columns are present and handles
    UTF-8 encoding for French characters.
    """

    # Expected column schemas for each CSV type
    LOGS_REQUIRED_COLUMNS = ['heure', 'pseudo', 'contexte', 'composant', 'evenement']
    NOTES_REQUIRED_COLUMNS = ['pseudo', 'note']

    def __init__(self):
        """Initialize the DataLoader."""
        pass

    def _validate_columns(self, df, required_columns, file_type):
        """
        Validate that all required columns are present in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate
            required_columns (list): List of required column names
            file_type (str): Type of file being validated (for error messages)

        Raises:
            ValueError: If any required columns are missing
        """
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {file_type}: {sorted(missing_columns)}. "
                f"Expected columns: {sorted(required_columns)}"
            )

    def load_logs(self, file_path):
        """
        Load the logs CSV file.

        Args:
            file_path (str): Path to the logs_info_25_pseudo.csv file

        Returns:
            pd.DataFrame: DataFrame containing the logs data with parsed datetime

        Raises:
            ValueError: If required columns are missing
        """
        # Load CSV with UTF-8 encoding for French characters
        df = pd.read_csv(file_path, encoding='utf-8')

        # Validate required columns
        self._validate_columns(df, self.LOGS_REQUIRED_COLUMNS, 'logs file')

        # Parse the 'heure' column to datetime with nanosecond resolution
        df['heure'] = pd.to_datetime(df['heure'], format='%Y-%m-%d %H:%M:%S', errors='coerce').astype('datetime64[ns]')

        # Check for any invalid datetime values that were coerced to NaT
        invalid_count = df['heure'].isna().sum()
        if invalid_count > 0:
            warnings.warn(
                f"Found {invalid_count} invalid datetime values in 'heure' column. "
                f"These have been set to NaT (Not a Time).",
                UserWarning
            )

        return df

    def load_notes(self, file_path):
        """
        Load the notes CSV file.

        Args:
            file_path (str): Path to the notes_info_25_pseudo.csv file

        Returns:
            pd.DataFrame: DataFrame containing the notes data

        Raises:
            ValueError: If required columns are missing
        """
        # Load CSV with UTF-8 encoding for French characters
        df = pd.read_csv(file_path, encoding='utf-8')

        # Validate required columns
        self._validate_columns(df, self.NOTES_REQUIRED_COLUMNS, 'notes file')

        return df
