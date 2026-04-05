#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 08/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
DataVis sur les exports d'Arche anonymisés

__author__ = "Matthieu PELINGRE"
__copyright__ = "Informations de droits d'auteur"
__credits__ = ["Matthieu PELINGRE"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Matthieu PELINGRE"
__email__ = "matthieu.pelingre1@etu.univ-lorraine.fr"
__status__ = "Production"
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from src.data_loader import DataLoader


# ----------------------------------------------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------------------------------------------
# les constantes par défaut sont définies dans les src.*
LOGS_FILE = 'data/logs_info_25_pseudo.csv'
NOTES_FILE = 'data/notes_info_25_pseudo.csv'


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize DataLoader
    loader = DataLoader()

    # Load logs data
    print("Loading ARCHE logs data...")
    logs_df = loader.load_logs(LOGS_FILE)
    print("Logs loaded successfully")
    print(f"  - Total log entries: {len(logs_df)}")
    print(f"  - Columns: {', '.join(logs_df.columns)}")
    print(f"  - Date range: {logs_df['heure'].min()} to {logs_df['heure'].max()}")
    print()

    # Load notes data
    print("Loading ARCHE notes data...")
    notes_df = loader.load_notes(NOTES_FILE)
    print("Notes loaded successfully")
    print(f"  - Total notes: {len(notes_df)}")
    print(f"  - Columns: {', '.join(notes_df.columns)}")
    print(f"  - Unique students: {notes_df['pseudo'].nunique()}")
    print()

    # Display sample data
    print("Sample log entries:")
    print(logs_df.head(3))
    print()
    print("Sample notes:")
    print(notes_df.head(3))