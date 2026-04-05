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
from src import Config, DataLoader, DataProcessor, Visualizer


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
    # Initialisation de la configuration
    config = Config()

    # Initialisation des composants principaux
    data_loader = DataLoader()
    data_processor = DataProcessor()
    visualizer = Visualizer()

    # Vérification de l'architecture OOP
    print("✓ Tous les modules sont chargés avec succès")
    print(f"✓ Config: {type(config).__name__}")
    print(f"✓ DataLoader: {type(data_loader).__name__}")
    print(f"✓ DataProcessor: {type(data_processor).__name__}")
    print(f"✓ Visualizer: {type(visualizer).__name__}")
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