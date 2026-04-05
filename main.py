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