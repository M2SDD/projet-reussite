#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 08/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module de configuration pour l'application de visualisation de données

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


# ----------------------------------------------------------------------------------------------------------------------
# Classe Config
# ----------------------------------------------------------------------------------------------------------------------
class Config:
    """
    Classe de configuration pour centraliser les paramètres de l'application
    """

    # Note range validation bounds (French grading system)
    NOTE_MIN = 0
    NOTE_MAX = 20

    # Datetime format for parsing log timestamps
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    # Duplicate removal settings
    DUPLICATE_KEEP = 'first'
    DUPLICATE_SUBSET = None  # None means all columns

    # Column name mappings (raw CSV -> internal names)
    LOGS_COLUMN_MAPPING = {
        'heure': 'heure',
        'pseudo': 'pseudo',
        'contexte': 'contexte',
        'composant': 'composant',
        'evenement': 'evenement',
    }

    NOTES_COLUMN_MAPPING = {
        'pseudo': 'pseudo',
        'note': 'note',
    }

    # Feature engineering toggles
    FEATURE_HOUR_OF_DAY = True
    FEATURE_DAY_OF_WEEK = True
    FEATURE_SESSION_COUNT = True
    FEATURE_TOTAL_EVENTS = True

    # Engagement feature settings
    SESSION_GAP_MINUTES = 30  # Time gap to define new session
    FEATURE_EVENT_TYPES = ['view', 'submit', 'forum']  # Event types for engagement tracking

    # Composant category mappings
    COMPOSANT_CATEGORIES = {
        'Système': 'system',
        'Devoir': 'assignment',
        'Fichier': 'file',
        'Forum': 'forum',
        'Dossier': 'folder',
        'URL': 'url',
        'Page': 'page',
        'Cours': 'course',
    }

    # Evenement category mappings
    EVENEMENT_CATEGORIES = {
        'Cours consulté': 'view',
        'Module de cours consulté': 'view',
        'Activité de devoir consultée': 'view',
        'Un fichier a été déposé.': 'submit',
        'Une tentative a été soumise.': 'submit',
        'Discussion consultée': 'forum',
        'Message créé': 'forum',
    }

    # Regression model settings
    REGRESSION_TEST_SIZE = 0.2  # Default train/test split ratio (80/20)
    REGRESSION_RANDOM_STATE = 42  # Random seed for reproducibility
    REGRESSION_NORMALITY_ALPHA = 0.05  # Significance level for normality tests (Shapiro-Wilk)

    def __init__(self):
        """
        Initialise la configuration avec les paramètres par défaut.
        """
        pass
