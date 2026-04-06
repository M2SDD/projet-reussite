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
import json
import os

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ----------------------------------------------------------------------------------------------------------------------
# Classe Config
# ----------------------------------------------------------------------------------------------------------------------
class Config:
    """
    Classe de configuration pour centraliser les paramètres de l'application
    """

    # Data file paths
    LOGS_FILE_PATH = 'data/logs.csv'
    NOTES_FILE_PATH = 'data/notes.csv'
    OUTPUT_DIR = 'output'

    # Note range validation bounds (French grading system)
    NOTE_MIN = 0
    NOTE_MAX = 20

    # Risk assessment thresholds
    RISK_THRESHOLD_HIGH = 10  # Below this grade indicates high risk
    RISK_THRESHOLD_MEDIUM = 12  # Below this grade indicates medium risk

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

    # ML/Statistical analysis parameters
    TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% training, 20% testing
    CV_FOLDS = 5  # Number of folds for cross-validation
    RANDOM_STATE = 42  # Random seed for reproducibility

    # Visualization and plotting parameters
    PLOT_DPI = 300  # Resolution for saved plots
    PLOT_FIGSIZE = (10, 6)  # Default figure size (width, height) in inches
    PLOT_STYLE = 'seaborn-v0_8'  # Matplotlib style
    PLOT_COLOR_PALETTE = 'Set2'  # Default color palette
    PLOT_SAVE_FORMAT = 'png'  # Default format for saved plots
    PLOT_FONT_SIZE = 12  # Default font size for plots

    def __init__(self, config_file=None):
        """
        Initialise la configuration avec les paramètres par défaut.

        Args:
            config_file (str, optional): Chemin vers un fichier JSON ou YAML de configuration.
                                         Les valeurs du fichier surchargent les valeurs par défaut.
                                         Formats supportés: .json, .yaml, .yml

        Raises:
            ValueError: Si les paramètres de configuration sont invalides après chargement
        """
        if config_file is not None:
            try:
                _, ext = os.path.splitext(config_file)
                ext = ext.lower()

                with open(config_file, 'r', encoding='utf-8') as f:
                    if ext in ['.yaml', '.yml']:
                        if not YAML_AVAILABLE:
                            raise ImportError(
                                "PyYAML n'est pas installé. "
                                "Installez-le avec: pip install pyyaml"
                            )
                        config_data = yaml.safe_load(f)
                    elif ext == '.json':
                        config_data = json.load(f)
                    else:
                        # Default to JSON for backwards compatibility
                        config_data = json.load(f)

                # Override class attributes with values from config file
                for key, value in config_data.items():
                    setattr(self, key, value)
            except FileNotFoundError:
                # Gracefully fallback to defaults if config file is missing
                pass

        # Validate configuration parameters
        self._validate()

    def _validate(self):
        """
        Valide les paramètres de configuration.

        Vérifie que tous les paramètres respectent les contraintes attendues :
        - TRAIN_TEST_SPLIT_RATIO doit être strictement entre 0 et 1
        - CV_FOLDS doit être au moins 2
        - NOTE_MIN doit être inférieur à NOTE_MAX
        - PLOT_DPI doit être strictement positif
        - RISK_THRESHOLD_HIGH doit être inférieur à RISK_THRESHOLD_MEDIUM
        - Les seuils de risque doivent être dans la plage [NOTE_MIN, NOTE_MAX]
        - SESSION_GAP_MINUTES doit être strictement positif

        Raises:
            ValueError: Si un paramètre est invalide, avec un message descriptif
        """
        # Validate TRAIN_TEST_SPLIT_RATIO
        if not (0 < self.TRAIN_TEST_SPLIT_RATIO < 1):
            raise ValueError(
                f"TRAIN_TEST_SPLIT_RATIO doit être strictement entre 0 et 1, "
                f"reçu: {self.TRAIN_TEST_SPLIT_RATIO}"
            )

        # Validate CV_FOLDS
        if self.CV_FOLDS < 2:
            raise ValueError(
                f"CV_FOLDS doit être au moins 2, reçu: {self.CV_FOLDS}"
            )

        # Validate NOTE_MIN and NOTE_MAX
        if self.NOTE_MIN >= self.NOTE_MAX:
            raise ValueError(
                f"NOTE_MIN ({self.NOTE_MIN}) doit être inférieur à "
                f"NOTE_MAX ({self.NOTE_MAX})"
            )

        # Validate PLOT_DPI
        if self.PLOT_DPI <= 0:
            raise ValueError(
                f"PLOT_DPI doit être strictement positif, reçu: {self.PLOT_DPI}"
            )

        # Validate risk thresholds are within note range
        if not (self.NOTE_MIN <= self.RISK_THRESHOLD_HIGH <= self.NOTE_MAX):
            raise ValueError(
                f"RISK_THRESHOLD_HIGH ({self.RISK_THRESHOLD_HIGH}) doit être "
                f"dans la plage [{self.NOTE_MIN}, {self.NOTE_MAX}]"
            )

        if not (self.NOTE_MIN <= self.RISK_THRESHOLD_MEDIUM <= self.NOTE_MAX):
            raise ValueError(
                f"RISK_THRESHOLD_MEDIUM ({self.RISK_THRESHOLD_MEDIUM}) doit être "
                f"dans la plage [{self.NOTE_MIN}, {self.NOTE_MAX}]"
            )

        # Validate risk threshold ordering
        if self.RISK_THRESHOLD_HIGH >= self.RISK_THRESHOLD_MEDIUM:
            raise ValueError(
                f"RISK_THRESHOLD_HIGH ({self.RISK_THRESHOLD_HIGH}) doit être "
                f"inférieur à RISK_THRESHOLD_MEDIUM ({self.RISK_THRESHOLD_MEDIUM})"
            )

        # Validate SESSION_GAP_MINUTES
        if self.SESSION_GAP_MINUTES <= 0:
            raise ValueError(
                f"SESSION_GAP_MINUTES doit être strictement positif, "
                f"reçu: {self.SESSION_GAP_MINUTES}"
            )
