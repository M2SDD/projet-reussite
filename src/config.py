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

                # Post-process: Convert lists to tuples where needed
                # JSON arrays are loaded as lists, but some config values need to be tuples
                if hasattr(self, 'PLOT_FIGSIZE') and isinstance(self.PLOT_FIGSIZE, list):
                    self.PLOT_FIGSIZE = tuple(self.PLOT_FIGSIZE)
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
            TypeError: Si un paramètre n'a pas le type attendu
        """
        # Type checking for numeric parameters
        if not isinstance(self.TRAIN_TEST_SPLIT_RATIO, (int, float)):
            raise TypeError(
                f"TRAIN_TEST_SPLIT_RATIO doit être un nombre (int ou float), "
                f"reçu: {type(self.TRAIN_TEST_SPLIT_RATIO).__name__}"
            )

        if not isinstance(self.CV_FOLDS, int):
            raise TypeError(
                f"CV_FOLDS doit être un entier (int), "
                f"reçu: {type(self.CV_FOLDS).__name__}"
            )

        if not isinstance(self.RANDOM_STATE, int):
            raise TypeError(
                f"RANDOM_STATE doit être un entier (int), "
                f"reçu: {type(self.RANDOM_STATE).__name__}"
            )

        if not isinstance(self.NOTE_MIN, (int, float)):
            raise TypeError(
                f"NOTE_MIN doit être un nombre (int ou float), "
                f"reçu: {type(self.NOTE_MIN).__name__}"
            )

        if not isinstance(self.NOTE_MAX, (int, float)):
            raise TypeError(
                f"NOTE_MAX doit être un nombre (int ou float), "
                f"reçu: {type(self.NOTE_MAX).__name__}"
            )

        if not isinstance(self.RISK_THRESHOLD_HIGH, (int, float)):
            raise TypeError(
                f"RISK_THRESHOLD_HIGH doit être un nombre (int ou float), "
                f"reçu: {type(self.RISK_THRESHOLD_HIGH).__name__}"
            )

        if not isinstance(self.RISK_THRESHOLD_MEDIUM, (int, float)):
            raise TypeError(
                f"RISK_THRESHOLD_MEDIUM doit être un nombre (int ou float), "
                f"reçu: {type(self.RISK_THRESHOLD_MEDIUM).__name__}"
            )

        if not isinstance(self.PLOT_DPI, int):
            raise TypeError(
                f"PLOT_DPI doit être un entier (int), "
                f"reçu: {type(self.PLOT_DPI).__name__}"
            )

        if not isinstance(self.PLOT_FONT_SIZE, int):
            raise TypeError(
                f"PLOT_FONT_SIZE doit être un entier (int), "
                f"reçu: {type(self.PLOT_FONT_SIZE).__name__}"
            )

        if not isinstance(self.SESSION_GAP_MINUTES, int):
            raise TypeError(
                f"SESSION_GAP_MINUTES doit être un entier (int), "
                f"reçu: {type(self.SESSION_GAP_MINUTES).__name__}"
            )

        # Type checking for string parameters
        if not isinstance(self.LOGS_FILE_PATH, str):
            raise TypeError(
                f"LOGS_FILE_PATH doit être une chaîne (str), "
                f"reçu: {type(self.LOGS_FILE_PATH).__name__}"
            )

        if not isinstance(self.NOTES_FILE_PATH, str):
            raise TypeError(
                f"NOTES_FILE_PATH doit être une chaîne (str), "
                f"reçu: {type(self.NOTES_FILE_PATH).__name__}"
            )

        if not isinstance(self.OUTPUT_DIR, str):
            raise TypeError(
                f"OUTPUT_DIR doit être une chaîne (str), "
                f"reçu: {type(self.OUTPUT_DIR).__name__}"
            )

        if not isinstance(self.DATETIME_FORMAT, str):
            raise TypeError(
                f"DATETIME_FORMAT doit être une chaîne (str), "
                f"reçu: {type(self.DATETIME_FORMAT).__name__}"
            )

        if not isinstance(self.DUPLICATE_KEEP, str):
            raise TypeError(
                f"DUPLICATE_KEEP doit être une chaîne (str), "
                f"reçu: {type(self.DUPLICATE_KEEP).__name__}"
            )

        if not isinstance(self.PLOT_STYLE, str):
            raise TypeError(
                f"PLOT_STYLE doit être une chaîne (str), "
                f"reçu: {type(self.PLOT_STYLE).__name__}"
            )

        if not isinstance(self.PLOT_COLOR_PALETTE, str):
            raise TypeError(
                f"PLOT_COLOR_PALETTE doit être une chaîne (str), "
                f"reçu: {type(self.PLOT_COLOR_PALETTE).__name__}"
            )

        if not isinstance(self.PLOT_SAVE_FORMAT, str):
            raise TypeError(
                f"PLOT_SAVE_FORMAT doit être une chaîne (str), "
                f"reçu: {type(self.PLOT_SAVE_FORMAT).__name__}"
            )

        # Type checking for boolean parameters
        if not isinstance(self.FEATURE_HOUR_OF_DAY, bool):
            raise TypeError(
                f"FEATURE_HOUR_OF_DAY doit être un booléen (bool), "
                f"reçu: {type(self.FEATURE_HOUR_OF_DAY).__name__}"
            )

        if not isinstance(self.FEATURE_DAY_OF_WEEK, bool):
            raise TypeError(
                f"FEATURE_DAY_OF_WEEK doit être un booléen (bool), "
                f"reçu: {type(self.FEATURE_DAY_OF_WEEK).__name__}"
            )

        if not isinstance(self.FEATURE_SESSION_COUNT, bool):
            raise TypeError(
                f"FEATURE_SESSION_COUNT doit être un booléen (bool), "
                f"reçu: {type(self.FEATURE_SESSION_COUNT).__name__}"
            )

        if not isinstance(self.FEATURE_TOTAL_EVENTS, bool):
            raise TypeError(
                f"FEATURE_TOTAL_EVENTS doit être un booléen (bool), "
                f"reçu: {type(self.FEATURE_TOTAL_EVENTS).__name__}"
            )

        # Type checking for collection parameters
        if not isinstance(self.PLOT_FIGSIZE, tuple):
            raise TypeError(
                f"PLOT_FIGSIZE doit être un tuple, "
                f"reçu: {type(self.PLOT_FIGSIZE).__name__}"
            )

        if not isinstance(self.FEATURE_EVENT_TYPES, list):
            raise TypeError(
                f"FEATURE_EVENT_TYPES doit être une liste (list), "
                f"reçu: {type(self.FEATURE_EVENT_TYPES).__name__}"
            )

        if not isinstance(self.LOGS_COLUMN_MAPPING, dict):
            raise TypeError(
                f"LOGS_COLUMN_MAPPING doit être un dictionnaire (dict), "
                f"reçu: {type(self.LOGS_COLUMN_MAPPING).__name__}"
            )

        if not isinstance(self.NOTES_COLUMN_MAPPING, dict):
            raise TypeError(
                f"NOTES_COLUMN_MAPPING doit être un dictionnaire (dict), "
                f"reçu: {type(self.NOTES_COLUMN_MAPPING).__name__}"
            )

        if not isinstance(self.COMPOSANT_CATEGORIES, dict):
            raise TypeError(
                f"COMPOSANT_CATEGORIES doit être un dictionnaire (dict), "
                f"reçu: {type(self.COMPOSANT_CATEGORIES).__name__}"
            )

        if not isinstance(self.EVENEMENT_CATEGORIES, dict):
            raise TypeError(
                f"EVENEMENT_CATEGORIES doit être un dictionnaire (dict), "
                f"reçu: {type(self.EVENEMENT_CATEGORIES).__name__}"
            )

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

    def export_defaults(self, file_path):
        """
        Exporte la configuration actuelle vers un fichier JSON ou YAML.

        Cette méthode permet de sauvegarder tous les paramètres de configuration
        actuels dans un fichier, qui peut ensuite être rechargé via le paramètre
        config_file du constructeur.

        Args:
            file_path (str): Chemin du fichier de destination.
                            Le format est déterminé par l'extension (.json, .yaml, .yml).
                            Par défaut, utilise le format JSON.

        Raises:
            ImportError: Si le format YAML est demandé mais PyYAML n'est pas installé
            IOError: Si l'écriture du fichier échoue

        Example:
            >>> config = Config()
            >>> config.export_defaults('config.json')
            >>> # Le fichier config.json contient tous les paramètres de configuration
        """
        # Collect all configuration attributes
        config_data = {}
        for attr_name in dir(self):
            # Skip private/protected attributes and methods
            if attr_name.startswith('_'):
                continue

            # Skip methods
            attr_value = getattr(self, attr_name)
            if callable(attr_value):
                continue

            # Add to config data
            config_data[attr_name] = attr_value

        # Determine file format from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Write configuration to file
        with open(file_path, 'w', encoding='utf-8') as f:
            if ext in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "PyYAML n'est pas installé. "
                        "Installez-le avec: pip install pyyaml"
                    )
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                # Default to JSON format
                json.dump(config_data, f, indent=4, ensure_ascii=False)
