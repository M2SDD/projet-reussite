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
    Classe responsable de la gestion centralisée de la configuration de l'application.

    Cette classe fournit un point d'accès unique pour tous les paramètres de configuration,
    incluant les chemins de fichiers, les seuils de validation, les paramètres ML/statistiques,
    et les options de visualisation. Les paramètres peuvent être chargés depuis des fichiers
    JSON ou YAML, et sont automatiquement validés à l'initialisation.

    Attributes:
        LOGS_FILE_PATH (str): Chemin vers le fichier CSV des logs
        NOTES_FILE_PATH (str): Chemin vers le fichier CSV des notes
        OUTPUT_DIR (str): Répertoire de sortie pour les fichiers générés
        NOTE_MIN (int): Note minimale valide (système français: 0)
        NOTE_MAX (int): Note maximale valide (système français: 20)
        RISK_THRESHOLD_HIGH (int): Seuil de risque élevé (défaut: 10)
        RISK_THRESHOLD_MEDIUM (int): Seuil de risque moyen (défaut: 12)
        TEST_SPLIT_RATIO (float): Ratio de séparation train/test (défaut: 0.8)
        CV_FOLDS (int): Nombre de plis pour la validation croisée (défaut: 5)
        RANDOM_STATE (int): Graine aléatoire pour la reproductibilité (défaut: 42)
        RF_N_ESTIMATORS (int): Nombre d'arbres dans la forêt aléatoire (défaut: 100)
        RF_MAX_DEPTH (int or None): Profondeur maximale des arbres (défaut: None)
        RF_MIN_SAMPLES_SPLIT (int): Échantillons minimum pour diviser un noeud (défaut: 2)
        RF_MIN_SAMPLES_LEAF (int): Échantillons minimum dans une feuille (défaut: 1)
        RF_MAX_FEATURES (str or int): Features à considérer pour la meilleure division (défaut: 'sqrt')
        RF_BOOTSTRAP (bool): Utilisation de bootstrap pour construire les arbres (défaut: True)
        RF_CRITERION (str): Fonction pour mesurer la qualité d'une division (défaut: 'gini')
        GB_N_ESTIMATORS (int): Nombre d'étapes de boosting pour Gradient Boosting (défaut: 100)
        GB_LEARNING_RATE (float): Taux d'apprentissage pour Gradient Boosting (défaut: 0.1)
        GB_MAX_DEPTH (int): Profondeur maximale des estimateurs pour Gradient Boosting (défaut: 3)
        PLOT_DPI (int): Résolution des graphiques sauvegardés (défaut: 300)
        PLOT_FIGSIZE (tuple): Taille par défaut des figures (largeur, hauteur) en pouces

    Examples:
        Utilisation avec la configuration par défaut:

        >>> from src.config import Config
        >>> config = Config()
        >>> print(config.NOTE_MIN, config.NOTE_MAX)
        0 20
        >>> print(config.RISK_THRESHOLD_HIGH)
        10

        Chargement depuis un fichier JSON:

        >>> config = Config(config_file='custom_config.json')
        >>> # Les valeurs du fichier surchargent les valeurs par défaut

        Chargement depuis un fichier YAML:

        >>> config = Config(config_file='custom_config.yaml')
        >>> # Nécessite PyYAML: pip install pyyaml

        Accès aux paramètres de configuration:

        >>> config = Config()
        >>> data_loader = DataLoader(config.LOGS_FILE_PATH, config.NOTES_FILE_PATH)
        >>> processor = DataProcessor(config)
        >>> print(f"Seuil de risque: {config.RISK_THRESHOLD_HIGH}")

        Export de la configuration actuelle:

        >>> config = Config()
        >>> config.PLOT_DPI = 600  # Modifier un paramètre
        >>> config.export_defaults('my_config.json')
        >>> # Sauvegarder la configuration personnalisée
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
    TEST_SPLIT_RATIO = 0.2  # 80% training, 20% testing
    CV_FOLDS = 5  # Number of folds for cross-validation
    RANDOM_STATE = 42  # Random seed for reproducibility

    NORMALITY_ALPHA = 0.05  # Significance level for normality tests (Shapiro-Wilk)

    # Random Forest hyperparameters
    RF_N_ESTIMATORS = 100  # Number of trees in the forest
    RF_MAX_DEPTH = None  # Maximum depth of the tree (None = unlimited)
    RF_MIN_SAMPLES_SPLIT = 2  # Minimum number of samples required to split an internal node
    RF_MIN_SAMPLES_LEAF = 1  # Minimum number of samples required to be at a leaf node
    RF_MAX_FEATURES = 'sqrt'  # Number of features to consider when looking for the best split
    RF_BOOTSTRAP = True  # Whether bootstrap samples are used when building trees
    RF_CRITERION = 'gini'  # Function to measure the quality of a split ('gini' or 'entropy')

    # Gradient Boosting hyperparameters
    GB_N_ESTIMATORS = 100  # Number of boosting stages to perform
    GB_LEARNING_RATE = 0.1  # Learning rate shrinks the contribution of each tree
    GB_MAX_DEPTH = 3  # Maximum depth of the individual regression estimators

    # Data processing parameters
    RAPID_EVENT_THRESHOLD_SECONDS = 5  # Threshold for rapid event deduplication (miss-clicks)
    OUTLIER_REMOVAL_ENABLED = True  # Whether to remove outliers via IQR before ML training
    NA_FILL_STRATEGY = 'zero'  # Strategy for filling NaN in activity features ('zero', 'mean', 'median')

    # Feature name translations (English -> French)
    FEATURE_NAMES_FR = {
        # Activity metrics
        'total_actions': 'actions_totales',
        'unique_days_active': 'jours_actifs_uniques',
        'actions_per_day': 'actions_par_jour',
        'session_count': 'nombre_sessions',
        # Event types
        'view_count': 'nb_consultations',
        'submission_count': 'nb_soumissions',
        'forum_count': 'nb_forums',
        'quiz_count': 'nb_quiz',
        'download_count': 'nb_telechargements',
        'other_count': 'nb_autres',
        # Consistency
        'streak_days': 'jours_consecutifs_max',
        'avg_gap_days': 'ecart_moyen_jours',
        'std_gap_days': 'ecart_type_jours',
        'study_frequency': 'frequence_etude',
        # Interaction depth
        'component_diversity': 'diversite_composants',
        'context_diversity': 'diversite_contextes',
        'avg_interactions_per_component': 'interactions_moy_par_composant',
        'component_switch_rate': 'taux_changement_composant',
        # Temporal patterns
        'peak_hour': 'heure_pointe',
        'morning_activity': 'activite_matin',
        'afternoon_activity': 'activite_apres_midi',
        'evening_activity': 'activite_soir',
        'night_activity': 'activite_nuit',
        'weekend_activity_ratio': 'ratio_activite_weekend',
        # Component prefix
        'comp_': 'comp_',
    }

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
        - TEST_SPLIT_RATIO doit être strictement entre 0 et 1
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
        if not isinstance(self.TEST_SPLIT_RATIO, (int, float)):
            raise TypeError(
                f"TEST_SPLIT_RATIO doit être un nombre (int ou float), "
                f"reçu: {type(self.TEST_SPLIT_RATIO).__name__}"
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

        # Validate TEST_SPLIT_RATIO
        if not (0 < self.TEST_SPLIT_RATIO < 1):
            raise ValueError(
                f"TEST_SPLIT_RATIO doit être strictement entre 0 et 1, "
                f"reçu: {self.TEST_SPLIT_RATIO}"
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

        # Type checking for data processing parameters
        if not isinstance(self.RAPID_EVENT_THRESHOLD_SECONDS, (int, float)):
            raise TypeError(
                f"RAPID_EVENT_THRESHOLD_SECONDS doit être un nombre (int ou float), "
                f"reçu: {type(self.RAPID_EVENT_THRESHOLD_SECONDS).__name__}"
            )

        if not isinstance(self.OUTLIER_REMOVAL_ENABLED, bool):
            raise TypeError(
                f"OUTLIER_REMOVAL_ENABLED doit être un booléen (bool), "
                f"reçu: {type(self.OUTLIER_REMOVAL_ENABLED).__name__}"
            )

        if not isinstance(self.NA_FILL_STRATEGY, str):
            raise TypeError(
                f"NA_FILL_STRATEGY doit être une chaîne (str), "
                f"reçu: {type(self.NA_FILL_STRATEGY).__name__}"
            )

        if not isinstance(self.FEATURE_NAMES_FR, dict):
            raise TypeError(
                f"FEATURE_NAMES_FR doit être un dictionnaire (dict), "
                f"reçu: {type(self.FEATURE_NAMES_FR).__name__}"
            )

        # Validate RAPID_EVENT_THRESHOLD_SECONDS
        if self.RAPID_EVENT_THRESHOLD_SECONDS < 0:
            raise ValueError(
                f"RAPID_EVENT_THRESHOLD_SECONDS doit être positif ou nul, "
                f"reçu: {self.RAPID_EVENT_THRESHOLD_SECONDS}"
            )

        # Validate NA_FILL_STRATEGY
        valid_na_strategies = ['zero', 'mean', 'median']
        if self.NA_FILL_STRATEGY not in valid_na_strategies:
            raise ValueError(
                f"NA_FILL_STRATEGY doit être l'un de {valid_na_strategies}, "
                f"reçu: {self.NA_FILL_STRATEGY}"
            )

        # Type checking for Random Forest hyperparameters
        if not isinstance(self.RF_N_ESTIMATORS, int):
            raise TypeError(
                f"RF_N_ESTIMATORS doit être un entier (int), "
                f"reçu: {type(self.RF_N_ESTIMATORS).__name__}"
            )

        if self.RF_MAX_DEPTH is not None and not isinstance(self.RF_MAX_DEPTH, int):
            raise TypeError(
                f"RF_MAX_DEPTH doit être un entier (int) ou None, "
                f"reçu: {type(self.RF_MAX_DEPTH).__name__}"
            )

        if not isinstance(self.RF_MIN_SAMPLES_SPLIT, int):
            raise TypeError(
                f"RF_MIN_SAMPLES_SPLIT doit être un entier (int), "
                f"reçu: {type(self.RF_MIN_SAMPLES_SPLIT).__name__}"
            )

        if not isinstance(self.RF_MIN_SAMPLES_LEAF, int):
            raise TypeError(
                f"RF_MIN_SAMPLES_LEAF doit être un entier (int), "
                f"reçu: {type(self.RF_MIN_SAMPLES_LEAF).__name__}"
            )

        if not isinstance(self.RF_MAX_FEATURES, (str, int, float)):
            raise TypeError(
                f"RF_MAX_FEATURES doit être une chaîne (str), un entier (int) ou un flottant (float), "
                f"reçu: {type(self.RF_MAX_FEATURES).__name__}"
            )

        if not isinstance(self.RF_BOOTSTRAP, bool):
            raise TypeError(
                f"RF_BOOTSTRAP doit être un booléen (bool), "
                f"reçu: {type(self.RF_BOOTSTRAP).__name__}"
            )

        if not isinstance(self.RF_CRITERION, str):
            raise TypeError(
                f"RF_CRITERION doit être une chaîne (str), "
                f"reçu: {type(self.RF_CRITERION).__name__}"
            )

        # Validate Random Forest hyperparameters
        if self.RF_N_ESTIMATORS <= 0:
            raise ValueError(
                f"RF_N_ESTIMATORS doit être strictement positif, "
                f"reçu: {self.RF_N_ESTIMATORS}"
            )

        if self.RF_MAX_DEPTH is not None and self.RF_MAX_DEPTH <= 0:
            raise ValueError(
                f"RF_MAX_DEPTH doit être strictement positif ou None, "
                f"reçu: {self.RF_MAX_DEPTH}"
            )

        if self.RF_MIN_SAMPLES_SPLIT < 2:
            raise ValueError(
                f"RF_MIN_SAMPLES_SPLIT doit être au moins 2, "
                f"reçu: {self.RF_MIN_SAMPLES_SPLIT}"
            )

        if self.RF_MIN_SAMPLES_LEAF < 1:
            raise ValueError(
                f"RF_MIN_SAMPLES_LEAF doit être au moins 1, "
                f"reçu: {self.RF_MIN_SAMPLES_LEAF}"
            )

        if isinstance(self.RF_MAX_FEATURES, str):
            valid_max_features = ['sqrt', 'log2', 'auto']
            if self.RF_MAX_FEATURES not in valid_max_features:
                raise ValueError(
                    f"RF_MAX_FEATURES doit être l'un de {valid_max_features} ou un nombre, "
                    f"reçu: {self.RF_MAX_FEATURES}"
                )
        elif isinstance(self.RF_MAX_FEATURES, (int, float)):
            if self.RF_MAX_FEATURES <= 0:
                raise ValueError(
                    f"RF_MAX_FEATURES doit être strictement positif, "
                    f"reçu: {self.RF_MAX_FEATURES}"
                )

        valid_criteria = ['gini', 'entropy']
        if self.RF_CRITERION not in valid_criteria:
            raise ValueError(
                f"RF_CRITERION doit être l'un de {valid_criteria}, "
                f"reçu: {self.RF_CRITERION}"
            )

        # Gradient Boosting hyperparameters validation
        if not isinstance(self.GB_N_ESTIMATORS, int):
            raise TypeError(
                f"GB_N_ESTIMATORS doit être un entier (int), "
                f"reçu: {type(self.GB_N_ESTIMATORS).__name__}"
            )

        if self.GB_N_ESTIMATORS <= 0:
            raise ValueError(
                f"GB_N_ESTIMATORS doit être strictement positif, "
                f"reçu: {self.GB_N_ESTIMATORS}"
            )

        if not isinstance(self.GB_LEARNING_RATE, (int, float)):
            raise TypeError(
                f"GB_LEARNING_RATE doit être un nombre (int ou float), "
                f"reçu: {type(self.GB_LEARNING_RATE).__name__}"
            )

        if self.GB_LEARNING_RATE <= 0 or self.GB_LEARNING_RATE > 1:
            raise ValueError(
                f"GB_LEARNING_RATE doit être entre 0 et 1 (exclusif de 0), "
                f"reçu: {self.GB_LEARNING_RATE}"
            )

        if not isinstance(self.GB_MAX_DEPTH, int):
            raise TypeError(
                f"GB_MAX_DEPTH doit être un entier (int), "
                f"reçu: {type(self.GB_MAX_DEPTH).__name__}"
            )

        if self.GB_MAX_DEPTH <= 0:
            raise ValueError(
                f"GB_MAX_DEPTH doit être strictement positif, "
                f"reçu: {self.GB_MAX_DEPTH}"
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
