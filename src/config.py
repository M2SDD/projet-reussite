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
from typing import Optional

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

    Examples:
        Utilisation avec la configuration par défaut:

        >>> from src import Config
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
        >>> print(f"Seuil de risque: {config.RISK_THRESHOLD_HIGH}")

        Export de la configuration actuelle:

        >>> config = Config()
        >>> config.PLOT_DPI = 600  # Modifier un paramètre
        >>> config.save_to_file('my_config.json')
        >>> # Sauvegarder la configuration personnalisée
    """
    # --- CHARGEMENT DES DONNEES ---

    # Chemin des fichiers de données et répertoire de sortie
    LOGS_FILE_PATH = 'data/logs.csv'
    NOTES_FILE_PATH = 'data/notes.csv'
    OUTPUT_DIR = 'output'

    # Schéma des données (Noms des colonnes attendues)
    LOGS_REQUIRED_COLUMNS = ['heure', 'pseudo', 'contexte', 'composant', 'evenement']
    NOTES_REQUIRED_COLUMNS = ['pseudo', 'note']

    # Format de date et d'heure pour l'analyse des horodatages des logs
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


    # --- TRAITEMENT DES DONNEES ---

    # Clés de jointure et cible
    MERGE_KEY = 'pseudo'
    TARGET_COLUMN = 'note'

    # Plage de notes attendue
    NOTE_MIN = 0
    NOTE_MAX = 20

    # Seuils d'évaluation des risques
    RISK_THRESHOLD_HIGH = 10  # Une note inférieure à celle-ci indique un risque élevé
    RISK_THRESHOLD_MEDIUM = 12  # Une note inférieure à celle-ci indique un risque moyen
    # NB. moyenne des notes à 10.00 et médiane à 9.86

    # Duplicate removal settings
    DUPLICATE_KEEP = 'first'
    DUPLICATE_SUBSET = None  # None means all columns

    # Paramètres de data processing
    RAPID_EVENT_THRESHOLD_ENABLED = False
    RAPID_EVENT_THRESHOLD_SECONDS = 2  # Seuil pour le dédoublement des événements rapides (miss-clicks ?)

    # --- FEATURE ENGINEERING ---

    # Activation des composants pour le feature engineering
    FEATURE_HOUR_OF_DAY = True
    FEATURE_DAY_OF_WEEK = True
    FEATURE_SESSION_COUNT = True
    FEATURE_TOTAL_EVENTS = True

    # Paramètres des features d'engagement
    SESSION_GAP_MINUTES = 30  # Intervalle de temps nécessaire pour définir une nouvelle session


    # --- FEATURE SELECTION ---

    # Traitement des outliers et valeurs manquantes
    OUTLIER_REMOVAL_ENABLED = False  # Activer la suppression des outliers à l'aide de l'IQR
    NA_FILL_STRATEGY = 'zero'  # Strategy for filling NaN in activity features ('zero', 'mean', 'median')
    # TODO : ajouter 'mode' ?


    # --- PARAMÈTRES ML/STATS ---

    # Paramètres d'apprentissage automatique et d'analyse statistique
    TEST_SPLIT_RATIO = 0.2  # 80% train, 20% test
    CV_FOLDS = 5  # Nombre de folds pour la cross-val
    RANDOM_STATE = 42  # Graine pour la reproductibilité

    NORMALITY_ALPHA = 0.05  # Seuil de pertinence du test de normalité (Shapiro-Wilk)

    # Hyperparamètres : Random Forest
    RF_N_ESTIMATORS = 200  # Nombre d'arbres
    RF_MAX_DEPTH = 10  # Profondeur maximale des arbres (None = illimitée)
    RF_MIN_SAMPLES_SPLIT = 5  # Nombre minimal d'échantillons requis pour diviser un noeud
    RF_MIN_SAMPLES_LEAF = 3  # Nombre minimum d'échantillons requis pour une feuille
    RF_MAX_FEATURES = 'sqrt'  # Nombre maximum de variables pour trouver la meilleure séparation à chaque noeud (sqrt(n_features))
    RF_BOOTSTRAP = False  # Détermine si des échantillons de données sont tirés (True) ou sans remplacement (False -> petits jeux de données)
    RF_CRITERION = 'friedman_mse'  # Fonction de mesure de la qualité d'une division

    # Hyperparamètres : Gradient Boosting
    GB_N_ESTIMATORS = 100  # Nombre d'arbres
    GB_MAX_DEPTH = 3  # Profondeur maximale des arbres (None = illimitée)
    GB_LEARNING_RATE = 0.1  # Learning rate pour réduire l'impact de chaque arbre
    GB_MIN_SAMPLES_SPLIT = 10 # Nombre minimal d'échantillons requis pour diviser un noeud
    GB_SUBSAMPLE = 0.8 # Sous-échantillonnage pour chaque arbre (stochastic gradient boosting)
    GB_CRITERION = 'friedman_mse' # Fonction de mesure de la qualité d'une division


    FEATURE_NAMES_FR = {
        # Métriques d'activité
        'total_actions': 'actions_totales',
        'unique_days_active': 'jours_actifs_uniques',
        'actions_per_day': 'actions_par_jour',
        'session_count': 'nombre_sessions',
        # Type d'évenements
        'view_count': 'nb_consultations',
        'submission_count': 'nb_soumissions',
        'forum_count': 'nb_forums',
        'quiz_count': 'nb_quiz',
        'download_count': 'nb_telechargements',
        'other_count': 'nb_autres',
        # Cohérence
        'streak_days': 'jours_consecutifs_max',
        'avg_gap_days': 'ecart_moyen_jours',
        'std_gap_days': 'ecart_type_jours',
        'study_frequency': 'frequence_etude',
        # Niveau d'interaction
        'component_diversity': 'diversite_composants',
        'context_diversity': 'diversite_contextes',
        'avg_interactions_per_component': 'interactions_moy_par_composant',
        'component_switch_rate': 'taux_changement_composant',
        # Tendances temporelles
        'peak_hour': 'heure_pointe',
        'morning_activity': 'activite_matin',
        'afternoon_activity': 'activite_apres_midi',
        'evening_activity': 'activite_soir',
        'night_activity': 'activite_nuit',
        'weekend_activity_ratio': 'ratio_activite_weekend',
        # Préfixe de composant
        'comp_': 'comp_',
    }


    # --- VISUALISATION ---

    # Paramètres de visualisation
    PLOT_DPI = 300  # Résolution des tracés enregistrés
    PLOT_FIGSIZE = (10, 6)  # Dimensions par défaut des figures (largeur, hauteur) en pouces
    PLOT_STYLE = 'seaborn-v0_8'  # Style Matplotlib
    PLOT_COLOR_PALETTE = 'Set2'  # Palette de couleurs par défaut
    PLOT_SAVE_FORMAT = 'png'  # Format par défaut pour les graphiques enregistrés
    PLOT_FONT_SIZE = 12  # Taille de police par défaut pour les graphiques

    def __init__(self, config_file: Optional[str] = None) -> None:
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
                        # JSON par défaut pour assurer la rétrocompatibilité
                        config_data = json.load(f)

                # Remplacer les attributs de classe par les valeurs du fichier de config
                for key, value in config_data.items():
                    setattr(self, key, value)

                # Post-traitement : convertir les listes en tuples si nécessaire
                # Les tableaux JSON sont chargés sous forme de listes, mais certaines valeurs de configuration doivent être des tuples
                if hasattr(self, 'PLOT_FIGSIZE') and isinstance(self.PLOT_FIGSIZE, list):
                    self.PLOT_FIGSIZE = tuple(self.PLOT_FIGSIZE)
            except FileNotFoundError:
                # Revenir aux valeurs par défaut si le fichier de configuration est manquant
                pass

        # Validation des paramètres de configuration après chargement
        self._validate()

    def _validate(self) -> None:
        """Valide les paramètres de configuration."""
        # Vérification du type des paramètres numériques
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

        # Vérification du type des paramètres str
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

        # Vérification du type pour les paramètres booléens
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

        # Vérification du type pour les collections
        if not isinstance(self.PLOT_FIGSIZE, tuple):
            raise TypeError(
                f"PLOT_FIGSIZE doit être un tuple, "
                f"reçu: {type(self.PLOT_FIGSIZE).__name__}"
            )

        # Validation du TEST_SPLIT_RATIO
        if not (0 < self.TEST_SPLIT_RATIO < 1):
            raise ValueError(
                f"TEST_SPLIT_RATIO doit être strictement entre 0 et 1, "
                f"reçu: {self.TEST_SPLIT_RATIO}"
            )

        # Validation du CV_FOLDS
        if self.CV_FOLDS < 2:
            raise ValueError(
                f"CV_FOLDS doit être au moins 2, reçu: {self.CV_FOLDS}"
            )

        # Validation des NOTE_MIN and NOTE_MAX
        if self.NOTE_MIN >= self.NOTE_MAX:
            raise ValueError(
                f"NOTE_MIN ({self.NOTE_MIN}) doit être inférieur à "
                f"NOTE_MAX ({self.NOTE_MAX})"
            )

        # Validation du PLOT_DPI
        if self.PLOT_DPI <= 0:
            raise ValueError(
                f"PLOT_DPI doit être strictement positif, reçu: {self.PLOT_DPI}"
            )

        # Validation des seuils de risque (dans les bornes indiquées)
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

        # Validation de l'ordre des seuils de risque
        if self.RISK_THRESHOLD_HIGH >= self.RISK_THRESHOLD_MEDIUM:
            raise ValueError(
                f"RISK_THRESHOLD_HIGH ({self.RISK_THRESHOLD_HIGH}) doit être "
                f"inférieur à RISK_THRESHOLD_MEDIUM ({self.RISK_THRESHOLD_MEDIUM})"
            )

        # Validation du SESSION_GAP_MINUTES
        if self.SESSION_GAP_MINUTES <= 0:
            raise ValueError(
                f"SESSION_GAP_MINUTES doit être strictement positif, "
                f"reçu: {self.SESSION_GAP_MINUTES}"
            )

        # Vérification des types pour les paramètres de traitement des données
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

        # Validation du RAPID_EVENT_THRESHOLD_SECONDS
        if self.RAPID_EVENT_THRESHOLD_SECONDS < 0:
            raise ValueError(
                f"RAPID_EVENT_THRESHOLD_SECONDS doit être positif ou nul, "
                f"reçu: {self.RAPID_EVENT_THRESHOLD_SECONDS}"
            )

        # Validation du NA_FILL_STRATEGY
        valid_na_strategies = ['zero', 'mean', 'median']
        if self.NA_FILL_STRATEGY not in valid_na_strategies:
            raise ValueError(
                f"NA_FILL_STRATEGY doit être l'un de {valid_na_strategies}, "
                f"reçu: {self.NA_FILL_STRATEGY}"
            )

        # Vérification des types pour les hyperparamètres du Random Forest
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

        # Validation des hyperparamètres du Random Forest
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

        valid_criteria = ['friedman_mse', 'poisson', 'absolute_error', 'squared_error']
        if self.RF_CRITERION not in valid_criteria:
            raise ValueError(
                f"RF_CRITERION doit être l'un de {valid_criteria}, "
                f"reçu: {self.RF_CRITERION}"
            )

        # Validation des hyperparamètres du Gradient Boosting
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

        if not isinstance(self.GB_MIN_SAMPLES_SPLIT, int):
            raise TypeError(
                f"GB_MIN_SAMPLES_SPLIT doit être un entier (int), "
                f"reçu: {type(self.GB_MIN_SAMPLES_SPLIT).__name__}"
            )

        if self.GB_MIN_SAMPLES_SPLIT < 2:
            raise ValueError(
                f"GB_MIN_SAMPLES_SPLIT doit être au moins 2, "
                f"reçu: {self.GB_MIN_SAMPLES_SPLIT}"
            )

        if not isinstance(self.GB_SUBSAMPLE, (int, float)):
            raise TypeError(
                f"RISK_THRESHOLD_MEDIUM doit être un nombre (int ou float), "
                f"reçu: {type(self.GB_SUBSAMPLE).__name__}"
            )

        if not (0 < self.GB_SUBSAMPLE < 1):
            raise ValueError(
                f"GB_SUBSAMPLE doit être strictement entre 0 et 1, "
                f"reçu: {self.GB_SUBSAMPLE}"
            )

        valid_criteria = ['friedman_mse', 'poisson', 'absolute_error', 'squared_error']
        if self.GB_CRITERION not in valid_criteria:
            raise ValueError(
                f"GB_CRITERION doit être l'un de {valid_criteria}, "
                f"reçu: {self.GB_CRITERION}"
            )

    def save_to_file(self, file_path: str) -> None:
        """
        Exporte la configuration actuelle vers un fichier JSON ou YAML.

        Cette méthode permet de sauvegarder tous les paramètres de configuration
        actuels dans un fichier, qui peut ensuite être rechargé via le paramètre
        config_file du constructeur.

        Example:
            >>> config = Config()
            >>> config.save_to_file('config.json')
            >>> # Le fichier config.json contient tous les paramètres de configuration
        """
        # Récupérer tous les attributs
        config_data = {}
        for attr_name in dir(self):
            # Ignorer les méthodes et attributs privés
            if attr_name.startswith('_'):
                continue
            attr_value = getattr(self, attr_name)
            if callable(attr_value):
                continue

            config_data[attr_name] = attr_value

        # Déterminer le format via l'extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # S'assurer que le dossier parent existe
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Écriture du fichier
        with open(file_path, 'w', encoding='utf-8') as f:
            if ext in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "PyYAML n'est pas installé. "
                        "Installez-le avec: pip install pyyaml"
                    )
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                # JSON par défaut
                json.dump(config_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    config = Config()
    # sauvegarde du fichier de config dans le dossier parent
    os.chdir(os.pardir)
    print(os.getcwd())

    config_file = 'config.json'
    config.save_to_file(config_file)
    print(f"Fichier sauvegardé dans le fichier {os.path.join(os.getcwd(), config_file)}")

    # test de chargement
    config_loaded = Config(config_file='config.default.json')
    print(f"Configuration chargée depuis {os.path.join(os.getcwd(), 'config.default.json')}")

    # vérification des configurations
    config_data = {}
    for attr_name in dir(config):
        # Ignorer les méthodes et attributs privés
        if attr_name.startswith('_'):
            continue
        attr_value = getattr(config, attr_name)
        if callable(attr_value):
            continue

        config_data[attr_name] = attr_value

    config_loaded_data = {}
    for attr_name in dir(config_loaded):
        if attr_name.startswith('_'):
            continue
        attr_value = getattr(config_loaded, attr_name)
        if callable(attr_value):
            continue
        config_loaded_data[attr_name] = attr_value

    same_config = True
    if config_data != config_loaded_data:
        same_config = False
        print("Les configurations sont différentes !")
    else:
        print("Les configurations sont identiques !")