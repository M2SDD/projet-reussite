#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Contrôleur principal de l'application.

Objet singleton partagé par tous les onglets.  Il encapsule l'état
de l'application (données chargées, dataset construit, modèles entraînés)
et expose les méthodes métier qui appellent les classes de src/.
Les onglets ne font jamais d'appels directs au code métier : tout passe
par ce contrôleur.

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
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import Config
from ..data.data_loader import DataLoader
from ..data.dataset_builder import DatasetBuilder
from ..evaluation.model_evaluator import ModelEvaluator
from ..models.base_regressor import BaseRegressor
from ..models.ensemble_regressor import EnsembleRegressor
from ..models.linear_regressor import LinearRegressor

# Chemin racine du projet (trois niveaux au-dessus de ce fichier)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------------------------------------------------------
# Classe AppController
# ----------------------------------------------------------------------------------------------------------------------
class AppController:
    """
    Contrôleur central de l'application.

    Gère l'état global (données, dataset, modèles) et orchestre les appels
    aux classes métier.  La StatusBar est injectée par MainWindow après
    instanciation via l'attribut `status_bar`.
    """

    DEMO_LOGS  = os.path.join(_ROOT, 'data', 'logs_info_25_pseudo.csv')
    DEMO_NOTES = os.path.join(_ROOT, 'data', 'notes_info_25_pseudo.csv')

    def __init__(self):
        self.config = Config()
        self._loader  = DataLoader(self.config)
        self._builder = DatasetBuilder(self.config)

        # --- Etat : données brutes ---
        self.logs_path:  Optional[str] = None
        self.notes_path: Optional[str] = None
        self.df_logs:    Optional[pd.DataFrame] = None
        self.df_notes:   Optional[pd.DataFrame] = None

        # --- Etat : dataset ML ---
        self.X:       Optional[pd.DataFrame] = None
        self.y:       Optional[pd.Series]    = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test:  Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series]    = None
        self.y_test:  Optional[pd.Series]    = None
        self.selected_features: List[str] = []

        # --- Etat : modèles ---
        self.trained_models: Dict[str, BaseRegressor] = {}
        self.evaluator = ModelEvaluator(self.config)

        # Injecté par MainWindow
        self.status_bar = None

    # ------------------------------------------------------------------
    # Interface avec la StatusBar
    # ------------------------------------------------------------------

    def set_status(self, message: str) -> None:
        if self.status_bar:
            self.status_bar.set(message)

    def set_loading(self, loading: bool) -> None:
        if self.status_bar:
            self.status_bar.set_loading(loading)

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------

    def load_data(self, logs_path: str, notes_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Charge les fichiers CSV et les stocke dans l'état."""
        self.logs_path  = logs_path
        self.notes_path = notes_path
        self.df_logs  = self._loader.load_logs(logs_path)
        self.df_notes = self._loader.load_notes(notes_path)
        return self.df_logs, self.df_notes

    def load_demo_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Raccourci pour charger les fichiers de démonstration du dossier data/."""
        return self.load_data(self.DEMO_LOGS, self.DEMO_NOTES)

    # ------------------------------------------------------------------
    # Construction du dataset
    # ------------------------------------------------------------------

    def build_dataset(self, options: dict) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Construit X, y et la liste des features via DatasetBuilder.

        Parameters
        ----------
        options : dict
            drop_inactive_students : bool
            remove_outliers        : bool
            selection_methods      : list[str] | None
            k_features             : int
            prefilter_variance     : bool
            prefilter_correlation  : bool
        """
        self.X, self.y, self.selected_features = self._builder.build_dataset(
            logs_path=self.logs_path,
            notes_path=self.notes_path,
            drop_inactive_students=options.get('drop_inactive_students', True),
            remove_outliers=options.get('remove_outliers', False),
            selection_methods=options.get('selection_methods', 'linear'),
            k_features=options.get('k_features', 15),
            prefilter_variance=options.get('prefilter_variance', True),
            prefilter_correlation=options.get('prefilter_correlation', True),
        )
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self._builder.get_train_test_split(self.X, self.y)

        return self.X, self.y, self.selected_features

    # ------------------------------------------------------------------
    # Modèles
    # ------------------------------------------------------------------

    def train_model(self, name: str, model_type: str, hyperparams: dict) -> BaseRegressor:
        """
        Entraîne un modèle et l'enregistre dans l'état.

        Parameters
        ----------
        name        : str   Nom unique du modèle.
        model_type  : str   'linear' | 'random_forest' | 'gradient_boosting'
        hyperparams : dict  Clés correspondant aux attributs de Config (ex. RF_N_ESTIMATORS).
        """
        if name in self.trained_models:
            raise ValueError(f"Un modèle nommé '{name}' existe déjà.")

        # Appliquer les hyperparamètres sur la config partagée
        for key, value in hyperparams.items():
            setattr(self.config, key, value)

        if model_type == 'linear':
            model = LinearRegressor(self.config)
        elif model_type in ('random_forest', 'gradient_boosting'):
            model = EnsembleRegressor(self.config, model_type=model_type)
        else:
            raise ValueError(f"Type de modèle inconnu : '{model_type}'")

        model.fit(self.X_train, self.y_train)

        self.trained_models[name] = model
        self.evaluator.add_model(name, model, self.X_test, self.y_test)

        return model

    def delete_model(self, name: str) -> None:
        """Supprime un modèle de l'état et de l'évaluateur."""
        self.trained_models.pop(name, None)
        self.evaluator.models.pop(name, None)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, include_adjusted_r2: bool = False) -> pd.DataFrame:
        return self.evaluator.get_comparison_table(include_adjusted_r2=include_adjusted_r2)

    def get_recommendation(self, use_adjusted_r2: bool = False) -> dict:
        return self.evaluator.get_recommendation(use_adjusted_r2=use_adjusted_r2)

    def get_plot_predictions(self):
        return self.evaluator.plot_predictions()

    def get_plot_residuals(self):
        return self.evaluator.plot_residuals()

    def get_plot_metrics(self, include_adjusted_r2: bool = False):
        return self.evaluator.plot_metrics_comparison(include_adjusted_r2=include_adjusted_r2)

    def export_results(self, output_dir: str, include_adjusted_r2: bool = False) -> None:
        self.evaluator.export_results(output_dir, include_adjusted_r2=include_adjusted_r2)

    # ------------------------------------------------------------------
    # Etat pipeline (utilisés pour l'activation des onglets)
    # ------------------------------------------------------------------

    def has_data(self) -> bool:
        return self.df_logs is not None and self.df_notes is not None

    def has_dataset(self) -> bool:
        return self.X is not None and self.y is not None

    def has_models(self) -> bool:
        return bool(self.trained_models)
