#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 08/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module de modèle ML générique pour la prédiction des notes des étudiants

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
import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .base_regressor import BaseRegressor


# ----------------------------------------------------------------------------------------------------------------------
# Classe EnsembleRegressor
# ----------------------------------------------------------------------------------------------------------------------
class EnsembleRegressor(BaseRegressor):
    """
    Classe responsable de l'entraînement et de l'évaluation d'un modèle de machine learning.
    Hérite des fonctionnalités communes de BaseModel.
    """

    def __init__(self, config: Optional[Any] = None, model_type: str = 'random_forest'):
        """Initialise le modèle de machine learning en appelant le parent."""
        super().__init__(config)

        valid_types = ['random_forest', 'gradient_boosting']
        if model_type not in valid_types:
            raise ValueError(f"model_type doit être l'un de {valid_types}, reçu: {model_type}")

        self.model_type = model_type

    def _fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Entraîne le modèle Random Forest ou Gradient Boosting."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=getattr(self.config, 'RF_N_ESTIMATORS', 200),
                max_depth=getattr(self.config, 'RF_MAX_DEPTH', 10),
                min_samples_split=getattr(self.config, 'RF_MIN_SAMPLES_SPLIT', 5),
                min_samples_leaf=getattr(self.config, 'RF_MIN_SAMPLES_LEAF', 3),
                max_features=getattr(self.config, 'RF_MAX_FEATURES', 'sqrt'),
                bootstrap=getattr(self.config, 'RF_BOOTSTRAP', False),
                criterion=getattr(self.config, 'RF_CRITERION', 'friedman_mse'),
                random_state=getattr(self.config, 'RANDOM_STATE', 42),
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=getattr(self.config, 'GB_N_ESTIMATORS', 100),
                learning_rate=getattr(self.config, 'GB_LEARNING_RATE', 0.1),
                max_depth=getattr(self.config, 'GB_MAX_DEPTH', 3),
                min_samples_split=getattr(self.config, 'GB_MIN_SAMPLES_SPLIT', 10),
                subsample=getattr(self.config, 'GB_SUBSAMPLE', 0.8),
                criterion=getattr(self.config, 'GB_CRITERION', 'friedman_mse'),
                random_state=getattr(self.config, 'RANDOM_STATE', 42)
            )

        self.model.fit(X, y)

    def get_feature_importance(self, feature_names: Union[pd.Index, list]) -> pd.DataFrame:
        """Extrait l'importance des features du modèle de machine learning entraîné."""
        self._check_is_fitted()

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Le modèle ne supporte pas l'extraction d'importance des features.")
        if feature_names is None or len(feature_names) == 0:
            raise ValueError("feature_names ne peut pas être vide.")

        importance_values = self.model.feature_importances_
        if len(feature_names) != len(importance_values):
            raise ValueError("La taille de feature_names ne correspond pas au nombre de features.")

        feature_importance = pd.DataFrame({
            'feature': list(feature_names),
            'importance': importance_values
        })

        return feature_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)
