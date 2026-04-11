#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 09/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module définissant la classe de base abstraite pour les modèles de prédiction.
NB. la classe contient le scaler pour la standardisation des données.

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
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, Union
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import NotFittedError

from ..config import Config


class BaseRegressor(ABC):
    """
    Classe de base abstraite pour les modèles de prédiction

    Fournit les méthodes communes de préparation des données, de prédiction
    et d'évaluation des performances.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config if config is not None else Config()
        self.model = None
        self.scaler = RobustScaler()  # beaucoup de disparités entre les actions et donc "d'outliers dans les données"

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'BaseRegressor':
        """
        Valide les données puis entraîne le modèle (Design Pattern : Template Method).
        """
        self._validate_input_data(X, y)
        X = self.scaler.fit_transform(X)  # Entrainement du scaler sur les données d'entraînement
        self._fit(X, y)
        return self

    @abstractmethod
    def _fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        Logique d'entraînement interne. Doit être implémentée par les enfants.
        """
        pass

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prédit les valeurs cibles pour les features fournies."""
        self._check_is_fitted()
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X ne peut pas être vide.")

        # Scaling des données avant prédiction
        X = self.scaler.transform(X)

        return self.model.predict(X)

    def _check_is_fitted(self) -> None:
        """Vérifie si le modèle a été entraîné."""
        if self.model is None:
            raise NotFittedError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant l'utilisation."
            )

    def _validate_input_data(self, X: Any, y: Any) -> None:
        """Vérifie la validité des entrées (pas vides, même taille)."""
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X ne peut pas être vide.")
        if y is None or (hasattr(y, '__len__') and len(y) == 0):
            raise ValueError("y ne peut pas être vide.")
        if len(X) != len(y):
            raise ValueError(f"X et y doivent avoir le même nombre d'échantillons. X: {len(X)}, y: {len(y)}.")

    def _validate_data(self, X: Any, y: Any) -> None:
        """Méthode utilitaire interne pour valider X et y avant calcul des métriques."""
        self._check_is_fitted()
        self._validate_input_data(X, y)

    def compute_r2_score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calcule le coefficient de détermination R²."""
        self._validate_data(X, y)
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def compute_rmse(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calcule l'erreur quadratique moyenne (RMSE)."""
        self._validate_data(X, y)
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))

    def compute_mae(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calcule l'erreur absolue moyenne (MAE)."""
        self._validate_data(X, y)
        y_pred = self.predict(X)
        return mean_absolute_error(y, y_pred)

    def compute_adjusted_r2(self, X, y):
        """Calcule le coefficient de détermination ajusté (R² ajusté)."""
        r2 = self.compute_r2_score(X, y)  # Calculer le R² score
        n = len(X)

        # Gérer les cas où X est un DataFrame ou un tableau numpy
        if hasattr(X, 'shape'):
            p = 1 if len(X.shape) == 1 else X.shape[1]
        else:
            # X est une structure de données sans attribut shape,
            # essayer de convertir en array pour obtenir le nombre de features
            X_array = np.array(X)
            p = 1 if len(X_array.shape) == 1 else X_array.shape[1]

        if n - p - 1 <= 0:
            warnings.warn("Le nombre d'échantillons est insuffisant pour calculer le R² ajusté. Retourne NaN.")
            return float('nan')

        return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Évalue le modèle et retourne les métriques de base."""
        return {
            'r2': self.compute_r2_score(X, y),
            'rmse': self.compute_rmse(X, y),
            'mae': self.compute_mae(X, y),
            'adjusted_r2': self.compute_adjusted_r2(X, y)
        }

    def compute_residuals(self, X, y):
        """
        Calcule les résidus du modèle de régression.
        """
        self._validate_data(X, y)

        # Faire les prédictions
        y_pred = self.predict(X)

        # Calculer les résidus
        residuals = y - y_pred

        # Convertir en numpy array si nécessaire
        if hasattr(residuals, 'values'):
            # residuals est une Series pandas
            residuals = residuals.values

        return residuals

    def check_residuals_normality(self, X, y):
        """
        Teste la normalité des résidus du modèle de régression à l'aide du test de Shapiro-Wilk.

        Le test de Shapiro-Wilk évalue si les résidus suivent une distribution normale,
        ce qui est une hypothèse importante de la régression linéaire. Un p-value > 0.05
        suggère que les résidus suivent une distribution normale.
        """
        self._validate_data(X, y)

        # Calculer les résidus
        residuals = self.compute_residuals(X, y)

        # Effectuer le test de Shapiro-Wilk
        test_statistic, p_value = stats.shapiro(residuals)

        # Retourner les résultats sous forme de dictionnaire
        return {
            'test_statistic': float(test_statistic),
            'p_value': float(p_value)
        }
