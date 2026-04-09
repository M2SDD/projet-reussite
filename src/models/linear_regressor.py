#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 06/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module de régression linéaire multiple pour prédire les notes des étudiants

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
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.linear_model import LinearRegression

from .base_regressor import BaseRegressor


# ----------------------------------------------------------------------------------------------------------------------
# Classe LinearRegressor
# ----------------------------------------------------------------------------------------------------------------------
class LinearRegressor(BaseRegressor):
    """
    Classe responsable de l'entraînement et de l'évaluation d'un modèle de régression linéaire multiple.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialise le modèle en appelant le parent."""
        super().__init__(config)

    def _fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Entraîne le modèle de régression linéaire sur les données fournies."""
        self.model = LinearRegression()
        self.model.fit(X, y)

    def get_coefficients(self) -> Dict[str, Any]:
        """Extrait les coefficients et l'intercept."""
        self._check_is_fitted()
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }
