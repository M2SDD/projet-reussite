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
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .config import Config


# ----------------------------------------------------------------------------------------------------------------------
# Classe RegressionModel
# ----------------------------------------------------------------------------------------------------------------------
class RegressionModel:
    """
    Classe responsable de l'entraînement et de l'évaluation d'un modèle de régression linéaire multiple.

    Cette classe fournit des méthodes pour entraîner un modèle de régression linéaire
    sur les indicateurs d'engagement des étudiants afin de prédire leurs notes finales.
    Elle inclut également des méthodes pour l'évaluation du modèle (R², RMSE, MAE, R² ajusté)
    et l'analyse des résidus.
    """

    def __init__(self, config=None):
        """
        Initialise le modèle de régression.

        Args:
            config (Config): Instance de configuration. Si None, utilise la configuration par défaut.
        """
        self.config = config if config is not None else Config()
        self.model = None
