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

    def train_test_split(self, df, target_column, test_size=0.2, random_state=None):
        """
        Divise les données en ensembles d'entraînement et de test.

        Sépare les features (X) de la variable cible (y) et effectue
        un split train/test avec le ratio configuré.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features et la cible.
            target_column (str): Le nom de la colonne cible à prédire.
            test_size (float, optional): Proportion de l'ensemble de test (défaut: 0.2).
            random_state (int, optional): Seed pour la reproductibilité (défaut: None).

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
                - X_train (pd.DataFrame): Features d'entraînement
                - X_test (pd.DataFrame): Features de test
                - y_train (pd.Series): Cible d'entraînement
                - y_test (pd.Series): Cible de test

        Raises:
            ValueError: Si la colonne cible n'existe pas dans le DataFrame.
        """
        if target_column not in df.columns:
            raise ValueError(
                f"La colonne cible '{target_column}' n'existe pas dans le DataFrame. "
                f"Colonnes disponibles: {list(df.columns)}"
            )

        # Séparer les features (X) de la cible (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Effectuer le split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test
