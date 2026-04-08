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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .config import Config


# ----------------------------------------------------------------------------------------------------------------------
# Classe MLModel
# ----------------------------------------------------------------------------------------------------------------------
class MLModel:
    """
    Classe responsable de l'entraînement et de l'évaluation d'un modèle de machine learning.

    Cette classe fournit une interface générique pour entraîner différents types de modèles
    de machine learning sur les indicateurs d'engagement des étudiants afin de prédire
    leurs notes finales. Elle inclut des méthodes pour l'entraînement, la prédiction,
    et l'évaluation du modèle.
    """

    def __init__(self, config=None):
        """
        Initialise le modèle de machine learning.

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

    def fit(self, X, y):
        """
        Entraîne le modèle de machine learning sur les données fournies.

        Crée une instance de LinearRegression et l'entraîne avec les features (X)
        et la variable cible (y). Le modèle entraîné est stocké dans self.model.

        Args:
            X (pd.DataFrame ou np.ndarray): Les features d'entraînement.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
            y (pd.Series ou np.ndarray): La variable cible à prédire.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            self: Retourne l'instance pour permettre le chaînage de méthodes.

        Raises:
            ValueError: Si X ou y sont vides, ou si leurs dimensions sont incompatibles.
        """
        # Validation des entrées
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X ne peut pas être vide.")

        if y is None or (hasattr(y, '__len__') and len(y) == 0):
            raise ValueError("y ne peut pas être vide.")

        # Vérification de la compatibilité des dimensions
        n_samples_X = len(X)
        n_samples_y = len(y)

        if n_samples_X != n_samples_y:
            raise ValueError(
                f"X et y doivent avoir le même nombre d'échantillons. "
                f"X a {n_samples_X} échantillons, y en a {n_samples_y}."
            )

        # Créer et entraîner le modèle
        self.model = LinearRegression()
        self.model.fit(X, y)

        return self

    def predict(self, X):
        """
        Prédit les valeurs cibles pour les features fournies.

        Utilise le modèle de machine learning entraîné pour faire des prédictions
        sur de nouvelles données.

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles faire des prédictions.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.

        Returns:
            np.ndarray: Les valeurs prédites de shape (n_samples,).

        Raises:
            ValueError: Si le modèle n'a pas été entraîné ou si X est vide.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant de faire des prédictions."
            )

        # Validation des entrées
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X ne peut pas être vide.")

        # Faire les prédictions
        predictions = self.model.predict(X)

        return predictions
