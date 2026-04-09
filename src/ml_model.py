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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

    def __init__(self, config=None, model_type='random_forest'):
        """
        Initialise le modèle de machine learning.

        Supporte deux types de modèles:
        - 'random_forest': RandomForestRegressor (par défaut)
        - 'gradient_boosting': GradientBoostingRegressor

        Args:
            config (Config): Instance de configuration. Si None, utilise la configuration par défaut.
            model_type (str): Type de modèle ('random_forest' ou 'gradient_boosting'). Défaut: 'random_forest'.

        Raises:
            ValueError: Si model_type n'est pas un type supporté.
        """
        self.config = config if config is not None else Config()

        # Validate model_type
        valid_types = ['random_forest', 'gradient_boosting']
        if model_type not in valid_types:
            raise ValueError(
                f"model_type doit être l'un de {valid_types}, reçu: {model_type}"
            )

        self.model_type = model_type
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

        Crée une instance de RandomForestRegressor ou GradientBoostingRegressor
        selon le model_type spécifié, puis l'entraîne avec les features (X)
        et la variable cible (y). Le modèle entraîné est stocké dans self.model.

        Les hyperparamètres sont chargés depuis self.config:
        - Random Forest: RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, etc.
        - Gradient Boosting: GB_N_ESTIMATORS, GB_LEARNING_RATE, GB_MAX_DEPTH

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

        # Créer et entraîner le modèle selon le type avec hyperparamètres du config
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.config.RF_N_ESTIMATORS,
                max_depth=self.config.RF_MAX_DEPTH,
                min_samples_split=self.config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=self.config.RF_MIN_SAMPLES_LEAF,
                max_features=self.config.RF_MAX_FEATURES,
                bootstrap=self.config.RF_BOOTSTRAP,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1  # Use all CPU cores for training
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=getattr(self.config, 'GB_N_ESTIMATORS', 100),
                learning_rate=getattr(self.config, 'GB_LEARNING_RATE', 0.1),
                max_depth=getattr(self.config, 'GB_MAX_DEPTH', 3),
                random_state=self.config.RANDOM_STATE
            )
        else:
            raise ValueError(
                f"model_type invalide: {self.model_type}"
            )
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

    def get_feature_importance(self, feature_names):
        """
        Extrait l'importance des features du modèle de machine learning entraîné.

        Pour les modèles Random Forest et Gradient Boosting, cette méthode utilise
        l'attribut feature_importances_ qui mesure l'importance moyenne de chaque
        feature basée sur la réduction de l'impureté (Gini importance). Les valeurs
        sont déjà normalisées et somment à 1.0.

        Cette méthode est différente de celle de RegressionModel qui utilise les
        coefficients de régression linéaire. Les feature importances des modèles
        basés sur des arbres capturent mieux les relations non-linéaires et les
        interactions entre features.

        Args:
            feature_names (pd.Index ou list): Les noms des features.
                Peut être un Index pandas (ex: X.columns) ou une liste de noms.
                Le nombre de noms doit correspondre au nombre de features du modèle.

        Returns:
            pd.DataFrame: DataFrame contenant l'importance des features avec deux colonnes :
                - 'feature' (str): Le nom de la feature.
                - 'importance' (float): L'importance de la feature (Gini importance).
                Les lignes sont triées par ordre décroissant d'importance.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné ou si le nombre de noms
                       ne correspond pas au nombre de features du modèle.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant d'extraire l'importance des features."
            )

        # Validation des entrées
        if feature_names is None or len(feature_names) == 0:
            raise ValueError("feature_names ne peut pas être vide.")

        # Extraire les importances (disponible pour RandomForest et GradientBoosting)
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(
                f"Le modèle {type(self.model).__name__} ne supporte pas l'extraction "
                f"d'importance des features."
            )

        importance_values = self.model.feature_importances_

        # Vérifier que le nombre de noms correspond au nombre de features
        n_features = len(importance_values)
        n_names = len(feature_names)

        if n_names != n_features:
            raise ValueError(
                f"Le nombre de noms de features ({n_names}) ne correspond pas "
                f"au nombre de features du modèle ({n_features})."
            )

        # Créer un DataFrame avec les noms et les importances
        feature_importance = pd.DataFrame({
            'feature': list(feature_names),
            'importance': importance_values
        })

        # Trier par ordre décroissant d'importance
        feature_importance = feature_importance.sort_values(
            by='importance',
            ascending=False
        ).reset_index(drop=True)

        return feature_importance

    def compute_r2_score(self, X, y):
        """
        Calcule le coefficient de détermination R² du modèle.

        Le R² (R-squared) mesure la proportion de la variance dans la variable cible
        qui est prédictible à partir des features. Il varie généralement entre 0 et 1,
        où 1 indique une prédiction parfaite et 0 indique que le modèle n'est pas
        meilleur qu'une simple moyenne. Des valeurs négatives sont possibles si le
        modèle est pire que la prédiction par la moyenne.

        Formule : R² = 1 - (SS_res / SS_tot)
        où SS_res = Σ(y_true - y_pred)² et SS_tot = Σ(y_true - y_mean)²

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles calculer le R².
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.
            y (pd.Series ou np.ndarray): Les valeurs cibles réelles.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            float: Le coefficient de détermination R². Une valeur proche de 1 indique
                   un bon ajustement du modèle, tandis qu'une valeur proche de 0
                   indique un ajustement médiocre.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant de calculer le R²."
            )

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

        # Faire les prédictions
        y_pred = self.predict(X)

        # Calculer le R² score
        r2 = r2_score(y, y_pred)

        return r2

    def compute_rmse(self, X, y):
        """
        Calcule l'erreur quadratique moyenne (RMSE - Root Mean Squared Error) du modèle.

        Le RMSE mesure l'écart-type des erreurs de prédiction (résidus). Il indique
        à quel point les prédictions sont proches des valeurs réelles, dans les mêmes
        unités que la variable cible. Plus le RMSE est faible, meilleure est la
        précision du modèle.

        Formule : RMSE = √(Σ(y_true - y_pred)² / n)

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles calculer le RMSE.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.
            y (pd.Series ou np.ndarray): Les valeurs cibles réelles.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            float: Le RMSE. Une valeur faible indique que les prédictions sont proches
                   des valeurs réelles. La valeur est toujours positive (>= 0).

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant de calculer le RMSE."
            )

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

        # Faire les prédictions
        y_pred = self.predict(X)

        # Calculer le RMSE
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        return rmse

    def compute_mae(self, X, y):
        """
        Calcule l'erreur absolue moyenne (MAE - Mean Absolute Error) du modèle.

        Le MAE mesure la moyenne des valeurs absolues des erreurs de prédiction.
        Il donne une idée de l'amplitude moyenne des erreurs du modèle, dans les mêmes
        unités que la variable cible. Plus le MAE est faible, meilleure est la
        précision du modèle. Contrairement au RMSE, le MAE est moins sensible aux
        valeurs aberrantes car il n'élève pas les erreurs au carré.

        Formule : MAE = Σ|y_true - y_pred| / n

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles calculer le MAE.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.
            y (pd.Series ou np.ndarray): Les valeurs cibles réelles.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            float: Le MAE. Une valeur faible indique que les prédictions sont proches
                   des valeurs réelles. La valeur est toujours positive (>= 0).

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant de calculer le MAE."
            )

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

        # Faire les prédictions
        y_pred = self.predict(X)

        # Calculer le MAE
        mae = np.mean(np.abs(y - y_pred))

        return mae

    def evaluate(self, X, y):
        """
        Évalue le modèle et retourne toutes les métriques de performance dans un dictionnaire.

        Cette méthode calcule toutes les métriques d'évaluation du modèle de machine learning :
        - R² (coefficient de détermination)
        - RMSE (erreur quadratique moyenne)
        - MAE (erreur absolue moyenne)

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles évaluer le modèle.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.
            y (pd.Series ou np.ndarray): Les valeurs cibles réelles.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            dict: Dictionnaire contenant toutes les métriques d'évaluation :
                - 'r2' (float): Le coefficient de détermination R².
                - 'rmse' (float): L'erreur quadratique moyenne.
                - 'mae' (float): L'erreur absolue moyenne.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Calculer toutes les métriques en utilisant les méthodes existantes
        metrics = {
            'r2': self.compute_r2_score(X, y),
            'rmse': self.compute_rmse(X, y),
            'mae': self.compute_mae(X, y)
        }

        return metrics
