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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

    def fit(self, X, y):
        """
        Entraîne le modèle de régression linéaire sur les données fournies.

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

        Utilise le modèle de régression linéaire entraîné pour faire des prédictions
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

    def get_coefficients(self):
        """
        Extrait les coefficients et l'intercept du modèle de régression linéaire entraîné.

        Cette méthode retourne un dictionnaire contenant les coefficients (poids) associés
        à chaque feature ainsi que l'intercept (ordonnée à l'origine) du modèle.

        Returns:
            dict: Dictionnaire contenant :
                - 'coefficients' (np.ndarray): Tableau des coefficients de shape (n_features,).
                  Chaque valeur représente le poids de la feature correspondante.
                - 'intercept' (float): L'intercept (ordonnée à l'origine) du modèle.

        Raises:
            ValueError: Si le modèle n'a pas encore été entraîné.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant d'extraire les coefficients."
            )

        # Extraire les coefficients et l'intercept
        coefficients = {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }

        return coefficients

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

    def compute_adjusted_r2(self, X, y):
        """
        Calcule le coefficient de détermination ajusté (R² ajusté) du modèle.

        Le R² ajusté est une version modifiée du R² qui ajuste la valeur en fonction
        du nombre de features dans le modèle. Contrairement au R², le R² ajusté pénalise
        l'ajout de features qui n'améliorent pas significativement le modèle. Il est
        particulièrement utile pour comparer des modèles avec un nombre différent de
        variables explicatives.

        Formule : R²_ajusté = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
        où n = nombre d'échantillons, p = nombre de features

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles calculer le R² ajusté.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.
            y (pd.Series ou np.ndarray): Les valeurs cibles réelles.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            float: Le coefficient de détermination ajusté. Une valeur proche de 1 indique
                   un bon ajustement du modèle. Peut être négatif si le modèle est très mauvais.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant de calculer le R² ajusté."
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

        # Calculer le R² score
        r2 = self.compute_r2_score(X, y)

        # Obtenir le nombre d'échantillons et de features
        n = len(X)

        # Gérer les cas où X est un DataFrame ou un tableau numpy
        if hasattr(X, 'shape'):
            # X est un tableau numpy ou un DataFrame
            if len(X.shape) == 1:
                # X est un tableau 1D (une seule feature)
                p = 1
            else:
                # X est un tableau 2D
                p = X.shape[1]
        else:
            # X est une structure de données sans attribut shape
            # Essayer de convertir en array pour obtenir le nombre de features
            X_array = np.array(X)
            if len(X_array.shape) == 1:
                p = 1
            else:
                p = X_array.shape[1]

        # Calculer le R² ajusté
        # Formule : R²_ajusté = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
        if n - p - 1 <= 0:
            raise ValueError(
                f"Le nombre d'échantillons ({n}) doit être supérieur au nombre de features ({p}) + 1 "
                f"pour calculer le R² ajusté."
            )

        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        return adjusted_r2

    def compute_residuals(self, X, y):
        """
        Calcule les résidus du modèle de régression.

        Les résidus sont définis comme la différence entre les valeurs observées (réelles)
        et les valeurs prédites par le modèle : résidu = y_réel - y_prédit.
        Les résidus permettent d'analyser la qualité du modèle et de vérifier les hypothèses
        de la régression linéaire (normalité, homoscédasticité, indépendance).

        Formule : résidus = y_true - y_pred

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles calculer les résidus.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.
            y (pd.Series ou np.ndarray): Les valeurs cibles réelles.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            np.ndarray: Les résidus de shape (n_samples,). Des résidus proches de zéro
                       indiquent des bonnes prédictions. Les résidus doivent idéalement
                       suivre une distribution normale centrée sur zéro.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant de calculer les résidus."
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

        # Calculer les résidus
        residuals = y - y_pred

        # Convertir en numpy array si nécessaire
        if hasattr(residuals, 'values'):
            # residuals est une Series pandas
            residuals = residuals.values

        return residuals

    def evaluate(self, X, y):
        """
        Évalue le modèle et retourne toutes les métriques de performance dans un dictionnaire.

        Cette méthode calcule toutes les métriques d'évaluation du modèle de régression :
        - R² (coefficient de détermination)
        - RMSE (erreur quadratique moyenne)
        - MAE (erreur absolue moyenne)
        - R² ajusté (coefficient de détermination ajusté)

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
                - 'adjusted_r2' (float): Le coefficient de détermination ajusté.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Calculer toutes les métriques en utilisant les méthodes existantes
        metrics = {
            'r2': self.compute_r2_score(X, y),
            'rmse': self.compute_rmse(X, y),
            'mae': self.compute_mae(X, y),
            'adjusted_r2': self.compute_adjusted_r2(X, y)
        }

        return metrics

    def plot_residuals(self, X, y):
        """
        Crée un graphique des résidus du modèle de régression.

        Cette méthode génère un graphique de résidus (residual plot) qui affiche
        les résidus en fonction des valeurs prédites. Ce type de graphique permet
        de vérifier visuellement les hypothèses de la régression linéaire :
        - Homoscédasticité : la variance des résidus doit être constante
        - Indépendance : les résidus ne doivent pas montrer de pattern
        - Moyenne nulle : les résidus doivent être centrés autour de zéro

        Le graphique comprend :
        - Un scatter plot des résidus vs valeurs prédites
        - Une ligne horizontale à y=0 pour référence

        Args:
            X (pd.DataFrame ou np.ndarray): Les features pour lesquelles créer le graphique.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
                Doit avoir le même nombre de features que les données d'entraînement.
            y (pd.Series ou np.ndarray): Les valeurs cibles réelles.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Returns:
            matplotlib.figure.Figure: La figure matplotlib contenant le graphique des résidus.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné, si X ou y sont vides,
                       ou si leurs dimensions sont incompatibles.
        """
        # Vérifier que le modèle a été entraîné
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. "
                "Veuillez appeler la méthode fit() avant de créer un graphique des résidus."
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

        # Calculer les prédictions et les résidus
        y_pred = self.predict(X)
        residuals = self.compute_residuals(X, y)

        # Créer la figure et les axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Tracer le scatter plot des résidus vs valeurs prédites
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)

        # Ajouter une ligne horizontale à y=0
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='y=0')

        # Configurer les labels et le titre
        ax.set_xlabel('Valeurs prédites', fontsize=12)
        ax.set_ylabel('Résidus', fontsize=12)
        ax.set_title('Graphique des résidus', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Ajuster la mise en page
        fig.tight_layout()

        return fig
