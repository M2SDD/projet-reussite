#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 09/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module d'évaluation et de comparaison de modèles de prédiction

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .config import Config


# ----------------------------------------------------------------------------------------------------------------------
# Classe ModelEvaluator
# ----------------------------------------------------------------------------------------------------------------------
class ModelEvaluator:
    """
    Classe responsable de l'évaluation et de la comparaison de plusieurs modèles de prédiction.

    Cette classe fournit un framework unifié pour évaluer et comparer n'importe quel nombre
    de modèles entraînés. Elle produit des tableaux de comparaison avec R², RMSE, MAE et
    R² ajusté côte à côte, et génère des visualisations comparatives : graphiques de
    prédictions vs valeurs réelles, distributions des résidus, et graphiques à barres
    des métriques pour chaque modèle.

    Attributes:
        config (Config): Instance de configuration pour les paramètres de l'application.
        models (dict): Dictionnaire stockant les modèles enregistrés avec leur nom comme clé.

    Examples:
        Utilisation basique avec enregistrement de modèles:

        >>> from src.model_evaluator import ModelEvaluator
        >>> from src.regression_model import RegressionModel
        >>> evaluator = ModelEvaluator()
        >>> model = RegressionModel()
        >>> evaluator.register_model('linear_regression', model)

        Comparaison de plusieurs modèles:

        >>> evaluator = ModelEvaluator()
        >>> evaluator.register_model('model1', model1)
        >>> evaluator.register_model('model2', model2)
        >>> results = evaluator.evaluate_all(X_test, y_test)
    """

    def __init__(self, config=None):
        """
        Initialise l'évaluateur de modèles.

        Args:
            config (Config, optional): Instance de configuration. Si None, utilise la configuration par défaut.
        """
        self.config = config if config is not None else Config()
        self.models = {}

    def register_model(self, name, model):
        """
        Enregistre un modèle pour l'évaluation et la comparaison.

        Cette méthode permet d'ajouter un modèle au framework de comparaison.
        Le modèle peut être de n'importe quel type (régression linéaire, forêt aléatoire,
        gradient boosting, etc.) tant qu'il possède une méthode predict().

        Args:
            name (str): Nom unique identifiant le modèle (ex: 'linear_regression', 'random_forest').
            model: Instance du modèle entraîné. Doit avoir une méthode predict(X).

        Raises:
            ValueError: Si le nom est vide ou si le modèle est None.
            ValueError: Si un modèle avec ce nom existe déjà.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> from src.regression_model import RegressionModel
            >>> model = RegressionModel()
            >>> model.fit(X_train, y_train)
            >>> evaluator.register_model('linear_regression', model)
        """
        # Validation du nom
        if not name or not isinstance(name, str):
            raise ValueError("Le nom du modèle doit être une chaîne non vide.")

        # Validation du modèle
        if model is None:
            raise ValueError("Le modèle ne peut pas être None.")

        # Vérifier si le nom existe déjà
        if name in self.models:
            raise ValueError(
                f"Un modèle avec le nom '{name}' existe déjà. "
                f"Veuillez choisir un nom différent ou supprimer le modèle existant."
            )

        # Enregistrer le modèle (sans données d'évaluation)
        self.models[name] = {
            'model': model,
            'X': None,
            'y': None
        }

    def add_model(self, name, model, X, y):
        """
        Enregistre un modèle avec ses données d'évaluation.

        Cette méthode permet d'ajouter un modèle au framework de comparaison avec
        les données nécessaires pour l'évaluation automatique. Le modèle peut être
        de n'importe quel type tant qu'il possède une méthode predict().

        Args:
            name (str): Nom unique identifiant le modèle (ex: 'linear_regression', 'random_forest').
            model: Instance du modèle entraîné. Doit avoir une méthode predict(X).
            X (pd.DataFrame ou np.ndarray): Features pour l'évaluation du modèle.
                Peut être un DataFrame pandas ou un tableau numpy de shape (n_samples, n_features).
            y (pd.Series ou np.ndarray): Valeurs cibles réelles pour l'évaluation.
                Peut être une Series pandas ou un tableau numpy de shape (n_samples,).

        Raises:
            ValueError: Si le nom est vide ou si le modèle est None.
            ValueError: Si X ou y sont vides ou None.
            ValueError: Si X et y n'ont pas le même nombre d'échantillons.
            ValueError: Si un modèle avec ce nom existe déjà.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> from src.regression_model import RegressionModel
            >>> model = RegressionModel()
            >>> model.fit(X_train, y_train)
            >>> evaluator.add_model('linear_regression', model, X_test, y_test)
        """
        # Validation du nom
        if not name or not isinstance(name, str):
            raise ValueError("Le nom du modèle doit être une chaîne non vide.")

        # Validation du modèle
        if model is None:
            raise ValueError("Le modèle ne peut pas être None.")

        # Validation des données X
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X ne peut pas être vide.")

        # Validation des données y
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

        # Vérifier si le nom existe déjà
        if name in self.models:
            raise ValueError(
                f"Un modèle avec le nom '{name}' existe déjà. "
                f"Veuillez choisir un nom différent ou supprimer le modèle existant."
            )

        # Enregistrer le modèle avec ses données d'évaluation
        self.models[name] = {
            'model': model,
            'X': X,
            'y': y
        }

    def evaluate_all(self):
        """
        Évalue tous les modèles enregistrés et retourne leurs métriques de performance.

        Cette méthode calcule les métriques de performance (R², RMSE, MAE, R² ajusté)
        pour tous les modèles qui ont été enregistrés avec des données d'évaluation
        via la méthode add_model(). Les modèles enregistrés sans données (via register_model())
        sont ignorés.

        Les métriques calculées sont :
        - R² (coefficient de détermination) : Mesure la proportion de variance expliquée (0-1)
        - RMSE (Root Mean Squared Error) : Erreur quadratique moyenne (plus faible = meilleur)
        - MAE (Mean Absolute Error) : Erreur absolue moyenne (plus faible = meilleur)
        - R² ajusté : R² ajusté pour le nombre de features (pénalise la complexité)

        Returns:
            dict: Dictionnaire avec les noms de modèles comme clés et un dictionnaire
                  de métriques comme valeurs. Chaque dictionnaire de métriques contient :
                  - 'r2' (float): Coefficient de détermination R²
                  - 'rmse' (float): Erreur quadratique moyenne
                  - 'mae' (float): Erreur absolue moyenne
                  - 'adjusted_r2' (float): R² ajusté

        Raises:
            ValueError: Si aucun modèle n'a été enregistré avec des données d'évaluation.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> evaluator.add_model('model1', model1, X_test, y_test)
            >>> evaluator.add_model('model2', model2, X_test, y_test)
            >>> results = evaluator.evaluate_all()
            >>> print(results['model1']['r2'])  # Affiche le R² du modèle 1
        """
        # Vérifier qu'il y a au moins un modèle avec des données d'évaluation
        models_with_data = {
            name: data for name, data in self.models.items()
            if data['X'] is not None and data['y'] is not None
        }

        if not models_with_data:
            raise ValueError(
                "Aucun modèle avec données d'évaluation n'a été enregistré. "
                "Veuillez utiliser la méthode add_model() pour enregistrer des modèles "
                "avec leurs données d'évaluation."
            )

        # Calculer les métriques pour chaque modèle
        results = {}

        for name, model_data in models_with_data.items():
            model = model_data['model']
            X = model_data['X']
            y = model_data['y']

            # Faire les prédictions
            y_pred = model.predict(X)

            # Calculer les métriques de base
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            # Calculer le R² ajusté
            # Formule : R²_adj = 1 - ((1 - R²) * (n - 1) / (n - p - 1))
            # où n = nombre d'échantillons, p = nombre de features
            n = len(y)
            p = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1

            if n > p + 1:
                adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            else:
                # Si n <= p + 1, le R² ajusté n'est pas défini, on utilise le R² normal
                adjusted_r2 = r2

            # Stocker les résultats
            results[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'adjusted_r2': adjusted_r2
            }

        return results

    def get_comparison_table(self):
        """
        Crée un tableau de comparaison avec les métriques de tous les modèles côte à côte.

        Cette méthode génère un DataFrame pandas où chaque ligne représente un modèle
        et chaque colonne représente une métrique de performance (R², RMSE, MAE, R² ajusté).
        Cela permet une comparaison visuelle facile des performances de tous les modèles
        enregistrés.

        Les métriques incluses sont :
        - R² (coefficient de détermination)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - R² ajusté (adjusted R²)

        Returns:
            pd.DataFrame: DataFrame avec les modèles en lignes (index) et les métriques
                          en colonnes. Les valeurs sont formatées avec 4 décimales.
                          Les métriques manquantes sont représentées par NaN.

        Raises:
            ValueError: Si aucun modèle n'a été enregistré avec des données d'évaluation.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> evaluator.add_model('regression', model1, X_test, y_test)
            >>> evaluator.add_model('random_forest', model2, X_test, y_test)
            >>> table = evaluator.get_comparison_table()
            >>> print(table)
                              r2    rmse     mae  adjusted_r2
            regression    0.9500  0.1000  0.0800       0.9400
            random_forest 0.9200  0.1200  0.0950       0.9100
        """
        # Obtenir les métriques de tous les modèles
        metrics_dict = self.evaluate_all()

        # Convertir le dictionnaire en DataFrame
        # Les clés du dictionnaire deviennent l'index (noms des modèles)
        # Les sous-dictionnaires deviennent les colonnes
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')

        # Arrondir toutes les valeurs à 4 décimales
        df = df.round(4)

        return df
