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

        # Enregistrer le modèle
        self.models[name] = model
