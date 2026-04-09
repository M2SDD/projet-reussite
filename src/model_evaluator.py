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
import matplotlib.pyplot as plt
from scipy import stats
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

    def plot_predictions(self):
        """
        Crée des graphiques de prédictions vs valeurs réelles pour tous les modèles.

        Cette méthode génère une grille de sous-graphiques (subplots) où chaque graphique
        représente un modèle et affiche ses prédictions par rapport aux valeurs réelles.
        Chaque graphique comprend :
        - Un scatter plot des valeurs prédites vs valeurs réelles
        - Une ligne diagonale de référence (y=x) montrant la prédiction parfaite
        - Des labels et titres en français

        La grille de sous-graphiques est automatiquement dimensionnée en fonction du nombre
        de modèles enregistrés, avec un maximum de 3 colonnes par ligne.

        Returns:
            matplotlib.figure.Figure: La figure matplotlib contenant tous les graphiques.
                La figure contient une grille de sous-graphiques avec un graphique par modèle.

        Raises:
            ValueError: Si aucun modèle n'a été enregistré avec des données d'évaluation.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> evaluator.add_model('regression', model1, X_test, y_test)
            >>> evaluator.add_model('random_forest', model2, X_test, y_test)
            >>> fig = evaluator.plot_predictions()
            >>> fig.savefig('predictions.png')
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

        # Calculer la disposition de la grille de sous-graphiques
        n_models = len(models_with_data)
        n_cols = min(3, n_models)  # Maximum 3 colonnes
        n_rows = (n_models + n_cols - 1) // n_cols  # Division arrondie vers le haut

        # Créer la figure et les axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        # S'assurer que axes est toujours un tableau, même avec un seul subplot
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Créer un graphique pour chaque modèle
        for idx, (name, model_data) in enumerate(models_with_data.items()):
            model = model_data['model']
            X = model_data['X']
            y = model_data['y']

            # Faire les prédictions
            y_pred = model.predict(X)

            # Obtenir l'axe courant
            ax = axes[idx]

            # Tracer le scatter plot des prédictions vs valeurs réelles
            ax.scatter(y, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)

            # Calculer les limites pour la ligne diagonale
            min_val = min(np.min(y), np.min(y_pred))
            max_val = max(np.max(y), np.max(y_pred))

            # Ajouter une ligne diagonale de référence y=x
            ax.plot([min_val, max_val], [min_val, max_val],
                   'r--', linewidth=2, label='Prédiction parfaite (y=x)')

            # Calculer le R² pour le titre
            r2 = r2_score(y, y_pred)

            # Configurer les labels et le titre
            ax.set_xlabel('Valeurs réelles', fontsize=12)
            ax.set_ylabel('Valeurs prédites', fontsize=12)
            ax.set_title(f'{name}\n(R² = {r2:.4f})', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Masquer les axes vides s'il y en a
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        # Ajuster la mise en page
        fig.tight_layout()

        return fig

    def plot_residuals(self):
        """
        Crée des graphiques de distribution des résidus pour tous les modèles.

        Cette méthode génère une grille de sous-graphiques (subplots) où chaque graphique
        représente un modèle et affiche la distribution de ses résidus (différences entre
        valeurs réelles et valeurs prédites). Chaque graphique comprend :
        - Un histogramme des résidus normalisé (densité de probabilité)
        - Une courbe de distribution normale théorique superposée
        - Une ligne verticale à x=0 indiquant la moyenne des résidus
        - Des labels et titres en français

        Les résidus sont calculés comme : résidus = valeurs réelles - valeurs prédites
        Une distribution proche de la normale centrée en 0 indique un bon modèle.

        La grille de sous-graphiques est automatiquement dimensionnée en fonction du nombre
        de modèles enregistrés, avec un maximum de 3 colonnes par ligne.

        Returns:
            matplotlib.figure.Figure: La figure matplotlib contenant tous les graphiques.
                La figure contient une grille de sous-graphiques avec un graphique par modèle.

        Raises:
            ValueError: Si aucun modèle n'a été enregistré avec des données d'évaluation.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> evaluator.add_model('regression', model1, X_test, y_test)
            >>> evaluator.add_model('random_forest', model2, X_test, y_test)
            >>> fig = evaluator.plot_residuals()
            >>> fig.savefig('residuals.png')
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

        # Calculer la disposition de la grille de sous-graphiques
        n_models = len(models_with_data)
        n_cols = min(3, n_models)  # Maximum 3 colonnes
        n_rows = (n_models + n_cols - 1) // n_cols  # Division arrondie vers le haut

        # Créer la figure et les axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        # S'assurer que axes est toujours un tableau, même avec un seul subplot
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Créer un graphique pour chaque modèle
        for idx, (name, model_data) in enumerate(models_with_data.items()):
            model = model_data['model']
            X = model_data['X']
            y = model_data['y']

            # Faire les prédictions
            y_pred = model.predict(X)

            # Calculer les résidus
            residuals = y - y_pred

            # Obtenir l'axe courant
            ax = axes[idx]

            # Tracer l'histogramme des résidus (normalisé en densité)
            n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7,
                                       color='skyblue', edgecolor='black', linewidth=0.5)

            # Calculer la courbe de distribution normale théorique
            mu = np.mean(residuals)
            sigma = np.std(residuals)

            # Générer les points pour la courbe normale
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            normal_curve = stats.norm.pdf(x_range, mu, sigma)

            # Tracer la courbe normale
            ax.plot(x_range, normal_curve, 'r-', linewidth=2,
                   label=f'Distribution normale\n(μ={mu:.4f}, σ={sigma:.4f})')

            # Ajouter une ligne verticale à x=0 (résidu moyen idéal)
            ax.axvline(x=0, color='green', linestyle='--', linewidth=2,
                      label='Résidu moyen idéal (0)')

            # Ajouter une ligne verticale à la moyenne réelle des résidus
            ax.axvline(x=mu, color='orange', linestyle='--', linewidth=2,
                      label=f'Résidu moyen ({mu:.4f})')

            # Configurer les labels et le titre
            ax.set_xlabel('Résidus (Réel - Prédit)', fontsize=12)
            ax.set_ylabel('Densité de probabilité', fontsize=12)
            ax.set_title(f'Distribution des résidus - {name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Masquer les axes vides s'il y en a
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        # Ajuster la mise en page
        fig.tight_layout()

        return fig

    def plot_metrics_comparison(self, include_adjusted_r2=True):
        """
        Crée un graphique à barres comparant les métriques de tous les modèles.

        Cette méthode génère un graphique à barres groupées où chaque groupe représente
        un modèle et chaque barre dans le groupe représente une métrique de performance
        différente (R², RMSE, MAE, et optionnellement R² ajusté). Cela permet une
        comparaison visuelle rapide des performances de tous les modèles enregistrés.

        Le graphique comprend :
        - L'axe x avec les noms des modèles
        - Des barres groupées pour chaque métrique (R², RMSE, MAE, R² ajusté)
        - Une légende identifiant chaque métrique
        - Des labels et titres en français
        - Une grille pour faciliter la lecture des valeurs

        Args:
            include_adjusted_r2 (bool, optional): Si True, inclut le R² ajusté dans le graphique.
                Par défaut True.

        Returns:
            matplotlib.figure.Figure: La figure matplotlib contenant le graphique à barres.

        Raises:
            ValueError: Si aucun modèle n'a été enregistré avec des données d'évaluation.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> evaluator.add_model('regression', model1, X_test, y_test)
            >>> evaluator.add_model('random_forest', model2, X_test, y_test)
            >>> fig = evaluator.plot_metrics_comparison()
            >>> fig.savefig('metrics_comparison.png')

            >>> # Sans le R² ajusté
            >>> fig = evaluator.plot_metrics_comparison(include_adjusted_r2=False)
        """
        # Obtenir les métriques de tous les modèles
        metrics_dict = self.evaluate_all()

        # Extraire les noms des modèles et les métriques
        model_names = list(metrics_dict.keys())
        n_models = len(model_names)

        # Déterminer quelles métriques afficher
        if include_adjusted_r2:
            metric_keys = ['r2', 'rmse', 'mae', 'adjusted_r2']
            metric_labels = ['R²', 'RMSE', 'MAE', 'R² ajusté']
        else:
            metric_keys = ['r2', 'rmse', 'mae']
            metric_labels = ['R²', 'RMSE', 'MAE']

        n_metrics = len(metric_keys)

        # Préparer les données pour chaque métrique
        metric_data = {key: [] for key in metric_keys}
        for model_name in model_names:
            for key in metric_keys:
                metric_data[key].append(metrics_dict[model_name][key])

        # Configurer la position des barres
        x = np.arange(n_models)  # Positions des groupes de modèles
        width = 0.2  # Largeur de chaque barre
        offset = width * (n_metrics - 1) / 2  # Offset pour centrer les groupes

        # Créer la figure et l'axe
        fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), 6))

        # Créer les barres pour chaque métrique
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']  # Vert, Rouge, Orange, Bleu
        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            bar_position = x - offset + i * width
            ax.bar(bar_position, metric_data[key], width,
                  label=label, color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Configurer les labels et le titre
        ax.set_xlabel('Modèles', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur de la métrique', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des métriques de performance par modèle',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Ajuster la mise en page pour éviter le chevauchement
        fig.tight_layout()

        return fig

    def get_recommendation(self):
        """
        Fournit une recommandation sur le meilleur modèle basée sur les métriques de performance.

        Cette méthode analyse tous les modèles enregistrés et sélectionne le meilleur en fonction
        de leurs métriques de performance. Le critère principal est le R² ajusté (adjusted_r2),
        car il prend en compte la complexité du modèle. En cas d'égalité, le RMSE est utilisé
        comme critère secondaire (plus faible est meilleur).

        Le R² ajusté est préféré au R² simple car il pénalise l'ajout de features inutiles,
        ce qui évite le sur-ajustement et favorise les modèles plus simples.

        Returns:
            dict: Dictionnaire contenant la recommandation avec les clés suivantes :
                - 'best_model' (str): Nom du modèle recommandé
                - 'reason' (str): Explication de la recommandation
                - 'metrics' (dict): Métriques du modèle recommandé (r2, rmse, mae, adjusted_r2)
                - 'all_metrics' (dict): Métriques de tous les modèles pour comparaison

        Raises:
            ValueError: Si aucun modèle n'a été enregistré avec des données d'évaluation.

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> evaluator.add_model('regression', model1, X_test, y_test)
            >>> evaluator.add_model('random_forest', model2, X_test, y_test)
            >>> recommendation = evaluator.get_recommendation()
            >>> print(recommendation['best_model'])
            'random_forest'
            >>> print(recommendation['reason'])
            "Meilleur R² ajusté (0.9500)"
        """
        # Obtenir les métriques de tous les modèles
        all_metrics = self.evaluate_all()

        # Vérifier qu'il y a au moins un modèle
        if not all_metrics:
            raise ValueError(
                "Aucun modèle avec données d'évaluation n'a été enregistré. "
                "Veuillez utiliser la méthode add_model() pour enregistrer des modèles "
                "avec leurs données d'évaluation."
            )

        # Trouver le modèle avec le meilleur R² ajusté
        # En cas d'égalité, utiliser le RMSE le plus faible comme critère secondaire
        best_model = None
        best_metrics = None
        best_adjusted_r2 = -np.inf
        best_rmse = np.inf

        for model_name, metrics in all_metrics.items():
            adjusted_r2 = metrics['adjusted_r2']
            rmse = metrics['rmse']

            # Comparer d'abord par R² ajusté (plus élevé est meilleur)
            if adjusted_r2 > best_adjusted_r2:
                best_model = model_name
                best_metrics = metrics
                best_adjusted_r2 = adjusted_r2
                best_rmse = rmse
            # En cas d'égalité de R² ajusté, utiliser le RMSE (plus faible est meilleur)
            elif adjusted_r2 == best_adjusted_r2 and rmse < best_rmse:
                best_model = model_name
                best_metrics = metrics
                best_adjusted_r2 = adjusted_r2
                best_rmse = rmse

        # Construire la raison de la recommandation
        reason = f"Meilleur R² ajusté ({best_adjusted_r2:.4f})"

        # Si plusieurs modèles ont le même R² ajusté, mentionner le RMSE
        models_with_same_r2 = [
            name for name, m in all_metrics.items()
            if m['adjusted_r2'] == best_adjusted_r2
        ]
        if len(models_with_same_r2) > 1:
            reason += f" avec le RMSE le plus faible ({best_rmse:.4f})"

        # Retourner la recommandation
        return {
            'best_model': best_model,
            'reason': reason,
            'metrics': best_metrics,
            'all_metrics': all_metrics
        }
