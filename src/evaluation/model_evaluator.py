#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 09/04/2026
# version ='1.0'
# TODO : ajouter un attribut 'can_use_adjusted_r2' plutôt que d'utiliser des booléens ?
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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Optional, Union

from ..config import Config
from ..models.base_regressor import BaseRegressor
from ..visualization.model_visualizer import ModelVisualizer

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

        >>> from src import ModelEvaluator
        >>> from src import LinearRegressor
        >>> evaluator = ModelEvaluator()
        >>> model = LinearRegressor()
        >>> evaluator.add_model('linear_regression', model, X_test, y_test)

        Comparaison de plusieurs modèles:

        >>> evaluator = ModelEvaluator()
        >>> evaluator.add_model('model1', model1, X, y)
        >>> evaluator.add_model('model2', model2, X_reduced, y)
        >>> results = evaluator.evaluate_all()
    """

    def __init__(self, config=None):
        """Initialise l'évaluateur de modèles."""
        self.config = config if config is not None else Config()
        self.models: Dict[str, Dict[str, Any]] = {}

    def add_model(self, name: str, model: BaseRegressor,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray]) -> None:
        """Enregistre un modèle avec ses données d'évaluation spécifiques."""
        if not name or not isinstance(name, str):
            raise ValueError("Le nom du modèle doit être une chaîne non vide.")

        if model is None:
            raise ValueError("Le modèle ne peut pas être None.")

        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X ne peut pas être vide.")

        if y is None or (hasattr(y, '__len__') and len(y) == 0):
            raise ValueError("y ne peut pas être vide.")

        if len(X) != len(y):
            raise ValueError("X et y doivent avoir le même nombre d'échantillons.")

        if name in self.models:
            raise ValueError(f"Un modèle avec le nom '{name}' existe déjà.")

        self.models[name] = {
            'model': model,
            'X': X,
            'y': y
        }

    def _valid_model_data(self):
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

        return models_with_data

    def evaluate_all(self, include_adjusted_r2=False) -> Dict[str, Dict[str, float]]:
        """
        Évalue tous les modèles enregistrés et retourne leurs métriques de performance.

        Cette méthode calcule les métriques de performance (R², RMSE, MAE, R² ajusté)
        pour tous les modèles qui ont été enregistrés avec des données d'évaluation
        via la méthode add_model().
        """
        # Vérifier qu'il y a au moins un modèle avec des données d'évaluation
        models_with_data = self._valid_model_data()

        # Calculer les métriques pour chaque modèle
        results = {}

        for name, model_data in models_with_data.items():
            model: BaseRegressor = model_data['model']
            X = model_data['X']
            y = model_data['y']

            # Délégation totale au modèle pour le calcul de toutes les métriques
            results[name] = model.evaluate(X, y, include_adjusted_r2=include_adjusted_r2)

        return results

    def get_comparison_table(self, include_adjusted_r2=False):
        """Crée un tableau de comparaison avec les métriques de tous les modèles côte à côte."""
        metrics_dict = self.evaluate_all(include_adjusted_r2=include_adjusted_r2)
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        return df.round(4)

    def _set_plot_layout(self, models_with_data):
        """
        Cette méthode génère une grille de sous-graphiques (subplots) où chaque graphique
        représente un modèle et affiche ses prédictions par rapport aux valeurs réelles.
        """
        n_models = len(models_with_data)
        n_cols = min(3, n_models)  # Maximum 3 colonnes
        n_rows = (n_models + n_cols - 1) // n_cols  # Division arrondie vers le haut

        # Créer la figure et les axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        # S'assurer que axes est toujours un tableau, même avec un seul subplot
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Masquer les axes vides s'il y en a
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        return fig, axes


    def plot_predictions(self):
        """Crée des graphiques de prédictions vs valeurs réelles pour tous les modèles."""
        # Vérifier qu'il y a au moins un modèle avec des données d'évaluation
        models_with_data = self._valid_model_data()
        # Créer la figure et les axes
        fig, axes = self._set_plot_layout(models_with_data)

        # Créer un graphique pour chaque modèle
        for idx, (name, model_data) in enumerate(models_with_data.items()):
            model = model_data['model']
            X = model_data['X']
            y = model_data['y']
            ax = axes[idx]

            # Faire les prédictions
            ModelVisualizer.plot_predictions(model, X, y, ax=ax)

            r2 = model.compute_r2_score(X, y)  # Calculer le R² pour le titre
            ax.set_title(f'{name}\n(R² = {r2:.4f})', fontsize=14, fontweight='bold')

        # Ajuster la mise en page
        fig.tight_layout()

        return fig

    def plot_residuals(self):
        """Crée des graphiques de distribution des résidus pour tous les modèles."""
        # Vérifier qu'il y a au moins un modèle avec des données d'évaluation
        models_with_data = self._valid_model_data()
        # Créer la figure et les axes
        fig, axes = self._set_plot_layout(models_with_data)

        # Créer un graphique pour chaque modèle
        for idx, (name, model_data) in enumerate(models_with_data.items()):
            model = model_data['model']
            X = model_data['X']
            y = model_data['y']
            ax = axes[idx]

            # Délégation du tracé au ModelVisualizer en lui passant les données spécifiques au modèle
            ModelVisualizer.plot_residuals(model, X, y, ax=ax)

            # Surcharge du titre pour inclure le nom du modèle
            ax.set_title(f'Distribution des résidus - {name}', fontsize=14, fontweight='bold')

        # Ajuster la mise en page
        fig.tight_layout()

        return fig

    def plot_metrics_comparison(self, include_adjusted_r2=False):
        """Crée un graphique à barres comparant les métriques de tous les modèles."""
        # Obtenir les métriques de tous les modèles
        metrics_dict = self.evaluate_all(include_adjusted_r2=include_adjusted_r2)

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

    def get_recommendation(self, use_adjusted_r2=False):
        """
        Fournit une recommandation sur le meilleur modèle basée sur les métriques de performance.

        Cette méthode analyse tous les modèles enregistrés et sélectionne le meilleur en fonction
        de leurs métriques de performance. Le critère principal est le R² ajusté (adjusted_r2),
        car il prend en compte la complexité du modèle. En cas d'égalité, le RMSE est utilisé
        comme critère secondaire (plus faible est meilleur).

        Le R² ajusté est préféré au R² simple car il pénalise l'ajout de features inutiles,
        ce qui évite le sur-ajustement et favorise les modèles plus simples.
        """
        # Obtenir les métriques de tous les modèles
        all_metrics = self.evaluate_all(include_adjusted_r2=use_adjusted_r2)

        # Trouver le modèle avec le meilleur R² ajusté
        # En cas d'égalité, utiliser le RMSE le plus faible comme critère secondaire
        best_model = None
        best_metrics = None
        best_r2 = -np.inf
        best_rmse = np.inf

        for model_name, metrics in all_metrics.items():
            r2 = getattr(metrics, 'adjusted_r2', metrics['r2'])

            rmse = metrics['rmse']

            # Comparer d'abord par R² ajusté (plus élevé est meilleur)
            if r2 > best_r2:
                best_model = model_name
                best_metrics = metrics
                best_r2 = r2
                best_rmse = rmse
            # En cas d'égalité de R² ajusté, utiliser le RMSE (plus faible est meilleur)
            elif r2 == best_r2 and rmse < best_rmse:
                best_model = model_name
                best_metrics = metrics
                best_r2 = r2
                best_rmse = rmse

        # Construire la raison de la recommandation
        reason = f"Meilleur R²{' ajusté' if use_adjusted_r2 else ''} ({best_r2:.4f})"

        # Si plusieurs modèles ont le même R² ajusté, mentionner le RMSE
        models_with_same_r2 = [
            name for name, m in all_metrics.items()
            if getattr(m, 'adjusted_r2', m['r2']) == best_r2
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

    def export_results(self, output_dir, include_adjusted_r2=False):
        """
        Exporte les résultats de comparaison (tableau et graphiques) dans un répertoire.
        Elle crée automatiquement le répertoire s'il n'existe pas, puis exporte :
        - Le tableau de comparaison des métriques au format CSV
        - Le graphique des prédictions vs valeurs réelles au format PNG
        - Le graphique de distribution des résidus au format PNG
        - Le graphique de comparaison des métriques au format PNG
        """
        import os

        # Validation du répertoire de sortie
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("Le répertoire de sortie doit être une chaîne non vide.")

        # Créer le répertoire s'il n'existe pas
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Impossible de créer le répertoire '{output_dir}': {e}"
            )

        # Récupération des configurations des outputs
        plot_format = getattr(self.config, 'PLOT_SAVE_FORMAT', 'png')
        plot_dpi = getattr(self.config, 'PLOT_DPI', 300)

        # 1. Exporter le tableau de comparaison en CSV
        comparison_table = self.get_comparison_table(include_adjusted_r2=include_adjusted_r2)
        csv_path = os.path.join(output_dir, 'comparison_table.csv')
        comparison_table.to_csv(csv_path)

        # 2. Exporter le graphique des prédictions
        predictions_fig = self.plot_predictions()
        predictions_path = os.path.join(output_dir, f'predictions_comparison.{plot_format}')
        predictions_fig.savefig(predictions_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(predictions_fig)

        # 3. Exporter le graphique des résidus
        residuals_fig = self.plot_residuals()
        residuals_path = os.path.join(output_dir, f'residuals_comparison.{plot_format}')
        residuals_fig.savefig(residuals_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(residuals_fig)

        # 4. Exporter le graphique de comparaison des métriques
        metrics_fig = self.plot_metrics_comparison(include_adjusted_r2=include_adjusted_r2)
        metrics_path = os.path.join(output_dir, f'metrics_comparison.{plot_format}')
        metrics_fig.savefig(metrics_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(metrics_fig)
