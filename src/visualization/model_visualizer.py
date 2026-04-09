#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
"""
Module dédié à la visualisation des performances des modèles.
Prend en entrée un modèle entraîné et des données, et génère des graphiques.
"""
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy import stats
from typing import Union
import pandas as pd
import numpy as np

# On importe BaseModel uniquement pour le typage
from ..models.base_regressor import BaseRegressor


class ModelVisualizer:
    """
    Classe utilitaire pour générer des graphiques d'évaluation des modèles.
    """
    @staticmethod
    def plot_predictions(regressor: BaseRegressor,
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         ax: plt.Axes = None,
                         ):
        """Crée un graphique des prédictions vs valeurs réelles du modèle."""
        # Calculer les prédictions
        y_pred = regressor.predict(X)

        # Création d'un axe si aucun n'est fourni
        if ax is None:
            ax = plt.gca()

        # Tracer le scatter plot des prédictions vs valeurs réelles
        ax.scatter(y, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)

        # Calculer les limites pour la ligne diagonale
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))

        # Ajouter une ligne diagonale de référence y=x
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Prédiction parfaite (y=x)')

        # Calculer le R² pour le titre
        r2 = regressor.compute_r2_score(X, y)

        # Configurer les labels et le titre
        ax.set_xlabel('Valeurs réelles', fontsize=12)
        ax.set_ylabel('Valeurs prédites', fontsize=12)
        ax.set_title(f'Visualisation des prédictions\n(R² = {r2:.4f})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_residuals(regressor: BaseRegressor,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       ax: plt.Axes = None,
                       ):
        """Crée un graphique des résidus du modèle."""
        # Calculer les résidus
        residuals = regressor.compute_residuals(X, y)

        # Création d'un axe si aucun n'est fourni
        if ax is None:
            ax = plt.gca()

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
        ax.set_title(f'Distribution des résidus', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_qq_plot(regressor: BaseRegressor,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     ax: plt.Axes = None,
                     ):
        """Crée un graphique Q-Q (quantile-quantile) des résidus du modèle."""
        residuals = regressor.compute_residuals(X, y)


        # Création d'un axe si aucun n'est fourni
        if ax is None:
            ax = plt.gca()

        (theoretical_quantiles, ordered_values), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

        ax.scatter(theoretical_quantiles, ordered_values, alpha=0.6, edgecolors='k', linewidths=0.5)
        ax.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept,
                'r--', linewidth=1.5, label='Ligne de référence')

        ax.set_xlabel('Quantiles théoriques', fontsize=12)
        ax.set_ylabel('Quantiles empiriques', fontsize=12)
        ax.set_title('Graphique Q-Q des résidus', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax