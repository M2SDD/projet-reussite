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
    def plot_residuals(regressor: BaseRegressor,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       ax: plt.Axes = None,
                       ):
        """Crée un graphique des résidus du modèle."""
        # Calculer les prédictions et les résidus
        y_pred = regressor.predict(X)
        residuals = regressor.compute_residuals(X, y)

        # Création d'un axe si aucun n'est fourni
        if ax is None:
            ax = plt.gca()

        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='y=0')
        ax.set_xlabel('Valeurs prédites', fontsize=12)
        ax.set_ylabel('Résidus', fontsize=12)
        ax.set_title('Graphique des résidus', fontsize=14, fontweight='bold')
        ax.legend()
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