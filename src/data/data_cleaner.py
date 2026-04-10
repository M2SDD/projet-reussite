#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
"""
Module dédié au nettoyage des données et à la sélection de variables (Feature Selection).
Ne contient que des opérations purement mathématiques/statistiques.
"""
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import warnings
from typing import Tuple, List, Optional
from sklearn.feature_selection import SelectKBest, f_regression

from ..config import Config


class DataCleaner:
    """
    Classe responsable de l'élimination des valeurs aberrantes (outliers)
    et de la sélection des variables les plus pertinentes.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config if config is not None else Config()

    def remove_outliers_iqr(self, X: pd.DataFrame, y: pd.Series, threshold: float = 1.5) -> Tuple[
        pd.DataFrame, pd.Series]:
        """
        Supprime les valeurs aberrantes de X et y en utilisant la méthode
        de l'écart interquartile (IQR).

        Args:
            X (pd.DataFrame): Les features.
            y (pd.Series): La variable cible.
            threshold (float): Multiplicateur de l'IQR (défaut: 1.5).
                               Un seuil plus élevé rend le filtre moins strict.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: X et y nettoyés.
        """
        if len(X) != len(y):
            raise ValueError("X et y doivent avoir la même taille.")

        # Calculer le premier quartile (Q1) et le troisième quartile (Q3) pour chaque colonne de X
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        # Identifier les lignes où aucune valeur n'est en dehors des limites
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Condition : True si toutes les valeurs de la ligne sont dans les bornes
        condition = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)

        X_clean = X[condition]
        y_clean = y[condition]

        # Avertir l'utilisateur de l'impact du nettoyage
        removed_count = len(X) - len(X_clean)
        if removed_count > 0:
            percentage = (removed_count / len(X)) * 100
            warnings.warn(
                f"{removed_count} valeurs aberrantes supprimées ({percentage:.1f}% des données) "
                f"en utilisant la méthode IQR (seuil={threshold}).",
                UserWarning
            )

        return X_clean, y_clean

    def select_top_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Sélectionne les 'k' meilleures features en fonction de leur corrélation
        linéaire avec la cible (f_regression).

        Args:
            X (pd.DataFrame): Les features.
            y (pd.Series): La variable cible.
            k (int): Le nombre de features à conserver.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Le DataFrame filtré et la liste des features sélectionnées.
        """
        n_features = X.shape[1]
        k = min(k, n_features)

        if k <= 0:
            raise ValueError("Le nombre de features k doit être supérieur à 0.")

        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected_array = selector.fit_transform(X, y)

        # Récupérer les noms des colonnes sélectionnées
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        # Reconstruire un DataFrame propre avec les colonnes sélectionnées
        X_selected = pd.DataFrame(
            X_selected_array,
            columns=selected_features,
            index=X.index
        )

        return X_selected, selected_features