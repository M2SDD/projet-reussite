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
from typing import Tuple, List, Optional, Union
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor

from ..config import Config


class DataCleaner:
    """
    Classe responsable de l'élimination des valeurs aberrantes (outliers),
    du nettoyage des variables redondantes, et de la sélection des variables les plus pertinentes.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config if config is not None else Config()

    def remove_duplicates(self, df):
        """
        Supprime les lignes dupliquées du DataFrame.
        """
        initial_count = len(df)
        df_clean = df.drop_duplicates(
            keep=self.config.DUPLICATE_KEEP,
            subset=self.config.DUPLICATE_SUBSET,
        )
        removed_count = initial_count - len(df_clean)

        if removed_count > 0:
            warnings.warn(
                f"{removed_count} lignes dupliquées supprimées.",
                UserWarning,
            )

        return df_clean.reset_index(drop=True)

    def remove_outliers_iqr(self, X: pd.DataFrame, y: pd.Series, threshold: float = 1.5,
                            strategy: str = 'majority') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Supprime les valeurs aberrantes de X et y en utilisant la méthode
        de l'écart interquartile (IQR).

        Args:
            strategy (str): Critère de suppression d'une ligne.
                - 'any'      : supprime si AU MOINS UNE feature est hors borne (défaut, strict).
                - 'all'      : supprime seulement si TOUTES les features sont hors borne (permissif).
                - 'majority' : supprime si plus de 50% des features sont hors borne (recommandé
                               pour les jeux de données à haute dimensionnalité).
        """
        if len(X) != len(y):
            raise ValueError("X et y doivent avoir la même taille.")

        valid_strategies = ('any', 'all', 'majority')
        if strategy not in valid_strategies:
            raise ValueError(
                f"strategy doit être l'un de {valid_strategies}, reçu : '{strategy}'"
            )

        # Calculer le premier quartile (Q1) et le troisième quartile (Q3) pour chaque colonne de X
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        # Identifier les lignes où aucune valeur n'est en dehors des limites
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        is_outlier = (X < lower_bound) | (X > upper_bound)

        if strategy == 'any':
            condition = ~is_outlier.any(axis=1)
        elif strategy == 'all':
            condition = ~is_outlier.all(axis=1)
        else:  # majority
            condition = is_outlier.sum(axis=1) <= (X.shape[1] / 2)

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

    def remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Supprime les features dont la variance est quasi nulle (qui ont presque toujours la même valeur).
        Ex: Un composant ARCHE que personne n'utilise jamais.
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        features_to_keep = X.columns[selector.get_support()]

        removed_features = set(X.columns) - set(features_to_keep)
        if removed_features:
            warnings.warn(
                f"{len(removed_features)} variables à faible variance supprimées : {list(removed_features)}",
                UserWarning
            )

        return X[features_to_keep]

    def remove_highly_correlated_features(self, X: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
        """
        Supprime les features fortement corrélées entre elles (colinéarité) pour éviter
        de donner des informations redondantes au modèle.

        Pour chaque paire corrélée au-delà du seuil, la feature avec la variance
        la plus faible est supprimée (celle avec la plus haute variance est conservée
        car elle porte davantage d'information brute).
        """
        # Calculer la matrice de corrélation absolue
        corr_matrix = X.corr().abs()

        # Sélectionner le triangle supérieur de la matrice de corrélation
        upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper = corr_matrix.where(upper_mask)

        # Pour chaque paire corrélée, supprimer celle avec la variance la plus faible
        to_drop = set()
        for col in upper.columns:
            correlated_partners = upper.index[upper[col] > threshold].tolist()
            for partner in correlated_partners:
                if X[col].var() < X[partner].var():
                    to_drop.add(col)
                else:
                    to_drop.add(partner)

        if to_drop:
            warnings.warn(
                f"{len(to_drop)} variables fortement corrélées (>{threshold}) supprimées : {sorted(to_drop)}",
                UserWarning
            )

        return X.drop(columns=list(to_drop))

    def select_top_features_linear(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Sélectionne les 'k' meilleures features en fonction de leur corrélation
        linéaire avec la cible (test F). Idéal pour la Régression Linéaire.
        """
        n_features = X.shape[1]
        k = min(k, n_features)

        if k <= 0:
            raise ValueError("Le nombre de features k doit être supérieur à 0.")

        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        X_selected = X.loc[:, selected_features]
        return X_selected, selected_features

    def select_top_features_mutual_info(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[
        pd.DataFrame, List[str]]:
        """
        Sélectionne les 'k' meilleures features en utilisant l'Information Mutuelle.
        Capture n'importe quelle relation (linéaire ou non-linéaire) de manière univariée.
        """
        n_features = X.shape[1]
        k = min(k, n_features)

        if k <= 0:
            raise ValueError("Le nombre de features k doit être supérieur à 0.")

        # L'utilisation d'un lambda permet de passer le random_state pour garantir la reproductibilité
        random_state = getattr(self.config, 'RANDOM_STATE', 42)
        score_func = lambda X, y: mutual_info_regression(X, y, random_state=random_state, n_jobs=-1)

        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        X_selected = X.loc[:, selected_features]
        return X_selected, selected_features

    def select_features_by_importance(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[
        pd.DataFrame, List[str]]:
        """
        Sélectionne les 'k' meilleures features en utilisant l'importance des variables
        dérivée d'un Random Forest. Capture très bien les relations non linéaires avec interactions.
        Idéal pour les modèles ensemblistes.
        """
        n_features = X.shape[1]
        k = min(k, n_features)

        if k <= 0:
            raise ValueError("Le nombre de features k doit être supérieur à 0.")

        # Utiliser un Random Forest pour évaluer l'importance
        rf = RandomForestRegressor(
            n_estimators=getattr(self.config, 'RF_N_ESTIMATORS', 100),
            random_state=getattr(self.config, 'RANDOM_STATE', 42),
            n_jobs=-1
        )
        rf.fit(X, y)

        # Créer un DataFrame avec les importances et trier
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Garder le top K
        selected_features = importances['feature'].head(k).tolist()

        return X[selected_features], selected_features

    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Sélectionne les 'k' meilleures features en utilisant l'Élimination Récursive
        (Recursive Feature Elimination - RFE) avec un Random Forest.
        Plus lent que l'importance simple, mais potentiellement plus robuste.
        """
        n_features = X.shape[1]
        k = min(k, n_features)

        if k <= 0:
            raise ValueError("Le nombre de features k doit être supérieur à 0.")

        estimator = RandomForestRegressor(
            n_estimators=max(10, getattr(self.config, 'RF_N_ESTIMATORS', 100) // 4),  # réduit pour accélérer RFE
            random_state=getattr(self.config, 'RANDOM_STATE', 42),
            n_jobs=-1
        )

        selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        X_selected = X.loc[:, selected_features]
        return X_selected, selected_features

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        methods: Union[str, List[str]] = 'linear',
                        k: int = 10,
                        prefilter_variance: bool = True,
                        prefilter_correlation: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Méthode routeur globale pour la sélection de features, incluant le pré-nettoyage.
        Permet de combiner plusieurs méthodes (Ensemble Feature Selection).

        Args:
            X (pd.DataFrame): Les features brutes.
            y (pd.Series): La variable cible.
            methods (str ou List[str]): 'linear', 'mutual_info', 'importance', ou 'rfe'.
            k (int): Le nombre de features à conserver PAR méthode.
            prefilter_variance (bool): Appliquer le filtre de faible variance avant la sélection.
            prefilter_correlation (bool): Appliquer le filtre de colinéarité avant la sélection.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Le DataFrame filtré final et la liste des features sélectionnées.
        """
        # 1. Pré-filtrage (Nettoyage de base)
        if prefilter_variance:
            X = self.remove_low_variance_features(X)

        if prefilter_correlation:
            X = self.remove_highly_correlated_features(X)

        # 2. Sélection avancée
        valid_methods = {
            'linear': self.select_top_features_linear,
            'mutual_info': self.select_top_features_mutual_info,
            'importance': self.select_features_by_importance,
            'rfe': self.select_features_rfe
        }

        # Conversion en liste si une seule méthode est passée en string
        if isinstance(methods, str):
            methods = [methods]

        all_selected_features = set()

        # Application de l'Ensemble Feature Selection (Union des ensembles de k features)
        for method in methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Méthode de sélection inconnue : '{method}'. "
                    f"Méthodes valides : {list(valid_methods.keys())}"
                )

            # Extraire les k features pour cette méthode spécifique
            _, selected_for_method = valid_methods[method](X, y, k=k)
            all_selected_features.update(selected_for_method)

        # Reconvertir le set en liste triée (pour assurer la reproductibilité de l'ordre)
        final_features = sorted(list(all_selected_features))

        return X[final_features], final_features