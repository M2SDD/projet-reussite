#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
"""
Module orchestrateur pour la préparation complète des données.

Ce module combine le DataLoader, le FeatureExtractor et le DataCleaner
pour produire un dataset prêt à l'emploi pour les modèles de Machine Learning.
"""
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split

from ..config import Config
from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .data_cleaner import DataCleaner


class DatasetBuilder:
    """
    Classe orchestrant tout le pipeline de préparation des données :
    Chargement -> Extraction de variables -> Jointure -> Nettoyage -> Split.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialise le constructeur avec ses dépendances."""
        self.config = config if config is not None else Config()
        self.loader = DataLoader(self.config)
        self.extractor = FeatureExtractor(self.config)
        self.cleaner = DataCleaner(self.config)

    def build_dataset(self,
                      logs_path: str,
                      notes_path: str,
                      drop_inactive_students: bool = True,
                      remove_outliers: bool = False,
                      selection_methods: Optional[Union[str, List[str]]] = ['linear', 'mutual_info', 'rfe'],
                      k_features: int = 15,
                      prefilter_variance: bool = True,
                      prefilter_correlation: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Construit l'ensemble de données final X (features) et y (cible) en enchaînant
        toutes les étapes du pipeline.
        """
        # 1. Chargement des données
        df_logs = self.loader.load_logs(logs_path)
        df_notes = self.loader.load_notes(notes_path)

        # 2. suppression des doublons
        df_logs = self.cleaner.remove_duplicates(df_logs)
        df_notes = self.cleaner.remove_duplicates(df_notes)

        # 3. Extraction des features métiers depuis les logs
        df_features = self.extractor.extract_features(df_logs)

        # 4. Jointure (Merge)
        join_how = 'inner' if drop_inactive_students else 'left'

        df_final = pd.merge(
            df_notes,
            df_features,
            how=join_how,
            left_on=self.config.MERGE_KEY,
            right_index=True
        )

        # 5. Traitement des valeurs manquantes si drop_inactive_students=False
        df_final = df_final.fillna(0)  # étudiants qui ne vont jamais sur ARCHE (redoublants ?)

        # 6. Séparation X et y
        y = df_final[self.config.TARGET_COLUMN]
        X = df_final.drop(columns=[self.config.MERGE_KEY, self.config.TARGET_COLUMN])

        # 7. suppression des outliers (optionnel, à activer si nécessaire)
        if remove_outliers:
            X = self.cleaner.remove_outliers_iqr(X)

        # 8. Sélection et nettoyage avancé des variables
        selected_features = X.columns.tolist()
        if selection_methods is not None:
            X, selected_features = self.cleaner.select_features(
                X, y,
                methods=selection_methods,
                k=k_features,
                prefilter_variance=prefilter_variance,
                prefilter_correlation=prefilter_correlation
            )

        return X, y, selected_features

    def get_train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divise les données finales en ensembles d'entraînement et de test pour le Machine Learning.

        Args:
            X (pd.DataFrame): Les features prêtes pour la modélisation.
            y (pd.Series): La variable cible prête pour la modélisation.

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if X is None or y is None:
            raise ValueError("Les matrices X et y doivent être fournies.")

        test_size = getattr(self.config, 'TEST_SPLIT_RATIO', 0.2)
        random_state = getattr(self.config, 'RANDOM_STATE', 42)

        return train_test_split(X, y, test_size=test_size, random_state=random_state)