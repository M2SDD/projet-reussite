#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 08/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module de traitement et d'analyse des données chargées

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
import warnings

from .config import Config


# ----------------------------------------------------------------------------------------------------------------------
# Classe DataProcessor
# ----------------------------------------------------------------------------------------------------------------------
class DataProcessor:
    """
    Classe responsable du traitement, de l'analyse et de la transformation des données.

    Cette classe fournit des méthodes de nettoyage et de prétraitement
    pour les données de logs et de notes exportées depuis ARCHE.
    """

    def __init__(self, config=None):
        """
        Initialise le processeur de données.

        Args:
            config (Config): Instance de configuration. Si None, utilise la configuration par défaut.
        """
        self.config = config if config is not None else Config()

    def remove_duplicates(self, df):
        """
        Supprime les lignes dupliquées exactes du DataFrame.

        Args:
            df (pd.DataFrame): Le DataFrame à nettoyer.

        Returns:
            pd.DataFrame: Le DataFrame sans doublons.
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

    def handle_missing_values(self, df):
        """
        Détecte et gère les valeurs manquantes (NaN/NaT) dans le DataFrame.

        Les valeurs manquantes sont signalées via un avertissement.
        Les lignes contenant des valeurs manquantes dans les colonnes clés
        sont conservées mais signalées.

        Args:
            df (pd.DataFrame): Le DataFrame à vérifier.

        Returns:
            pd.DataFrame: Le DataFrame avec les valeurs manquantes gérées.
        """
        missing_counts = df.isna().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            cols_with_missing = missing_counts[missing_counts > 0]
            details = ", ".join(
                f"{col}: {count}" for col, count in cols_with_missing.items()
            )
            warnings.warn(
                f"{total_missing} valeurs manquantes détectées ({details}).",
                UserWarning,
            )

        return df

    def clean_logs(self, df):
        """
        Orchestre le nettoyage complet des données de logs.

        Applique séquentiellement :
        1. Suppression des doublons
        2. Gestion des valeurs manquantes

        Args:
            df (pd.DataFrame): Le DataFrame de logs brut.

        Returns:
            pd.DataFrame: Le DataFrame de logs nettoyé.
        """
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        return df

    def process_data(self, data):
        """
        Traite et transforme les données brutes.

        Args:
            data (DataFrame): Les données à traiter.

        Returns:
            DataFrame: Les données traitées.
        """
        return self.clean_logs(data)
