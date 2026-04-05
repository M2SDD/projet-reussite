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

    def clean_notes(self, df):
        """
        Nettoie les données de notes : validation des plages, types et doublons.

        Applique séquentiellement :
        1. Suppression des doublons
        2. Gestion des valeurs manquantes
        3. Conversion de 'pseudo' en int et 'note' en float
        4. Clip des notes hors plage [NOTE_MIN, NOTE_MAX]

        Args:
            df (pd.DataFrame): Le DataFrame de notes brut.

        Returns:
            pd.DataFrame: Le DataFrame de notes nettoyé.
        """
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)

        # Ensure proper data types
        df['pseudo'] = pd.to_numeric(df['pseudo'], errors='coerce')
        df['note'] = pd.to_numeric(df['note'], errors='coerce')

        # Drop rows where pseudo is NaN after conversion
        invalid_pseudo = df['pseudo'].isna().sum()
        if invalid_pseudo > 0:
            warnings.warn(
                f"{invalid_pseudo} lignes avec pseudo invalide supprimées.",
                UserWarning,
            )
            df = df.dropna(subset=['pseudo'])

        df['pseudo'] = df['pseudo'].astype(int)
        df['note'] = df['note'].astype(float)

        # Flag and clip outlier notes
        out_of_range = (df['note'] < self.config.NOTE_MIN) | (df['note'] > self.config.NOTE_MAX)
        outlier_count = out_of_range.sum()

        if outlier_count > 0:
            warnings.warn(
                f"{outlier_count} notes hors plage [{self.config.NOTE_MIN}, {self.config.NOTE_MAX}] "
                f"ont été clippées.",
                UserWarning,
            )
            df['note'] = df['note'].clip(
                lower=self.config.NOTE_MIN,
                upper=self.config.NOTE_MAX,
            )

        return df.reset_index(drop=True)

    def validate_notes(self, df):
        """
        Valide les données de notes et retourne un rapport d'anomalies.

        Args:
            df (pd.DataFrame): Le DataFrame de notes à valider.

        Returns:
            dict: Rapport contenant les anomalies détectées avec les clés :
                - 'total_rows': nombre total de lignes
                - 'missing_pseudo': nombre de pseudos manquants
                - 'missing_note': nombre de notes manquantes
                - 'out_of_range': nombre de notes hors plage
                - 'duplicates': nombre de doublons
                - 'invalid_types': nombre de valeurs non numériques
                - 'is_valid': True si aucune anomalie détectée
        """
        report = {
            'total_rows': len(df),
            'missing_pseudo': int(df['pseudo'].isna().sum()),
            'missing_note': int(df['note'].isna().sum()),
            'out_of_range': 0,
            'duplicates': int(df.duplicated().sum()),
            'invalid_types': 0,
            'is_valid': True,
        }

        # Check for non-numeric values
        pseudo_numeric = pd.to_numeric(df['pseudo'], errors='coerce')
        note_numeric = pd.to_numeric(df['note'], errors='coerce')

        report['invalid_types'] = int(
            (pseudo_numeric.isna().sum() - df['pseudo'].isna().sum())
            + (note_numeric.isna().sum() - df['note'].isna().sum())
        )

        # Check for out-of-range notes (only on valid numeric values)
        valid_notes = note_numeric.dropna()
        report['out_of_range'] = int(
            ((valid_notes < self.config.NOTE_MIN) | (valid_notes > self.config.NOTE_MAX)).sum()
        )

        # Determine overall validity
        report['is_valid'] = all(
            report[key] == 0
            for key in ['missing_pseudo', 'missing_note', 'out_of_range', 'duplicates', 'invalid_types']
        )

        return report

    def process_data(self, data):
        """
        Traite et transforme les données brutes.

        Args:
            data (DataFrame): Les données à traiter.

        Returns:
            DataFrame: Les données traitées.
        """
        return self.clean_logs(data)
