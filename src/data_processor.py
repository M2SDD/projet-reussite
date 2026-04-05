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
        self._cleaning_report = {
            'logs_initial_rows': 0,
            'logs_duplicates_removed': 0,
            'logs_missing_values': 0,
            'notes_initial_rows': 0,
            'notes_duplicates_removed': 0,
            'notes_invalid_pseudo': 0,
            'notes_clipped': 0,
            'students_logs_only': 0,
            'students_notes_only': 0,
            'students_merged': 0,
        }

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

    def extract_temporal_features(self, df):
        """
        Extrait des caractéristiques temporelles à partir de la colonne 'heure'.

        Crée les colonnes : heure_hour, day_of_week, month, is_weekend.

        Args:
            df (pd.DataFrame): Le DataFrame de logs avec une colonne 'heure' datetime.

        Returns:
            pd.DataFrame: Le DataFrame enrichi des caractéristiques temporelles.
        """
        df = df.copy()
        df['heure'] = pd.to_datetime(df['heure'], errors='coerce')
        df['heure_hour'] = df['heure'].dt.hour
        df['day_of_week'] = df['heure'].dt.dayofweek
        df['month'] = df['heure'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def compute_activity_metrics(self, df):
        """
        Calcule des métriques d'activité par étudiant.

        Métriques calculées : total_actions, unique_days_active, actions_per_day, session_count.

        Args:
            df (pd.DataFrame): Le DataFrame de logs avec colonnes 'pseudo' et 'heure'.

        Returns:
            pd.DataFrame: DataFrame avec une ligne par étudiant et les métriques d'activité.
        """
        df = df.copy()
        df['heure'] = pd.to_datetime(df['heure'], errors='coerce')
        df['date'] = df['heure'].dt.date

        grouped = df.groupby('pseudo')

        metrics = pd.DataFrame({
            'total_actions': grouped.size(),
            'unique_days_active': grouped['date'].nunique(),
        })

        metrics['actions_per_day'] = (
            metrics['total_actions'] / metrics['unique_days_active']
        ).round(2)

        # Session count: count distinct periods separated by > 30 min gap
        def count_sessions(group):
            times = group['heure'].dropna().sort_values()
            if len(times) <= 1:
                return 1
            gaps = times.diff() > pd.Timedelta(minutes=30)
            return int(gaps.sum()) + 1

        metrics['session_count'] = grouped.apply(count_sessions)

        return metrics.reset_index()

    def compute_component_features(self, df):
        """
        Crée des colonnes comptant les actions par composant pour chaque étudiant.

        Args:
            df (pd.DataFrame): Le DataFrame de logs avec colonnes 'pseudo' et 'composant'.

        Returns:
            pd.DataFrame: DataFrame avec une ligne par étudiant et une colonne par composant.
        """
        pivot = df.groupby(['pseudo', 'composant']).size().unstack(fill_value=0)
        pivot.columns = [f"comp_{col}" for col in pivot.columns]
        return pivot.reset_index()

    def merge_logs_notes(self, logs_df, notes_df):
        """
        Fusionne les métriques d'activité par étudiant avec les notes.

        Effectue une jointure externe (outer merge) sur 'pseudo' pour conserver
        tous les étudiants, y compris ceux présents uniquement dans les logs
        ou uniquement dans les notes.

        Args:
            logs_df (pd.DataFrame): DataFrame de métriques d'activité par étudiant
                (sortie de compute_activity_metrics ou compute_component_features).
            notes_df (pd.DataFrame): DataFrame de notes nettoyé avec colonnes 'pseudo' et 'note'.

        Returns:
            pd.DataFrame: DataFrame fusionné avec une ligne par étudiant.
        """
        # Compute activity metrics and component features
        activity = self.compute_activity_metrics(logs_df)
        components = self.compute_component_features(logs_df)

        # Merge activity and component features
        student_features = activity.merge(components, on='pseudo', how='outer')

        # Merge with notes
        logs_students = set(student_features['pseudo'].unique())
        notes_students = set(notes_df['pseudo'].unique())

        self._cleaning_report['students_logs_only'] = len(logs_students - notes_students)
        self._cleaning_report['students_notes_only'] = len(notes_students - logs_students)

        merged = student_features.merge(notes_df[['pseudo', 'note']], on='pseudo', how='outer')
        self._cleaning_report['students_merged'] = len(merged)

        return merged.reset_index(drop=True)

    def build_student_dataset(self, logs_df, notes_df):
        """
        Orchestre la construction complète du dataset étudiant pour le ML.

        Appelle séquentiellement :
        1. clean_logs pour nettoyer les logs
        2. clean_notes pour nettoyer les notes
        3. extract_temporal_features pour enrichir les logs
        4. merge_logs_notes pour fusionner métriques et notes

        Args:
            logs_df (pd.DataFrame): DataFrame de logs brut.
            notes_df (pd.DataFrame): DataFrame de notes brut.

        Returns:
            pd.DataFrame: DataFrame final au niveau étudiant, prêt pour le ML.
        """
        # Track initial counts
        self._cleaning_report['logs_initial_rows'] = len(logs_df)
        self._cleaning_report['notes_initial_rows'] = len(notes_df)

        # Clean
        logs_clean = self.clean_logs(logs_df)
        notes_clean = self.clean_notes(notes_df)

        self._cleaning_report['logs_duplicates_removed'] = (
            len(logs_df) - len(logs_clean)
        )

        # Extract temporal features
        logs_enriched = self.extract_temporal_features(logs_clean)

        # Merge into student-level dataset
        result = self.merge_logs_notes(logs_enriched, notes_clean)

        return result

    def get_cleaning_report(self):
        """
        Retourne un rapport sur les opérations de nettoyage effectuées.

        Returns:
            dict: Statistiques de nettoyage incluant :
                - logs_initial_rows: nombre initial de lignes de logs
                - logs_duplicates_removed: nombre de doublons supprimés
                - notes_initial_rows: nombre initial de lignes de notes
                - students_logs_only: étudiants présents uniquement dans les logs
                - students_notes_only: étudiants présents uniquement dans les notes
                - students_merged: nombre total d'étudiants dans le dataset final
        """
        return dict(self._cleaning_report)

    def process_data(self, data):
        """
        Traite et transforme les données brutes.

        Args:
            data (DataFrame): Les données à traiter.

        Returns:
            DataFrame: Les données traitées.
        """
        return self.clean_logs(data)
