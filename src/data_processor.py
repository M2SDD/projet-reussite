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
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression

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

    def compute_event_type_features(self, df):
        """
        Calcule des métriques basées sur les types d'événements pour chaque étudiant.

        Catégorise les événements en types (consultation, soumission, forum, etc.)
        et compte le nombre d'événements de chaque type par étudiant.

        Args:
            df (pd.DataFrame): Le DataFrame de logs avec colonnes 'pseudo' et 'evenement'.

        Returns:
            pd.DataFrame: DataFrame avec une ligne par étudiant et une colonne par type d'événement.
        """
        df = df.copy()

        # Define event type mappings
        def categorize_event(event):
            """Catégorise un événement selon son type."""
            if pd.isna(event):
                return 'other'
            event_lower = event.lower()

            # View/consultation events
            if any(keyword in event_lower for keyword in ['consulté', 'vue', 'affich', 'viewed', 'view']):
                return 'view'
            # Submission events
            elif any(keyword in event_lower for keyword in ['déposé', 'soumis', 'submit', 'upload', 'envoyé']):
                return 'submission'
            # Forum events
            elif any(keyword in event_lower for keyword in ['forum', 'discussion', 'message', 'post']):
                return 'forum'
            # Quiz/test events
            elif any(keyword in event_lower for keyword in ['test', 'quiz', 'tentative', 'attempt']):
                return 'quiz'
            # Download events
            elif any(keyword in event_lower for keyword in ['télécharg', 'download']):
                return 'download'
            else:
                return 'other'

        # Apply categorization
        df['event_type'] = df['evenement'].apply(categorize_event)

        # Count events by type for each student
        pivot = df.groupby(['pseudo', 'event_type']).size().unstack(fill_value=0)

        # Rename columns with descriptive names
        pivot.columns = [f"{col}_count" for col in pivot.columns]

        return pivot.reset_index()

    def compute_consistency_features(self, df):
        """
        Calcule des métriques de régularité et de cohérence de l'engagement étudiant.

        Métriques calculées :
        - streak_days: nombre maximum de jours consécutifs d'activité
        - avg_gap_days: nombre moyen de jours entre deux sessions
        - std_gap_days: écart-type des jours entre sessions (mesure de régularité)
        - study_frequency: nombre moyen de jours actifs par semaine

        Args:
            df (pd.DataFrame): Le DataFrame de logs avec colonnes 'pseudo' et 'heure'.

        Returns:
            pd.DataFrame: DataFrame avec une ligne par étudiant et les métriques de cohérence.
        """
        df = df.copy()
        df['heure'] = pd.to_datetime(df['heure'], errors='coerce')
        df['date'] = df['heure'].dt.date

        # Get unique active dates per student
        student_dates = df.groupby('pseudo')['date'].apply(lambda x: sorted(x.unique())).reset_index()

        def calculate_streak_days(dates):
            """Calcule la plus longue série de jours consécutifs."""
            if len(dates) == 0:
                return 0
            if len(dates) == 1:
                return 1

            # Convert dates to pandas datetime for easier manipulation
            dates = pd.to_datetime(dates)
            max_streak = 1
            current_streak = 1

            for i in range(1, len(dates)):
                diff = (dates[i] - dates[i-1]).days
                if diff == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1

            return max_streak

        def calculate_gap_stats(dates):
            """Calcule les statistiques sur les écarts entre sessions."""
            if len(dates) <= 1:
                return 0.0, 0.0

            dates = pd.to_datetime(dates)
            gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            return round(sum(gaps) / len(gaps), 2), round(pd.Series(gaps).std(), 2)

        def calculate_study_frequency(dates):
            """Calcule le nombre moyen de jours actifs par semaine."""
            if len(dates) == 0:
                return 0.0

            dates = pd.to_datetime(dates)
            total_days = (dates.max() - dates.min()).days
            if total_days == 0:
                return len(dates)

            weeks = total_days / 7.0
            return round(len(dates) / max(weeks, 1), 2)

        # Calculate metrics for each student
        student_dates['streak_days'] = student_dates['date'].apply(calculate_streak_days)
        student_dates[['avg_gap_days', 'std_gap_days']] = student_dates['date'].apply(
            lambda x: pd.Series(calculate_gap_stats(x))
        )
        student_dates['study_frequency'] = student_dates['date'].apply(calculate_study_frequency)

        # Drop the intermediate date column
        student_dates = student_dates.drop(columns=['date'])

        return student_dates

    def compute_interaction_depth_features(self, df):
        """
        Calcule des métriques de profondeur et de diversité d'engagement étudiant.

        Métriques calculées :
        - component_diversity: nombre de composants uniques utilisés
        - context_diversity: nombre de contextes uniques explorés
        - avg_interactions_per_component: nombre moyen d'interactions par composant
        - component_switch_rate: taux de changement de composant (transitions / total interactions)

        Args:
            df (pd.DataFrame): Le DataFrame de logs avec colonnes 'pseudo', 'composant', 'contexte', 'heure'.

        Returns:
            pd.DataFrame: DataFrame avec une ligne par étudiant et les métriques de profondeur d'engagement.
        """
        df = df.copy()
        df['heure'] = pd.to_datetime(df['heure'], errors='coerce')

        grouped = df.groupby('pseudo')

        metrics = pd.DataFrame({
            'component_diversity': grouped['composant'].nunique(),
            'context_diversity': grouped['contexte'].nunique(),
        })

        # Calculate average interactions per component
        def calc_avg_interactions_per_component(group):
            if len(group) == 0:
                return 0.0
            comp_counts = group['composant'].value_counts()
            return round(comp_counts.mean(), 2)

        metrics['avg_interactions_per_component'] = grouped.apply(
            calc_avg_interactions_per_component
        )

        # Calculate component switch rate
        def calc_component_switch_rate(group):
            if len(group) <= 1:
                return 0.0
            # Sort by time to get sequential interactions
            sorted_group = group.sort_values('heure')
            # Count switches (when component changes)
            switches = (sorted_group['composant'] != sorted_group['composant'].shift()).sum() - 1
            return round(switches / len(group), 2)

        metrics['component_switch_rate'] = grouped.apply(calc_component_switch_rate)

        return metrics.reset_index()

    def compute_temporal_patterns(self, df):
        """
        Calcule des patterns temporels d'engagement pour chaque étudiant.

        Métriques calculées :
        - peak_hour: heure de la journée avec le plus d'activité
        - morning_activity: proportion d'activité le matin (6h-12h)
        - afternoon_activity: proportion d'activité l'après-midi (12h-18h)
        - evening_activity: proportion d'activité le soir (18h-24h)
        - night_activity: proportion d'activité la nuit (0h-6h)
        - weekend_activity_ratio: proportion d'activité pendant le week-end

        Args:
            df (pd.DataFrame): Le DataFrame de logs avec colonnes 'pseudo' et 'heure'.

        Returns:
            pd.DataFrame: DataFrame avec une ligne par étudiant et les patterns temporels.
        """
        df = df.copy()
        df['heure'] = pd.to_datetime(df['heure'], errors='coerce')
        df['hour'] = df['heure'].dt.hour
        df['day_of_week'] = df['heure'].dt.dayofweek

        grouped = df.groupby('pseudo')

        # Calculate peak hour (most active hour)
        def get_peak_hour(group):
            if len(group) == 0:
                return 0
            hour_counts = group['hour'].value_counts()
            return int(hour_counts.idxmax()) if len(hour_counts) > 0 else 0

        metrics = pd.DataFrame({
            'peak_hour': grouped.apply(get_peak_hour)
        })

        # Calculate time period activity ratios
        def calc_time_periods(group):
            if len(group) == 0:
                return pd.Series([0.0, 0.0, 0.0, 0.0])

            total = len(group)
            morning = ((group['hour'] >= 6) & (group['hour'] < 12)).sum()
            afternoon = ((group['hour'] >= 12) & (group['hour'] < 18)).sum()
            evening = ((group['hour'] >= 18) & (group['hour'] < 24)).sum()
            night = (group['hour'] < 6).sum()

            return pd.Series([
                round(morning / total, 2),
                round(afternoon / total, 2),
                round(evening / total, 2),
                round(night / total, 2)
            ])

        metrics[['morning_activity', 'afternoon_activity', 'evening_activity', 'night_activity']] = grouped.apply(
            calc_time_periods
        )

        # Calculate weekend activity ratio
        def calc_weekend_ratio(group):
            if len(group) == 0:
                return 0.0
            weekend_count = group['day_of_week'].isin([5, 6]).sum()
            return round(weekend_count / len(group), 2)

        metrics['weekend_activity_ratio'] = grouped.apply(calc_weekend_ratio)

        return metrics.reset_index()

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

    def build_engagement_features(self, logs_df):
        """
        Orchestre la construction complète des features d'engagement étudiant.

        Appelle séquentiellement toutes les méthodes de calcul de features :
        1. compute_activity_metrics pour les métriques d'activité de base
        2. compute_component_features pour les features par composant
        3. compute_event_type_features pour les features par type d'événement
        4. compute_consistency_features pour les métriques de régularité
        5. compute_interaction_depth_features pour la profondeur d'engagement
        6. compute_temporal_patterns pour les patterns temporels

        Args:
            logs_df (pd.DataFrame): DataFrame de logs avec colonnes 'pseudo', 'heure', 'composant', 'contexte', 'evenement'.

        Returns:
            pd.DataFrame: DataFrame avec une ligne par étudiant et toutes les features d'engagement.
        """
        # Compute all feature sets
        activity = self.compute_activity_metrics(logs_df)
        components = self.compute_component_features(logs_df)
        event_types = self.compute_event_type_features(logs_df)
        consistency = self.compute_consistency_features(logs_df)
        interaction_depth = self.compute_interaction_depth_features(logs_df)
        temporal = self.compute_temporal_patterns(logs_df)

        # Merge all features on 'pseudo'
        result = activity
        result = result.merge(components, on='pseudo', how='outer')
        result = result.merge(event_types, on='pseudo', how='outer')
        result = result.merge(consistency, on='pseudo', how='outer')
        result = result.merge(interaction_depth, on='pseudo', how='outer')
        result = result.merge(temporal, on='pseudo', how='outer')

        return result.reset_index(drop=True)

    def build_student_dataset(self, logs_df, notes_df):
        """
        Orchestre la construction complète du dataset étudiant pour le ML.

        Appelle séquentiellement :
        1. clean_logs pour nettoyer les logs
        2. clean_notes pour nettoyer les notes
        3. extract_temporal_features pour enrichir les logs
        4. build_engagement_features pour calculer les features d'engagement
        5. Merge avec les notes

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

        # Build ALL engagement features (includes activity + component features)
        engagement_features = self.build_engagement_features(logs_enriched)

        # Merge directly with notes (don't call merge_logs_notes)
        result = engagement_features.merge(
            notes_clean[['pseudo', 'note']],
            on='pseudo',
            how='outer'
        )

        # Update tracking for students
        logs_students = set(engagement_features['pseudo'].unique())
        notes_students = set(notes_clean['pseudo'].unique())
        self._cleaning_report['students_logs_only'] = len(logs_students - notes_students)
        self._cleaning_report['students_notes_only'] = len(notes_students - logs_students)
        self._cleaning_report['students_merged'] = len(result)

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

    def compute_feature_correlations(self, df, target_column):
        """
        Calcule les corrélations entre toutes les features numériques et une colonne cible.

        Utilise la corrélation de Pearson pour mesurer les relations linéaires
        entre chaque feature et la variable cible.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features et la cible.
            target_column (str): Le nom de la colonne cible.

        Returns:
            pd.Series: Série contenant les corrélations de chaque feature avec la cible.
                Les features sont triées par valeur absolue de corrélation décroissante.
                L'index contient les noms des features (excluant la cible elle-même).

        Raises:
            ValueError: Si la colonne cible n'existe pas dans le DataFrame.
            ValueError: Si le DataFrame ne contient pas de colonnes numériques.
        """
        if target_column not in df.columns:
            raise ValueError(
                f"La colonne cible '{target_column}' n'existe pas dans le DataFrame."
            )

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        if len(numeric_df.columns) == 0:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique.")

        if target_column not in numeric_df.columns:
            raise ValueError(
                f"La colonne cible '{target_column}' n'est pas numérique."
            )

        # Compute correlations with target
        correlations = numeric_df.corr()[target_column]

        # Remove the target column from results
        correlations = correlations.drop(target_column)

        # Sort by absolute correlation value (descending)
        correlations = correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        )

        return correlations

    def compute_feature_feature_correlations(self, df, exclude_columns=None):
        """
        Calcule la matrice de corrélation complète entre toutes les features numériques.

        Utilise la corrélation de Pearson pour mesurer les relations linéaires
        entre toutes les paires de features. Utile pour détecter la multicolinéarité.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features.
            exclude_columns (list): Liste optionnelle de colonnes à exclure de l'analyse
                (par exemple, ['pseudo', 'note']).

        Returns:
            pd.DataFrame: Matrice de corrélation (n_features x n_features).
                Les valeurs sont comprises entre -1 (corrélation négative parfaite)
                et +1 (corrélation positive parfaite).

        Raises:
            ValueError: Si le DataFrame ne contient pas de colonnes numériques.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        if len(numeric_df.columns) == 0:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique.")

        # Exclude specified columns if provided
        if exclude_columns is not None:
            exclude_columns = [col for col in exclude_columns if col in numeric_df.columns]
            numeric_df = numeric_df.drop(columns=exclude_columns)

        if len(numeric_df.columns) == 0:
            raise ValueError(
                "Aucune colonne numérique restante après exclusion."
            )

        # Compute full correlation matrix
        correlation_matrix = numeric_df.corr()

        return correlation_matrix

    def test_feature_significance(self, df, target_column):
        """
        Teste la significativité statistique des corrélations entre features et cible.

        Utilise le test de corrélation de Pearson pour calculer les p-values
        de chaque feature par rapport à la variable cible. Une p-value faible
        (typiquement < 0.05) indique une corrélation statistiquement significative.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features et la cible.
            target_column (str): Le nom de la colonne cible.

        Returns:
            pd.DataFrame: DataFrame avec colonnes :
                - 'feature': nom de la feature
                - 'correlation': coefficient de corrélation de Pearson
                - 'p_value': p-value du test de significativité
                - 'is_significant': True si p_value < 0.05
                Trié par p-value croissante (features les plus significatives en premier).

        Raises:
            ValueError: Si la colonne cible n'existe pas dans le DataFrame.
            ValueError: Si le DataFrame ne contient pas de colonnes numériques.
            ValueError: Si la colonne cible n'est pas numérique.
        """
        if target_column not in df.columns:
            raise ValueError(
                f"La colonne cible '{target_column}' n'existe pas dans le DataFrame."
            )

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        if len(numeric_df.columns) == 0:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique.")

        if target_column not in numeric_df.columns:
            raise ValueError(
                f"La colonne cible '{target_column}' n'est pas numérique."
            )

        # Get target values
        target = numeric_df[target_column].dropna()

        # Test each feature
        results = []
        for col in numeric_df.columns:
            if col == target_column:
                continue

            # Get feature values aligned with target
            feature = numeric_df[col]

            # Align feature and target (drop NaN in either)
            valid_indices = target.index.intersection(feature.dropna().index)
            if len(valid_indices) < 3:
                # Need at least 3 points for meaningful correlation
                warnings.warn(
                    f"Feature '{col}' a moins de 3 valeurs valides, test ignoré.",
                    UserWarning,
                )
                continue

            aligned_feature = feature.loc[valid_indices]
            aligned_target = target.loc[valid_indices]

            # Compute Pearson correlation and p-value
            try:
                correlation, p_value = stats.pearsonr(aligned_feature, aligned_target)
                results.append({
                    'feature': col,
                    'correlation': round(correlation, 4),
                    'p_value': round(p_value, 6),
                    'is_significant': p_value < 0.05
                })
            except Exception as e:
                warnings.warn(
                    f"Erreur lors du test de '{col}': {str(e)}",
                    UserWarning,
                )
                continue

        # Create result DataFrame
        result_df = pd.DataFrame(results)

        if len(result_df) == 0:
            warnings.warn(
                "Aucune feature valide pour le test de significativité.",
                UserWarning,
            )
            return pd.DataFrame(columns=['feature', 'correlation', 'p_value', 'is_significant'])

        # Sort by p-value (most significant first)
        result_df = result_df.sort_values('p_value').reset_index(drop=True)

        return result_df

    def compute_descriptive_statistics(self, df):
        """
        Calcule les statistiques descriptives pour toutes les features numériques.

        Calcule les statistiques suivantes pour chaque colonne numérique :
        - count: nombre de valeurs non manquantes
        - mean: moyenne
        - std: écart-type
        - min: valeur minimale
        - 25%: premier quartile
        - 50%: médiane
        - 75%: troisième quartile
        - max: valeur maximale

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features à analyser.

        Returns:
            pd.DataFrame: DataFrame avec les features en lignes et les statistiques en colonnes.
                Chaque ligne représente une feature, chaque colonne une statistique.

        Raises:
            ValueError: Si le DataFrame ne contient pas de colonnes numériques.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        if len(numeric_df.columns) == 0:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique.")

        # Compute descriptive statistics
        stats = numeric_df.describe()

        # Transpose to have features as rows and statistics as columns
        stats_transposed = stats.T

        return stats_transposed

    def select_by_variance(self, df, threshold=0.0):
        """
        Sélectionne les features dont la variance dépasse un seuil donné.

        Cette méthode de sélection de features élimine les colonnes avec une variance
        faible, qui contiennent peu d'information utile pour la prédiction.
        Une variance faible indique que les valeurs d'une feature varient peu,
        ce qui la rend moins discriminante.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features à filtrer.
            threshold (float): Le seuil minimal de variance. Les colonnes avec une variance
                strictement inférieure à ce seuil seront éliminées. Par défaut 0.0
                (élimine uniquement les features constantes).

        Returns:
            pd.DataFrame: DataFrame contenant uniquement les colonnes dont la variance
                est supérieure ou égale au seuil. Les colonnes non numériques sont
                préservées dans le résultat.

        Raises:
            ValueError: Si le DataFrame ne contient pas de colonnes numériques.
        """
        # Select numeric and non-numeric columns separately
        numeric_df = df.select_dtypes(include=['number'])
        non_numeric_df = df.select_dtypes(exclude=['number'])

        if len(numeric_df.columns) == 0:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique.")

        # Compute variance for each numeric column
        variances = numeric_df.var()

        # Select columns with variance >= threshold
        selected_columns = variances[variances >= threshold].index.tolist()
        removed_count = len(numeric_df.columns) - len(selected_columns)

        if removed_count > 0:
            removed_cols = [col for col in numeric_df.columns if col not in selected_columns]
            warnings.warn(
                f"{removed_count} colonnes avec variance < {threshold} ont été supprimées: {removed_cols}",
                UserWarning,
            )

        # Filter numeric columns and combine with non-numeric columns
        filtered_numeric = numeric_df[selected_columns]

        # Combine non-numeric columns with filtered numeric columns
        if len(non_numeric_df.columns) > 0:
            result = pd.concat([non_numeric_df, filtered_numeric], axis=1)
        else:
            result = filtered_numeric

        return result

    def select_by_correlation(self, df, threshold=0.95):
        """
        Sélectionne les features en éliminant celles fortement corrélées entre elles.

        Cette méthode de sélection de features identifie et supprime les features
        redondantes en calculant la matrice de corrélation. Lorsque deux features
        ont une corrélation (en valeur absolue) supérieure au seuil, l'une d'elles
        est supprimée pour réduire la multicolinéarité.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features à filtrer.
            threshold (float): Le seuil de corrélation (en valeur absolue). Les paires
                de features avec une corrélation |r| > threshold seront réduites en
                ne gardant qu'une seule feature. Par défaut 0.95.

        Returns:
            pd.DataFrame: DataFrame contenant uniquement les features sélectionnées.
                Les colonnes non numériques sont préservées dans le résultat.

        Raises:
            ValueError: Si le DataFrame ne contient pas de colonnes numériques.
        """
        # Select numeric and non-numeric columns separately
        numeric_df = df.select_dtypes(include=['number'])
        non_numeric_df = df.select_dtypes(exclude=['number'])

        if len(numeric_df.columns) == 0:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique.")

        # Compute correlation matrix
        corr_matrix = numeric_df.corr().abs()

        # Find features with correlation greater than threshold
        to_drop = set()
        columns = corr_matrix.columns

        # Iterate through the correlation matrix
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    # Add the second column to the drop list
                    to_drop.add(columns[j])

        # Keep only columns not in to_drop
        selected_columns = [col for col in numeric_df.columns if col not in to_drop]
        removed_count = len(to_drop)

        if removed_count > 0:
            removed_cols = list(to_drop)
            warnings.warn(
                f"{removed_count} colonnes fortement corrélées (|r| > {threshold}) ont été supprimées: {removed_cols}",
                UserWarning,
            )

        # Filter numeric columns and combine with non-numeric columns
        filtered_numeric = numeric_df[selected_columns]

        # Combine non-numeric columns with filtered numeric columns
        if len(non_numeric_df.columns) > 0:
            result = pd.concat([non_numeric_df, filtered_numeric], axis=1)
        else:
            result = filtered_numeric

        return result

    def select_k_best(self, df, target, k=10, score_func=f_regression):
        """
        Sélectionne les k meilleures features basées sur des tests statistiques univariés.

        Cette méthode de sélection de features utilise SelectKBest de scikit-learn
        pour évaluer chaque feature individuellement par rapport à la cible (target)
        et sélectionner les k features ayant les scores les plus élevés.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features et la cible.
            target (str): Le nom de la colonne cible à prédire.
            k (int): Le nombre de features à sélectionner. Par défaut 10.
            score_func (callable): La fonction de score à utiliser pour l'évaluation.
                Par défaut f_regression pour les problèmes de régression.
                Utilisez f_classif pour les problèmes de classification.

        Returns:
            pd.DataFrame: DataFrame contenant uniquement les k meilleures features.

        Raises:
            ValueError: Si la colonne cible n'existe pas dans le DataFrame.
            ValueError: Si le DataFrame ne contient pas assez de colonnes numériques.
            ValueError: Si k est supérieur au nombre de features disponibles.
        """
        if target not in df.columns:
            raise ValueError(f"La colonne cible '{target}' n'existe pas dans le DataFrame.")

        # Separate target from features
        y = df[target]
        X = df.drop(columns=[target])

        # Select only numeric columns for feature selection
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) == 0:
            raise ValueError("Le DataFrame ne contient aucune colonne numérique pour la sélection de features.")

        if k > len(numeric_cols):
            raise ValueError(
                f"k ({k}) est supérieur au nombre de features disponibles ({len(numeric_cols)})."
            )

        # Extract numeric features
        X_numeric = X[numeric_cols]

        # Apply SelectKBest
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X_numeric, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [col for col, selected in zip(numeric_cols, selected_mask) if selected]

        removed_count = len(numeric_cols) - len(selected_features)
        if removed_count > 0:
            removed_cols = [col for col in numeric_cols if col not in selected_features]
            warnings.warn(
                f"{removed_count} colonnes ont été exclues par SelectKBest: {removed_cols}",
                UserWarning,
            )

        # Return DataFrame with selected features
        return X[selected_features]

    def select_features_pipeline(self, df, target, variance_threshold=0.0, correlation_threshold=0.95, k=10, score_func=f_regression):
        """
        Pipeline complet de sélection de features combinant plusieurs méthodes.

        Cette méthode applique séquentiellement plusieurs techniques de sélection de features
        pour réduire la dimensionnalité et améliorer la qualité du jeu de données :
        1. Filtrage par variance : élimine les features avec variance faible
        2. Filtrage par corrélation : élimine les features redondantes
        3. Sélection K-Best : sélectionne les k meilleures features statistiquement

        Args:
            df (pd.DataFrame): Le DataFrame contenant les features et la cible.
            target (str): Le nom de la colonne cible à prédire.
            variance_threshold (float): Seuil minimal de variance. Par défaut 0.0.
            correlation_threshold (float): Seuil de corrélation entre features. Par défaut 0.95.
            k (int): Nombre de features à sélectionner finalement. Par défaut 10.
            score_func (callable): Fonction de score pour SelectKBest. Par défaut f_regression.

        Returns:
            pd.DataFrame: DataFrame contenant uniquement les features sélectionnées après
                application du pipeline complet.

        Raises:
            ValueError: Si la colonne cible n'existe pas dans le DataFrame.
            ValueError: Si le DataFrame ne contient pas de colonnes numériques.

        Example:
            >>> df = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [1, 1, 1],
            ...                    'feat3': [4, 5, 6], 'target': [10, 20, 30]})
            >>> processor = DataProcessor()
            >>> selected = processor.select_features_pipeline(df, 'target', k=2)
        """
        if target not in df.columns:
            raise ValueError(f"La colonne cible '{target}' n'existe pas dans le DataFrame.")

        # Separate target from features
        y = df[target]
        X = df.drop(columns=[target])

        initial_features = len(X.select_dtypes(include=['number']).columns)

        # Step 1: Remove low variance features
        X_variance = self.select_by_variance(X, threshold=variance_threshold)
        features_after_variance = len(X_variance.select_dtypes(include=['number']).columns)

        # Step 2: Remove highly correlated features
        X_correlation = self.select_by_correlation(X_variance, threshold=correlation_threshold)
        features_after_correlation = len(X_correlation.select_dtypes(include=['number']).columns)

        # Step 3: Select k best features
        # Re-add target column for select_k_best
        X_with_target = X_correlation.copy()
        X_with_target[target] = y

        # Adjust k if necessary
        numeric_features = len(X_correlation.select_dtypes(include=['number']).columns)
        k_adjusted = min(k, numeric_features)

        if k_adjusted < k:
            warnings.warn(
                f"k ajusté de {k} à {k_adjusted} car seulement {numeric_features} features disponibles après filtrage.",
                UserWarning,
            )

        X_final = self.select_k_best(X_with_target, target, k=k_adjusted, score_func=score_func)

        # Pipeline summary
        warnings.warn(
            f"Pipeline de sélection terminé: {initial_features} features → "
            f"{features_after_variance} (variance) → "
            f"{features_after_correlation} (corrélation) → "
            f"{len(X_final.columns)} (k-best)",
            UserWarning,
        )

        return X_final

    def process_data(self, data):
        """
        Traite et transforme les données brutes.

        Args:
            data (DataFrame): Les données à traiter.

        Returns:
            DataFrame: Les données traitées.
        """
        return self.clean_logs(data)
