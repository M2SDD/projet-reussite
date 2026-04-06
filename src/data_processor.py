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

    def process_data(self, data):
        """
        Traite et transforme les données brutes.

        Args:
            data (DataFrame): Les données à traiter.

        Returns:
            DataFrame: Les données traitées.
        """
        return self.clean_logs(data)
