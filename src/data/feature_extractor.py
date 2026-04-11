#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 10/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module d'extraction des variables (Feature Engineering) depuis les logs ARCHE.

Ce module transforme le fichier de logs brut (une ligne = un clic) en un
tableau récapitulatif par étudiant (une ligne = un étudiant, avec ses statistiques).

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
import numpy as np
from typing import Optional

from ..config import Config


# ----------------------------------------------------------------------------------------------------------------------
# Classe FeatureExtractor
# ----------------------------------------------------------------------------------------------------------------------
class FeatureExtractor:
    """
    Classe responsable de l'extraction des variables métiers à partir des logs bruts.
    Les noms des variables générées sont directement en français.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config if config is not None else Config()

    def compute_activity_metrics(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Calcule les métriques d'activité globale par étudiant."""
        df_metrics = pd.DataFrame(index=df_logs['pseudo'].unique())
        df_metrics.index.name = 'pseudo'
        grouped = df_logs.groupby('pseudo')

        # Volume total d'actions
        df_metrics['actions_totales'] = grouped.size()

        # Nombre de jours actifs
        df_metrics['jours_actifs'] = grouped['heure'].apply(
            lambda x: x.dt.date.nunique()
        )

        # Durée d'engagement (en jours)
        date_min = grouped['heure'].min()
        date_max = grouped['heure'].max()
        df_metrics['duree_engagement_jours'] = (date_max - date_min).dt.days

        # Session count: count distinct periods separated by > 30 min gap
        def count_sessions(group):
            times = group['heure'].dropna().sort_values()
            if len(times) <= 1:
                return 1
            gaps = times.diff() > pd.Timedelta(minutes=self.config.SESSION_GAP_MINUTES)
            return int(gaps.sum()) + 1

        df_metrics['nombre_sessions'] = grouped.apply(count_sessions, include_groups=False)

        return df_metrics

    def compute_temporal_patterns(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Extrait les habitudes de travail temporelles (semaine, week-end, heures)."""
        df_temp = pd.DataFrame(index=df_logs['pseudo'].unique())
        df_temp.index.name = 'pseudo'
        actions_totales = df_logs.groupby('pseudo').size()

        # Calculate l'heure de point (l'heure la plus active)
        def get_peak_hour(group):
            group = group.copy()
            group['hour'] = group['heure'].dt.hour
            if len(group) == 0:
                return 0
            hour_counts = group['hour'].value_counts()
            return int(hour_counts.idxmax()) if len(hour_counts) > 0 else 0

        df_temp['heure_pointe'] = df_logs.groupby('pseudo').apply(get_peak_hour, include_groups=False)

        # 1. Semaine vs Week-end (5 = Samedi, 6 = Dimanche)
        est_weekend = df_logs['heure'].dt.dayofweek >= 5
        df_temp['actions_weekend'] = df_logs[est_weekend].groupby('pseudo').size()
        df_temp['actions_semaine'] = df_logs[~est_weekend].groupby('pseudo').size()

        df_temp['ratio_activite_weekend'] = df_temp['actions_weekend'] / actions_totales.replace(0, 1)  # pour éviter division par 0

        # 2. Tranches horaires
        heure_jour = df_logs['heure'].dt.hour
        df_temp['actions_matin'] = df_logs[(heure_jour >= 6) & (heure_jour < 12)].groupby('pseudo').size()
        df_temp['actions_aprem'] = df_logs[(heure_jour >= 12) & (heure_jour < 18)].groupby('pseudo').size()
        df_temp['actions_soir'] = df_logs[(heure_jour >= 18) & (heure_jour < 24)].groupby('pseudo').size()
        df_temp['actions_nuit'] = df_logs[(heure_jour >= 0) & (heure_jour < 6)].groupby('pseudo').size()

        return df_temp.fillna(0)

    def compute_component_features(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Calcule l'engagement par composant ARCHE (Fichier, Forum, etc.)."""
        # Utilisation de crosstab pour compter les occurrences
        df_comp = pd.crosstab(df_logs['pseudo'], df_logs['composant'])

        # Ajout d'un préfixe pour bien identifier que ce sont des composants
        df_comp.columns = [f"comp_{col.lower().replace(' ', '_')}" for col in df_comp.columns]

        return df_comp

    def compute_event_type_features(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Calcule l'engagement par type d'événement (Consultation, Création, etc.)."""
        # Utilisation de crosstab pour compter les occurrences
        df_event = pd.crosstab(df_logs['pseudo'], df_logs['evenement'])

        # Ajout d'un préfixe pour l'identification
        df_event.columns = [f"evt_{col.lower().replace(' ', '_')}" for col in df_event.columns]

        return df_event

    def compute_consistency_features(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Calcule les métriques de régularité du travail."""
        df_consist = pd.DataFrame(index=df_logs['pseudo'].unique())
        df_consist.index.name = 'pseudo'

        # Extraire uniquement la date (variable locale, sans muter df_logs)
        date_col = df_logs['heure'].dt.date

        # Obtenir les dates uniques par étudiant et trier
        dates_par_etudiant = df_logs.assign(date=date_col)\
                                    .groupby('pseudo')['date']\
                                    .unique()\
                                    .apply(sorted)

        # Fonction pour calculer la plus longue série de jours consécutifs
        def calculate_streak_days(dates):
            if len(dates) == 0:
                return 0
            if len(dates) == 1:
                return 1

            max_streak = 1
            current_streak = 1

            dates = pd.to_datetime(pd.Series(dates))
            for i in range(1, len(dates)):
                diff = (dates[i] - dates[i-1]).days
                if diff == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1

            return max_streak

        # Fonction pour calculer l'écart moyen en jours
        def calculate_gap_stats(dates):
            if len(dates) <= 1:
                return 0, 0
            dates = pd.to_datetime(pd.Series(dates))
            gaps = dates.diff().dt.days.dropna()
            std_val = gaps.std() if len(gaps) > 1 else 0.0
            return gaps.mean(), std_val

        # Fonction pour calculer le nombre moyen de jours actifs par semaine
        def calculate_study_frequency(dates):
            if len(dates) == 0:
                return 0.0

            dates = pd.to_datetime(pd.Series(dates))
            total_days = (dates.max() - dates.min()).days
            if total_days == 0:
                study_frequency = len(dates)
            else:
                weeks = total_days / 7.0
                study_frequency = len(dates) / max(weeks, 1)
            return study_frequency

        df_consist['jours_consecutifs_max'] = dates_par_etudiant.apply(calculate_streak_days)
        df_consist[['ecart_moyen_jours', 'ecart_type_jours']] = dates_par_etudiant.apply(
            lambda x: pd.Series(calculate_gap_stats(x))
        )
        df_consist['jours_actifs_hebdos'] = dates_par_etudiant.apply(calculate_study_frequency)
        return df_consist

    def compute_interaction_depth_features(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Calcule la profondeur d'interaction (ratio d'actions spécifiques par composant)."""
        df_depth = pd.DataFrame(index=df_logs['pseudo'].unique())
        df_depth.index.name = 'pseudo'
        grouped = df_logs.groupby('pseudo')

        # Moyenne d'actions par jour de connexion
        actions_totales = grouped.size()
        jours_actifs = grouped['heure'].apply(lambda x: x.dt.date.nunique())
        df_depth['actions_par_jour'] = actions_totales / jours_actifs.replace(0, 1)  # Éviter la division par zéro

        # Calcul de la diversité d'interactions
        df_depth['diversite_composants'] = grouped['composant'].nunique()
        df_depth['diversite_contextes'] = grouped['contexte'].nunique()

        # Calcule les interactions moyennes par composant
        def calc_avg_interactions_per_component(group):
            if len(group) == 0:
                return 0.0
            comp_counts = group['composant'].value_counts()
            return comp_counts.mean()

        df_depth['interactions_moy_par_composant'] = grouped.apply(
            calc_avg_interactions_per_component, include_groups=False
        )

        # Calcule le taux de changement de composant
        def calc_component_switch_rate(group):
            if len(group) <= 1:
                return 0.0
            # Trier par heure pour obtenir des interactions chronologiques
            sorted_group = group.sort_values('heure')
            # Compte lorsqu'il change de composant
            switches = (sorted_group['composant'] != sorted_group['composant'].shift()).sum() - 1
            return switches / len(group)

        df_depth['taux_changement_composant'] = grouped.apply(calc_component_switch_rate, include_groups=False)

        return df_depth

    def extract_features(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestre l'extraction de toutes les variables et les fusionne.

        Args:
            df_logs (pd.DataFrame): Le DataFrame contenant les logs chargés par DataLoader.

        Returns:
            pd.DataFrame: DataFrame avec 'pseudo' en index et toutes les features calculées.
        """
        if df_logs.empty:
            raise ValueError("Le DataFrame des logs est vide.")

        if not pd.api.types.is_datetime64_any_dtype(df_logs['heure']):
            raise TypeError("La colonne 'heure' doit être de type datetime.")

        # Construction du DataFrame final en appelant chaque sous-méthode
        df_features = self.compute_activity_metrics(df_logs)

        # Jointure avec les autres ensembles de features
        feature_methods = [
            self.compute_temporal_patterns,
            self.compute_component_features,
            self.compute_event_type_features,
            self.compute_consistency_features,
            self.compute_interaction_depth_features
        ]

        for method in feature_methods:
            df_new_features = method(df_logs)
            df_features = df_features.join(df_new_features)

        # Remplacer les valeurs manquantes générées par les jointures
        df_features = df_features.fillna(0)

        return df_features