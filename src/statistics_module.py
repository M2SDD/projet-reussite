#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 06/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module de calcul des statistiques descriptives

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

from .config import Config


# ----------------------------------------------------------------------------------------------------------------------
# Classe StatisticsModule
# ----------------------------------------------------------------------------------------------------------------------
class StatisticsModule:
    """
    Classe responsable du calcul des statistiques descriptives sur les données.

    Cette classe fournit des méthodes pour calculer des statistiques
    de base (moyenne, médiane, écart-type, etc.) ainsi que des statistiques
    avancées sur les données de logs et de notes.
    """

    def __init__(self, config=None):
        """
        Initialise le module de statistiques.

        Args:
            config (Config): Instance de configuration. Si None, utilise la configuration par défaut.
        """
        self.config = config if config is not None else Config()

    def compute_data_summary(self, df):
        """
        Calcule les métadonnées et informations générales sur le dataset.

        Retourne des informations structurelles sur le DataFrame :
        - Dimensions (nombre de lignes, colonnes)
        - Types de données et leur répartition
        - Valeurs manquantes
        - Utilisation mémoire
        - Plage de dates (si colonnes datetime présentes)

        Args:
            df (pd.DataFrame): Le DataFrame à analyser.

        Returns:
            dict: Dictionnaire contenant les métadonnées du dataset.
                  Structure : {
                      'total_rows': int,
                      'total_columns': int,
                      'column_types': dict,
                      'missing_values': dict,
                      'memory_usage_mb': float,
                      'date_range': dict (optionnel)
                  }
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
        }

        # Analyser les types de données
        type_counts = df.dtypes.value_counts().to_dict()
        summary['column_types'] = {str(k): int(v) for k, v in type_counts.items()}

        # Analyser les valeurs manquantes
        missing_total = df.isna().sum().sum()
        missing_by_column = df.isna().sum().to_dict()
        summary['missing_values'] = {
            'total': int(missing_total),
            'by_column': missing_by_column,
        }

        # Calculer l'utilisation mémoire
        memory_bytes = df.memory_usage(deep=True).sum()
        summary['memory_usage_mb'] = round(memory_bytes / (1024 * 1024), 2)

        # Analyser les colonnes datetime pour obtenir la plage de dates
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            date_ranges = {}
            for col in datetime_cols:
                non_null_dates = df[col].dropna()
                if len(non_null_dates) > 0:
                    date_ranges[col] = {
                        'min': non_null_dates.min(),
                        'max': non_null_dates.max(),
                        'range_days': (non_null_dates.max() - non_null_dates.min()).days,
                    }
            if date_ranges:
                summary['date_range'] = date_ranges

        return summary

    def calculate_basic_stats(self, data):
        """
        Calcule les statistiques descriptives de base.

        Args:
            data (pd.Series ou pd.DataFrame): Les données à analyser.

        Returns:
            dict: Dictionnaire contenant les statistiques calculées.
        """
        pass

    def calculate_central_tendency(self, data):
        """
        Calcule les mesures de tendance centrale (moyenne, médiane, mode).

        Args:
            data (pd.Series): Les données à analyser.

        Returns:
            dict: Dictionnaire avec mean, median, mode.
        """
        pass

    def calculate_dispersion(self, data):
        """
        Calcule les mesures de dispersion (variance, écart-type, étendue).

        Args:
            data (pd.Series): Les données à analyser.

        Returns:
            dict: Dictionnaire avec variance, std, range, etc.
        """
        pass

    def calculate_distribution(self, data):
        """
        Calcule les mesures de distribution (asymétrie, aplatissement).

        Args:
            data (pd.Series): Les données à analyser.

        Returns:
            dict: Dictionnaire avec skewness, kurtosis.
        """
        pass

    def calculate_quantiles(self, data, quantiles=None):
        """
        Calcule les quantiles des données.

        Args:
            data (pd.Series): Les données à analyser.
            quantiles (list): Liste des quantiles à calculer (par défaut [0.25, 0.5, 0.75]).

        Returns:
            dict: Dictionnaire avec les quantiles calculés.
        """
        pass

    def compute_summary_statistics(self, df):
        """
        Calcule les statistiques descriptives pour toutes les colonnes numériques.

        Calcule pour chaque colonne numérique :
        - Mesures de tendance centrale (mean, median)
        - Mesures de dispersion (std, variance, min, max, range)
        - Quantiles (25%, 50%, 75%)
        - Nombre d'observations (count)

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données à analyser.

        Returns:
            dict: Dictionnaire avec les statistiques pour chaque colonne numérique.
                  Structure : {
                      'mean': {col1: val1, col2: val2, ...},
                      'median': {col1: val1, col2: val2, ...},
                      'std': {col1: val1, col2: val2, ...},
                      ...
                  }
        """
        # Sélectionner uniquement les colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            import warnings
            warnings.warn(
                "Aucune colonne numérique trouvée dans le DataFrame.",
                UserWarning,
            )
            return {}

        # Calculer les statistiques
        stats = {
            'count': numeric_df.count().to_dict(),
            'mean': numeric_df.mean().to_dict(),
            'median': numeric_df.median().to_dict(),
            'std': numeric_df.std().to_dict(),
            'variance': numeric_df.var().to_dict(),
            'min': numeric_df.min().to_dict(),
            'max': numeric_df.max().to_dict(),
            'range': (numeric_df.max() - numeric_df.min()).to_dict(),
            'q25': numeric_df.quantile(0.25).to_dict(),
            'q50': numeric_df.quantile(0.50).to_dict(),
            'q75': numeric_df.quantile(0.75).to_dict(),
        }

        return stats

    def detect_outliers(self, df):
        """
        Détecte les valeurs aberrantes (outliers) en utilisant la méthode IQR.

        La méthode IQR (Interquartile Range) identifie les outliers comme
        les valeurs qui se situent en dehors de l'intervalle [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données à analyser.

        Returns:
            pd.DataFrame: DataFrame de même forme que l'entrée avec des valeurs booléennes.
                         True indique un outlier, False indique une valeur normale.
                         Les colonnes non-numériques contiennent uniquement False.
        """
        import warnings

        # Créer un DataFrame de résultat avec la même forme, initialisé à False
        outliers_df = pd.DataFrame(False, index=df.index, columns=df.columns)

        # Sélectionner uniquement les colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            warnings.warn(
                "Aucune colonne numérique trouvée dans le DataFrame.",
                UserWarning,
            )
            return outliers_df

        # Pour chaque colonne numérique, détecter les outliers avec la méthode IQR
        for col in numeric_df.columns:
            # Calculer Q1, Q3 et IQR
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Définir les bornes
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identifier les outliers
            outliers_df[col] = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)

        return outliers_df

    def characterize_distribution(self, df):
        """
        Calcule les caractéristiques de distribution (asymétrie et aplatissement).

        Calcule pour chaque colonne numérique :
        - Skewness (asymétrie) : mesure de l'asymétrie de la distribution
        - Kurtosis (aplatissement) : mesure de l'aplatissement de la distribution

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données à analyser.

        Returns:
            dict: Dictionnaire avec skewness et kurtosis pour chaque colonne numérique.
                  Structure : {
                      'skewness': {col1: val1, col2: val2, ...},
                      'kurtosis': {col1: val1, col2: val2, ...}
                  }
        """
        import warnings

        # Sélectionner uniquement les colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            warnings.warn(
                "Aucune colonne numérique trouvée dans le DataFrame.",
                UserWarning,
            )
            return {}

        # Calculer skewness et kurtosis
        stats = {
            'skewness': numeric_df.skew().to_dict(),
            'kurtosis': numeric_df.kurtosis().to_dict(),
        }

        return stats

    def generate_summary(self, data):
        """
        Génère un résumé statistique complet des données.

        Args:
            data (pd.DataFrame ou pd.Series): Les données à analyser.

        Returns:
            dict: Dictionnaire contenant toutes les statistiques calculées.
        """
        pass
