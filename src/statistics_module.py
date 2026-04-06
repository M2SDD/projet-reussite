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

    def generate_report(self, df):
        """
        Génère un rapport texte complet des statistiques descriptives.

        Produit un rapport formaté en texte contenant :
        - Résumé des données (dimensions, types, mémoire)
        - Statistiques descriptives pour chaque colonne numérique
        - Caractéristiques de distribution (asymétrie, aplatissement)
        - Détection des valeurs aberrantes

        Args:
            df (pd.DataFrame): Le DataFrame à analyser.

        Returns:
            str: Rapport formaté en texte avec toutes les statistiques calculées.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RAPPORT STATISTIQUE DESCRIPTIF")
        lines.append("=" * 80)
        lines.append("")

        # Section 1: Résumé des données
        lines.append("-" * 80)
        lines.append("1. RÉSUMÉ DES DONNÉES")
        lines.append("-" * 80)

        data_summary = self.compute_data_summary(df)
        lines.append(f"Nombre de lignes      : {data_summary['total_rows']}")
        lines.append(f"Nombre de colonnes    : {data_summary['total_columns']}")
        lines.append(f"Utilisation mémoire   : {data_summary['memory_usage_mb']} MB")
        lines.append("")

        lines.append("Types de colonnes :")
        for dtype, count in data_summary['column_types'].items():
            lines.append(f"  - {dtype}: {count} colonne(s)")
        lines.append("")

        missing = data_summary['missing_values']
        lines.append(f"Valeurs manquantes    : {missing['total']} total")
        if missing['total'] > 0:
            lines.append("Détail par colonne :")
            for col, count in missing['by_column'].items():
                if count > 0:
                    lines.append(f"  - {col}: {count}")
        lines.append("")

        # Section plage de dates (si applicable)
        if 'date_range' in data_summary:
            lines.append("Plage de dates :")
            for col, date_info in data_summary['date_range'].items():
                lines.append(f"  - {col}:")
                lines.append(f"    Min  : {date_info['min']}")
                lines.append(f"    Max  : {date_info['max']}")
                lines.append(f"    Durée: {date_info['range_days']} jours")
            lines.append("")

        # Section 2: Statistiques descriptives
        lines.append("-" * 80)
        lines.append("2. STATISTIQUES DESCRIPTIVES")
        lines.append("-" * 80)

        summary_stats = self.compute_summary_statistics(df)

        if summary_stats:
            # Obtenir toutes les colonnes numériques
            numeric_cols = list(summary_stats['count'].keys())

            for col in numeric_cols:
                lines.append(f"\nColonne: {col}")
                lines.append(f"  Nombre d'observations : {summary_stats['count'][col]}")
                lines.append(f"  Moyenne               : {summary_stats['mean'][col]:.4f}")
                lines.append(f"  Médiane               : {summary_stats['median'][col]:.4f}")
                lines.append(f"  Écart-type            : {summary_stats['std'][col]:.4f}")
                lines.append(f"  Variance              : {summary_stats['variance'][col]:.4f}")
                lines.append(f"  Minimum               : {summary_stats['min'][col]:.4f}")
                lines.append(f"  Maximum               : {summary_stats['max'][col]:.4f}")
                lines.append(f"  Étendue               : {summary_stats['range'][col]:.4f}")
                lines.append(f"  Q1 (25%)              : {summary_stats['q25'][col]:.4f}")
                lines.append(f"  Q2 (50%)              : {summary_stats['q50'][col]:.4f}")
                lines.append(f"  Q3 (75%)              : {summary_stats['q75'][col]:.4f}")
        else:
            lines.append("\nAucune colonne numérique disponible pour les statistiques.")
        lines.append("")

        # Section 3: Caractéristiques de distribution
        lines.append("-" * 80)
        lines.append("3. CARACTÉRISTIQUES DE DISTRIBUTION")
        lines.append("-" * 80)

        distribution = self.characterize_distribution(df)

        if distribution:
            for col in distribution['skewness'].keys():
                lines.append(f"\nColonne: {col}")
                skew = distribution['skewness'][col]
                kurt = distribution['kurtosis'][col]
                lines.append(f"  Asymétrie (Skewness)  : {skew:.4f}")

                # Interprétation de l'asymétrie
                if abs(skew) < 0.5:
                    skew_interp = "Distribution symétrique"
                elif skew > 0:
                    skew_interp = "Distribution asymétrique à droite (queue à droite)"
                else:
                    skew_interp = "Distribution asymétrique à gauche (queue à gauche)"
                lines.append(f"    → {skew_interp}")

                lines.append(f"  Aplatissement (Kurtosis): {kurt:.4f}")

                # Interprétation du kurtosis
                if kurt > 0:
                    kurt_interp = "Distribution leptokurtique (plus pointue)"
                elif kurt < 0:
                    kurt_interp = "Distribution platykurtique (plus plate)"
                else:
                    kurt_interp = "Distribution mésokurtique (normale)"
                lines.append(f"    → {kurt_interp}")
        else:
            lines.append("\nAucune colonne numérique disponible pour l'analyse de distribution.")
        lines.append("")

        # Section 4: Détection des valeurs aberrantes
        lines.append("-" * 80)
        lines.append("4. DÉTECTION DES VALEURS ABERRANTES (Méthode IQR)")
        lines.append("-" * 80)

        outliers_df = self.detect_outliers(df)
        numeric_cols_outliers = outliers_df.select_dtypes(include=[bool]).columns

        if len(numeric_cols_outliers) > 0:
            lines.append("")
            for col in numeric_cols_outliers:
                outlier_count = outliers_df[col].sum()
                outlier_pct = (outlier_count / len(df) * 100) if len(df) > 0 else 0
                lines.append(f"Colonne: {col}")
                lines.append(f"  Nombre d'outliers     : {outlier_count}")
                lines.append(f"  Pourcentage           : {outlier_pct:.2f}%")
        else:
            lines.append("\nAucune colonne numérique disponible pour la détection d'outliers.")
        lines.append("")

        # Pied de page
        lines.append("=" * 80)
        lines.append("FIN DU RAPPORT")
        lines.append("=" * 80)

        return "\n".join(lines)
