#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 06/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module de chargement des exports CSV depuis ARCHE.

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
from typing import List, Optional

from ..config import Config

# ----------------------------------------------------------------------------------------------------------------------
# Classe DataLoader
# ----------------------------------------------------------------------------------------------------------------------
class DataLoader:
    """Classe responsable du chargement et de la validation des données brutes."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialise le chargeur de données.

        Args:
            config (Config, optional): Instance de la classe Config contenant les paramètres de configuration.
                                       Si None, une instance par défaut sera créée.
        """
        self.config = config if config is not None else Config()

    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str], file_type: str) -> None:
        """
        Vérifie que toutes les colonnes requises sont présentes dans le DataFrame.

        Args:
            df (pd.DataFrame): Le DataFrame à valider.
            required_columns (list): Liste des noms de colonnes attendus.
            file_type (str): Le type de fichier (pour les messages d'erreur).

        Raises:
            ValueError: Si une ou plusieurs colonnes sont manquantes.
        """
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans le fichier {file_type} : {missing_columns}. "
                f"Colonnes attendues : {required_columns}. "
                f"Colonnes trouvées : {list(df.columns)}."
            )

        # Avertissement si le fichier contient des colonnes supplémentaires (non bloquant)
        extra_columns = set(df.columns) - set(required_columns)
        if extra_columns:
            warnings.warn(
                f"Le fichier {file_type} contient des colonnes inattendues qui seront ignorées : {extra_columns}",
                UserWarning
            )

    def _load_csv(self, file_path: str, required_columns: List[str], file_type: str) -> pd.DataFrame:
        """
        Méthode générique interne pour charger un fichier CSV, gérer les erreurs
        d'encodage et valider les colonnes.
        """
        # Vérifier si le fichier existe et gérer les problèmes d'encodage
        try:
            # Forcer l'UTF-8 pour bien gérer les accents français des exports ARCHE
            df = pd.read_csv(file_path, encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Fichier {file_type} introuvable : {file_path}. "
                f"Veuillez vérifier que le fichier existe et que le chemin est correct."
            )
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Erreur d'encodage dans {file_path}. "
                f"Le format UTF-8 était attendu mais des caractères incompatibles ont été trouvés. "
                f"Veuillez vérifier que le fichier a bien été exporté en UTF-8 depuis ARCHE."
            )

        # Validation de la structure du fichier
        self._validate_columns(df, required_columns, file_type)

        return df

    def load_logs(self, file_path: str) -> pd.DataFrame:
        """
        Charge et valide le fichier CSV des logs des étudiants (ARCHE).
        Convertit la colonne 'heure' en format datetime de manière stricte.

        Args:
            file_path (str): Chemin vers le fichier (ex: logs_info_25_pseudo.csv).

        Returns:
            pd.DataFrame: DataFrame contenant les données des logs brutes avec dates parsées.
        """
        df = self._load_csv(
            file_path=file_path,
            required_columns=self.config.LOGS_REQUIRED_COLUMNS,
            file_type="logs"
        )

        # Conversion explicite et rapide avec le format exact défini dans la configuration
        df['heure'] = pd.to_datetime(df['heure'], format=self.config.DATETIME_FORMAT, errors='coerce').astype('datetime64[ns]')

        # Vérifiez s'il existe des valeurs de date et d'heure non valides qui ont été converties en NaT
        invalid_count = df['heure'].isna().sum()
        if invalid_count > 0:
            warnings.warn(
                f"{invalid_count} valeurs non valides ont été détectées dans la colonne 'heure'. "
                f"Ces éléments ont été définis sur NaT (Not a Time).",
                UserWarning
            )

        return df

    def load_notes(self, file_path: str) -> pd.DataFrame:
        """
        Charge et valide le fichier CSV des notes finales des étudiants.

        Args:
            file_path (str): Chemin vers le fichier (ex: notes_info_25_pseudo.csv).

        Returns:
            pd.DataFrame: DataFrame contenant les données des notes.
        """
        return self._load_csv(
            file_path=file_path,
            required_columns=self.config.NOTES_REQUIRED_COLUMNS,
            file_type="notes"
        )

if __name__ == "__main__":
    loader = DataLoader()
    print("Chargement des logs...")
    logs = loader.load_logs("../../data/logs_info_25_pseudo.csv")
    print("Logs chargés :")
    print(logs.head())
    print("Type de colonnes pour 'heure' :", logs['heure'].dtype, " (datetime64[ns] attendu)")
    print()
    print("Chargement des notes...")
    notes = loader.load_notes("../../data/notes_info_25_pseudo.csv")
    print("Notes chargées :")
    print(notes.head())
