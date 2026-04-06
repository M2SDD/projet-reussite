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

    def generate_summary(self, data):
        """
        Génère un résumé statistique complet des données.

        Args:
            data (pd.DataFrame ou pd.Series): Les données à analyser.

        Returns:
            dict: Dictionnaire contenant toutes les statistiques calculées.
        """
        pass
