#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 08/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Module de modèle ML générique pour la prédiction des notes des étudiants

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
from .config import Config


# ----------------------------------------------------------------------------------------------------------------------
# Classe MLModel
# ----------------------------------------------------------------------------------------------------------------------
class MLModel:
    """
    Classe responsable de l'entraînement et de l'évaluation d'un modèle de machine learning.

    Cette classe fournit une interface générique pour entraîner différents types de modèles
    de machine learning sur les indicateurs d'engagement des étudiants afin de prédire
    leurs notes finales. Elle inclut des méthodes pour l'entraînement, la prédiction,
    et l'évaluation du modèle.
    """

    def __init__(self, config=None):
        """
        Initialise le modèle de machine learning.

        Args:
            config (Config): Instance de configuration. Si None, utilise la configuration par défaut.
        """
        self.config = config if config is not None else Config()
        self.model = None
