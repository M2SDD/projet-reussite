#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 09/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Sous-package pour la gestion des modèles de machine learning.

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
from .base_regressor import BaseRegressor
from .ensemble_regressor import EnsembleRegressor
from .linear_regressor import LinearRegressor

__all__ = [
    'BaseRegressor',
    'EnsembleRegressor',
    'LinearRegressor'
]
