#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 08/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Package principal pour l'analyse et la visualisation des données d'exports Arche anonymisés

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

# data
from .data.data_loader import DataLoader
from .data.feature_extractor import FeatureExtractor
from .data.data_cleaner import DataCleaner
from .data.dataset_builder import DatasetBuilder
from .data.statistics_module import StatisticsModule

from .data_processor import DataProcessor

# models
from .models.ensemble_regressor import EnsembleRegressor
from .models.linear_regressor import LinearRegressor

# evaluation
from .evaluation.model_evaluator import ModelEvaluator

# visualisation
from .visualization.model_visualizer import ModelVisualizer

__all__ = [
    'Config',
    'DataLoader',
    'FeatureExtractor',
    'DataCleaner',
    'DatasetBuilder',
    'DataProcessor',
    'EnsembleRegressor',
    'LinearRegressor',
    'ModelEvaluator',
    'ModelVisualizer',
    'StatisticsModule',
]
