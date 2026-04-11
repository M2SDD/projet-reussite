#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/04/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Sous-package des onglets de l'interface.

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
from .data_tab import DataTab
from .preprocessing_tab import PreprocessingTab
from .modeling_tab import ModelingTab
from .evaluation_tab import EvaluationTab

__all__ = [
    'DataTab', 
    'PreprocessingTab', 
    'ModelingTab', 
    'EvaluationTab'
    ]
