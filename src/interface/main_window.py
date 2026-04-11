#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Fenêtre principale de l'application Projet Réussite.

Instancie le ttk.Notebook avec les quatre onglets, la StatusBar,
et injecte la référence du contrôleur dans chaque composant.

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
import tkinter as tk
from tkinter import ttk

from .app_controller import AppController
from .tabs.data_tab import DataTab
from .tabs.preprocessing_tab import PreprocessingTab
from .tabs.modeling_tab import ModelingTab
from .tabs.evaluation_tab import EvaluationTab
from .widgets.status_bar import StatusBar


# ----------------------------------------------------------------------------------------------------------------------
# Classe MainWindowv
# ----------------------------------------------------------------------------------------------------------------------
class MainWindow:
    """
    Fenêtre principale de l'application.

    Crée le Notebook à 4 onglets et gère leur activation progressive :
      - Onglet 0 (Données)        : actif dès l'ouverture
      - Onglet 1 (Prétraitement)  : activé après chargement des données
      - Onglet 2 (Modélisation)   : activé après construction du dataset
      - Onglet 3 (Évaluation)     : activé après l'entraînement d'un modèle
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Prédiction de la réussite sur ARCHE")
        self.root.geometry("1180x760")
        self.root.minsize(960, 620)

        self.controller = AppController()

        self._build_ui()

        # Injecter la StatusBar dans le contrôleur
        self.controller.status_bar = self.status_bar

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Notebook principal
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Instanciation des onglets (chacun reçoit le notebook, le contrôleur et cette fenêtre)
        self.data_tab          = DataTab(self.notebook, self.controller, self)
        self.preprocessing_tab = PreprocessingTab(self.notebook, self.controller, self)
        self.modeling_tab      = ModelingTab(self.notebook, self.controller, self)
        self.evaluation_tab    = EvaluationTab(self.notebook, self.controller, self)

        self.notebook.add(self.data_tab.frame,          text='  Données  ')
        self.notebook.add(self.preprocessing_tab.frame, text='  Prétraitement  ')
        self.notebook.add(self.modeling_tab.frame,      text='  Modélisation  ')
        self.notebook.add(self.evaluation_tab.frame,    text='  Évaluation  ')

        # Désactiver les onglets en attente
        self.notebook.tab(1, state='disabled')
        self.notebook.tab(2, state='disabled')
        self.notebook.tab(3, state='disabled')

        # Barre de statut (en bas, avant le pack du notebook)
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=(0, 4))

    # ------------------------------------------------------------------
    # API publique utilisée par les onglets
    # ------------------------------------------------------------------

    def unlock_tab(self, index: int) -> None:
        """Active un onglet précédemment désactivé."""
        self.notebook.tab(index, state='normal')

    def go_to_tab(self, index: int) -> None:
        """Navigue vers l'onglet d'index donné."""
        self.notebook.select(index)
