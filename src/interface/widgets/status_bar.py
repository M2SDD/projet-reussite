#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Widget barre de statut affichée en bas de la fenêtre principale.

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


# ----------------------------------------------------------------------------------------------------------------------
# Classe StatusBar
# ----------------------------------------------------------------------------------------------------------------------
class StatusBar(ttk.Frame):
    """
    Barre de statut affichée en bas de la fenêtre.
    Affiche un message texte et une progressbar indéterminée lors des opérations longues.
    """

    def __init__(self, parent):
        super().__init__(parent, relief=tk.SUNKEN, borderwidth=1)
        self._label = ttk.Label(self, text='Prêt', anchor=tk.W)
        self._label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=2)

        self._progress = ttk.Progressbar(self, mode='indeterminate', length=120)

    def set(self, message: str) -> None:
        """Met à jour le texte de la barre de statut."""
        self._label.configure(text=message)

    def set_loading(self, loading: bool) -> None:
        """Affiche ou cache la progressbar animée."""
        if loading:
            self._progress.pack(side=tk.RIGHT, padx=6, pady=2)
            self._progress.start(10)
        else:
            self._progress.stop()
            self._progress.pack_forget()
