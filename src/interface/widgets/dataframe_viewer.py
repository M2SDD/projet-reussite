#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/04/2026
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Widget Treeview réutilisable pour afficher un aperçu d'un DataFrame pandas.

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
import pandas as pd


# ----------------------------------------------------------------------------------------------------------------------
# Classe DataFrameViewer
# ----------------------------------------------------------------------------------------------------------------------
class DataFrameViewer(ttk.Frame):
    """
    Widget encapsulant un ttk.Treeview avec scrollbars horizontale et verticale
    pour afficher les premières lignes d'un DataFrame pandas.
    """

    def __init__(self, parent, max_rows: int = 200):
        super().__init__(parent)
        self.max_rows = max_rows
        self._build()

    def _build(self) -> None:
        self.tree = ttk.Treeview(self, show='headings', selectmode='browse')

        vsb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Charge et affiche un DataFrame (tronqué à max_rows lignes)."""
        self.clear()

        cols = list(df.columns)
        self.tree['columns'] = cols

        for col in cols:
            self.tree.heading(col, text=col)
            # Largeur adaptée à la longueur du nom de colonne
            width = max(80, min(200, len(str(col)) * 10))
            self.tree.column(col, width=width, minwidth=60, stretch=True)

        display = df.head(self.max_rows)
        for _, row in display.iterrows():
            values = [str(v) for v in row]
            self.tree.insert('', tk.END, values=values)

    def clear(self) -> None:
        """Vide le Treeview."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree['columns'] = []
