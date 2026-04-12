#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Onglet 2 - Prétraitement et construction du dataset.

Expose les options de DatasetBuilder (suppression des inactifs, outliers,
méthodes de sélection de features, etc.) et affiche le dataset résultant.

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
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from ..app_controller import AppController
from ..widgets.dataframe_viewer import DataFrameViewer


# ----------------------------------------------------------------------------------------------------------------------
# Classe PreprocessingTab
# ----------------------------------------------------------------------------------------------------------------------
class PreprocessingTab:
    """Onglet de configuration du pipeline et de construction du dataset."""

    def __init__(self, parent, controller: AppController, main_window):
        self.controller  = controller
        self.main_window = main_window

        self.frame = ttk.Frame(parent)
        self._build()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # --- Section options ---
        opt_frame = ttk.LabelFrame(self.frame, text=' Options du pipeline ', padding=12)
        opt_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        # Ligne 0 – options booléennes globales
        self._drop_inactive   = tk.BooleanVar(value=True)
        self._remove_outliers = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Supprimer les étudiants inactifs',
                        variable=self._drop_inactive).grid(row=0, column=0, sticky='w', padx=8, pady=4)
        ttk.Checkbutton(opt_frame, text='Supprimer les outliers (méthode IQR)',
                        variable=self._remove_outliers).grid(row=0, column=1, sticky='w', padx=8, pady=4)

        # Ligne 1 – méthodes de sélection de features
        ttk.Label(opt_frame, text='Méthodes de sélection :').grid(
            row=1, column=0, sticky='w', padx=8, pady=(8, 2))

        methods_frame = ttk.Frame(opt_frame)
        methods_frame.grid(row=1, column=1, sticky='w', pady=(8, 2))

        self._method_linear     = tk.BooleanVar(value=True)
        self._method_mutual     = tk.BooleanVar(value=False)
        self._method_importance = tk.BooleanVar(value=False)
        self._method_rfe        = tk.BooleanVar(value=False)

        ttk.Checkbutton(methods_frame, text='Corrélation linéaire (Test F)',
                        variable=self._method_linear).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(methods_frame, text='Information Mutuelle',
                        variable=self._method_mutual).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(methods_frame, text='Feature Importance',
                        variable=self._method_importance).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(methods_frame, text='Recursive Feature Elimination (lent)',
                        variable=self._method_rfe).pack(side=tk.LEFT, padx=4)

        # Ligne 2 – k features + pré-filtres
        ttk.Label(opt_frame, text='K features (par méthode) :').grid(
            row=2, column=0, sticky='w', padx=8, pady=(4, 2))

        k_row = ttk.Frame(opt_frame)
        k_row.grid(row=2, column=1, sticky='w', pady=(4, 2))

        self._k_features      = tk.IntVar(value=15)
        self._prefilter_var   = tk.BooleanVar(value=True)
        self._prefilter_corr  = tk.BooleanVar(value=True)

        ttk.Spinbox(k_row, from_=1, to=100, textvariable=self._k_features,
                    width=6).pack(side=tk.LEFT)
        ttk.Checkbutton(k_row, text='Pré-filtre variance',
                        variable=self._prefilter_var).pack(side=tk.LEFT, padx=16)
        ttk.Checkbutton(k_row, text='Pré-filtre corrélation',
                        variable=self._prefilter_corr).pack(side=tk.LEFT, padx=4)

        opt_frame.columnconfigure(1, weight=1)

        # Bouton principal
        btn_row = ttk.Frame(opt_frame)
        btn_row.grid(row=3, column=0, columnspan=2, pady=(12, 0))
        self._build_btn = ttk.Button(btn_row, text='Construire le dataset',
                                     command=self._on_build)
        self._build_btn.pack()

        # --- Section résultat ---
        result_frame = ttk.LabelFrame(self.frame, text=' Dataset résultant ', padding=6)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self._shape_label    = ttk.Label(result_frame, text='')
        self._shape_label.pack(anchor='w', padx=6, pady=(2, 0))

        self._features_label = ttk.Label(result_frame, text='', foreground='gray',
                                         wraplength=1050, justify=tk.LEFT)
        self._features_label.pack(anchor='w', padx=6, pady=(0, 4))

        self._viewer = DataFrameViewer(result_frame)
        self._viewer.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Gestionnaires d'événements
    # ------------------------------------------------------------------

    def _get_methods(self):
        """Retourne la liste des méthodes cochées, ou None si aucune."""
        methods = []
        if self._method_linear.get():    methods.append('linear')
        if self._method_mutual.get():    methods.append('mutual_info')
        if self._method_importance.get(): methods.append('importance')
        if self._method_rfe.get():       methods.append('rfe')
        return methods if methods else None

    def _on_build(self) -> None:
        if not self.controller.has_data():
            messagebox.showwarning('Données manquantes',
                                   "Veuillez d'abord charger les fichiers dans l'onglet Données.")
            return

        options = {
            'drop_inactive_students': self._drop_inactive.get(),
            'remove_outliers':        self._remove_outliers.get(),
            'selection_methods':      self._get_methods(),
            'k_features':             self._k_features.get(),
            'prefilter_variance':     self._prefilter_var.get(),
            'prefilter_correlation':  self._prefilter_corr.get(),
        }

        self._build_btn.configure(state='disabled')
        self.controller.set_status('Construction du dataset en cours… (peut prendre quelques secondes)')
        self.controller.set_loading(True)

        self._run_thread(
            worker=lambda: self.controller.build_dataset(options),
            on_success=self._on_build_success,
            on_error=self._on_build_error,
        )

    def _on_build_success(self, result) -> None:
        X, y, features = result
        n, m = X.shape

        self._shape_label.configure(
            text=f'Shape : {n} échantillons × {m} features  |  Ratio train/test : '
                 f'{int(round(n * (1 - self.controller.config.TEST_SPLIT_RATIO)))} / '
                 f'{int(round(n * self.controller.config.TEST_SPLIT_RATIO))}')
        self._features_label.configure(
            text='Features retenues : ' + ', '.join(features))

        import pandas as pd
        df_preview = X.copy()
        df_preview[self.controller.config.TARGET_COLUMN] = y.values
        self._viewer.set_dataframe(df_preview)

        self._build_btn.configure(state='normal')
        self.controller.set_loading(False)
        self.controller.set_status(
            f'Dataset construit — {n} échantillons, {m} features sélectionnées')
        self.main_window.unlock_tab(2)

    def _on_build_error(self, message: str) -> None:
        self._build_btn.configure(state='normal')
        self.controller.set_loading(False)
        self.controller.set_status('Erreur lors de la construction du dataset')
        messagebox.showerror('Erreur de pipeline', message)

    # ------------------------------------------------------------------
    # Utilitaire threading
    # ------------------------------------------------------------------

    def _run_thread(self, worker, on_success, on_error=None) -> None:
        def target():
            try:
                result = worker()
                self.frame.after(0, on_success, result)
            except Exception as exc:
                if on_error:
                    self.frame.after(0, on_error, str(exc))
                else:
                    self.frame.after(0, messagebox.showerror, 'Erreur', str(exc))

        threading.Thread(target=target, daemon=True).start()
