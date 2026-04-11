#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Onglet 1 - Chargement et aperçu des données brutes.

Permet à l'utilisateur de :
  - Sélectionner les fichiers CSV logs et notes via un dialogue
  - Charger directement les fichiers de démonstration du dossier data/
  - Visualiser les premières lignes de chaque fichier dans un DataFrameViewer

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
from tkinter import filedialog, messagebox, ttk

from ..app_controller import AppController
from ..widgets.dataframe_viewer import DataFrameViewer


# ----------------------------------------------------------------------------------------------------------------------
# Classe DataTab
# ----------------------------------------------------------------------------------------------------------------------
class DataTab:
    """Onglet de chargement et d'aperçu des données."""

    def __init__(self, parent, controller: AppController, main_window):
        self.controller  = controller
        self.main_window = main_window

        self.frame = ttk.Frame(parent)
        self._build()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self._logs_path_var  = tk.StringVar()
        self._notes_path_var = tk.StringVar()
        self._active_view    = tk.StringVar(value='logs')

        # --- Section chargement ---
        load_frame = ttk.LabelFrame(self.frame, text=' Chargement des fichiers ', padding=10)
        load_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        # Logs
        ttk.Label(load_frame, text='Logs :').grid(row=0, column=0, sticky='w', pady=4)
        ttk.Entry(load_frame, textvariable=self._logs_path_var, width=60).grid(
            row=0, column=1, padx=6, sticky='ew')
        ttk.Button(load_frame, text='Parcourir…', command=self._browse_logs).grid(row=0, column=2)

        # Notes
        ttk.Label(load_frame, text='Notes :').grid(row=1, column=0, sticky='w', pady=4)
        ttk.Entry(load_frame, textvariable=self._notes_path_var, width=60).grid(
            row=1, column=1, padx=6, sticky='ew')
        ttk.Button(load_frame, text='Parcourir…', command=self._browse_notes).grid(row=1, column=2)

        load_frame.columnconfigure(1, weight=1)

        # Boutons d'action
        btn_frame = ttk.Frame(load_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(btn_frame, text='Utiliser les données démo',
                   command=self._load_demo).pack(side=tk.LEFT, padx=8)
        self._load_btn = ttk.Button(btn_frame, text='Charger les fichiers',
                                    command=self._on_load)
        self._load_btn.pack(side=tk.LEFT, padx=8)

        # --- Ligne d'information ---
        self._info_label = ttk.Label(self.frame, text='', foreground='gray')
        self._info_label.pack(anchor='w', padx=14, pady=(2, 0))

        # --- Section aperçu ---
        view_frame = ttk.LabelFrame(self.frame, text=' Aperçu ', padding=6)
        view_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        radio_bar = ttk.Frame(view_frame)
        radio_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Radiobutton(radio_bar, text='Logs', variable=self._active_view,
                        value='logs', command=self._switch_view).pack(side=tk.LEFT)
        ttk.Radiobutton(radio_bar, text='Notes', variable=self._active_view,
                        value='notes', command=self._switch_view).pack(side=tk.LEFT, padx=12)

        self._viewer = DataFrameViewer(view_frame)
        self._viewer.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Gestionnaires d'événements
    # ------------------------------------------------------------------

    def _browse_logs(self) -> None:
        path = filedialog.askopenfilename(
            title='Sélectionner le fichier de logs',
            filetypes=[('CSV', '*.csv'), ('Tous les fichiers', '*.*')]
        )
        if path:
            self._logs_path_var.set(path)

    def _browse_notes(self) -> None:
        path = filedialog.askopenfilename(
            title='Sélectionner le fichier de notes',
            filetypes=[('CSV', '*.csv'), ('Tous les fichiers', '*.*')]
        )
        if path:
            self._notes_path_var.set(path)

    def _load_demo(self) -> None:
        """Pré-remplit les champs avec les chemins démo puis lance le chargement."""
        self._logs_path_var.set(self.controller.DEMO_LOGS)
        self._notes_path_var.set(self.controller.DEMO_NOTES)
        self._on_load()

    def _on_load(self) -> None:
        logs  = self._logs_path_var.get().strip()
        notes = self._notes_path_var.get().strip()

        if not logs or not notes:
            messagebox.showwarning('Champs manquants',
                                   'Veuillez renseigner les chemins des deux fichiers.')
            return

        self._load_btn.configure(state='disabled')
        self.controller.set_status('Chargement des données en cours…')
        self.controller.set_loading(True)

        self._run_thread(
            worker=lambda: self.controller.load_data(logs, notes),
            on_success=self._on_load_success,
            on_error=self._on_load_error,
        )

    def _on_load_success(self, result) -> None:
        df_logs, df_notes = result
        n_logs    = len(df_logs)
        n_notes   = len(df_notes)
        n_pseudos = df_notes['pseudo'].nunique()

        self._info_label.configure(
            text=f'Logs : {n_logs:,} lignes  |  Notes : {n_notes} étudiants ({n_pseudos} pseudos uniques)',
            foreground='black'
        )
        self._switch_view()
        self._load_btn.configure(state='normal')
        self.controller.set_loading(False)
        self.controller.set_status(
            f'Données chargées — {n_logs:,} événements logs, {n_notes} étudiants')
        self.main_window.unlock_tab(1)

    def _on_load_error(self, message: str) -> None:
        self._load_btn.configure(state='normal')
        self.controller.set_loading(False)
        self.controller.set_status('Erreur lors du chargement')
        messagebox.showerror('Erreur de chargement', message)

    def _switch_view(self) -> None:
        """Affiche les logs ou les notes selon le radiobutton sélectionné."""
        view = self._active_view.get()
        if view == 'logs' and self.controller.df_logs is not None:
            self._viewer.set_dataframe(self.controller.df_logs)
        elif view == 'notes' and self.controller.df_notes is not None:
            self._viewer.set_dataframe(self.controller.df_notes)

    # ------------------------------------------------------------------
    # Utilitaire threading commun aux onglets
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
