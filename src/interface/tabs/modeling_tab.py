#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Onglet 3 - Configuration et entraînement des modèles.

Permet de :
  - Choisir le type de modèle (LinearRegressor, RandomForest, GradientBoosting)
  - Configurer les hyperparamètres via un panneau dynamique
  - Lancer l'entraînement dans un thread de fond
  - Lister les modèles entraînés et en supprimer

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


# ----------------------------------------------------------------------------------------------------------------------
# Classe ModelingTab
# ----------------------------------------------------------------------------------------------------------------------
class ModelingTab:
    """Onglet de création et d'entraînement des modèles de régression."""

    # Labels affichés dans l'UI pour chaque type interne
    _TYPE_LABELS = {
        'linear':            'Régression Linéaire',
        'random_forest':     'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
    }
    # Préfixes auto-générés pour les noms de modèles
    _NAME_PREFIXES = {
        'linear':            'LinearReg',
        'random_forest':     'RandomForest',
        'gradient_boosting': 'GradBoost',
    }

    def __init__(self, parent, controller: AppController, main_window):
        self.controller  = controller
        self.main_window = main_window

        self.frame = ttk.Frame(parent)
        self._build()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # --- Panneau de configuration ---
        config_frame = ttk.LabelFrame(self.frame, text=' Créer un modèle ', padding=12)
        config_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        # Sélection du type
        type_row = ttk.Frame(config_frame)
        type_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(type_row, text='Type :').pack(side=tk.LEFT, padx=(0, 12))

        self._model_type = tk.StringVar(value='linear')
        for val, label in self._TYPE_LABELS.items():
            ttk.Radiobutton(type_row, text=label, variable=self._model_type,
                            value=val, command=self._on_type_change).pack(side=tk.LEFT, padx=8)

        # Nom du modèle
        name_row = ttk.Frame(config_frame)
        name_row.pack(fill=tk.X, pady=4)
        ttk.Label(name_row, text='Nom :').pack(side=tk.LEFT, padx=(0, 12))
        self._model_name = tk.StringVar(value='LinearReg_1')
        ttk.Entry(name_row, textvariable=self._model_name, width=32).pack(side=tk.LEFT)

        # Conteneur des hyperparamètres (frames empilées, une seule visible à la fois)
        self._hp_container = ttk.Frame(config_frame)
        self._hp_container.pack(fill=tk.X, pady=6)
        self._hp_container.columnconfigure(0, weight=1)

        self._hp_frames = {
            'linear':            self._build_linear_hp(self._hp_container),
            'random_forest':     self._build_rf_hp(self._hp_container),
            'gradient_boosting': self._build_gb_hp(self._hp_container),
        }
        self._on_type_change()  # Affiche le bon panneau d'emblée

        # Bouton entraîner
        btn_row = ttk.Frame(config_frame)
        btn_row.pack(pady=(6, 0))
        self._train_btn = ttk.Button(btn_row, text='  Entraîner le modèle  ',
                                     command=self._on_train)
        self._train_btn.pack()

        # --- Liste des modèles entraînés ---
        list_frame = ttk.LabelFrame(self.frame, text=' Modèles entraînés ', padding=8)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        cols = ('Nom', 'Type', 'Statut')
        self._tree = ttk.Treeview(list_frame, columns=cols, show='headings', height=7)
        for col in cols:
            self._tree.heading(col, text=col)
        self._tree.column('Nom',    width=220)
        self._tree.column('Type',   width=200)
        self._tree.column('Statut', width=130)

        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(list_frame, text='Supprimer le modèle sélectionné',
                   command=self._on_delete).pack(pady=(6, 0))

    # ------------------------------------------------------------------
    # Panneaux d'hyperparamètres
    # ------------------------------------------------------------------

    def _build_linear_hp(self, parent) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text=' Hyperparamètres - Régression Linéaire ', padding=10)
        ttk.Label(f, text='Aucun hyperparamètre configurable (OLS standard).',
                  foreground='gray').pack()
        return f

    def _build_rf_hp(self, parent) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text=' Hyperparamètres - Random Forest ', padding=10)

        self._rf_n_est    = tk.IntVar(value=200)
        self._rf_max_depth = tk.StringVar(value='10')   # str pour autoriser "None"
        self._rf_min_split = tk.IntVar(value=5)
        self._rf_min_leaf  = tk.IntVar(value=3)

        params = [
            ('n_estimators',      self._rf_n_est,    'spinbox', {'from_': 10,  'to': 2000}),
            ('max_depth',         self._rf_max_depth, 'entry',   {}),
            ('min_samples_split', self._rf_min_split, 'spinbox', {'from_': 2,  'to': 100}),
            ('min_samples_leaf',  self._rf_min_leaf,  'spinbox', {'from_': 1,  'to': 50}),
        ]
        self._place_hp_grid(f, params)
        ttk.Label(f, text='  (max_depth : entier ou "None" pour illimité)',
                  foreground='gray', font=('', 8)).grid(row=1, column=3, sticky='w', padx=8)
        return f

    def _build_gb_hp(self, parent) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text=' Hyperparamètres - Gradient Boosting ', padding=10)

        self._gb_n_est     = tk.IntVar(value=100)
        self._gb_lr        = tk.StringVar(value='0.1')
        self._gb_max_depth = tk.IntVar(value=3)
        self._gb_min_split = tk.IntVar(value=10)
        self._gb_subsample = tk.StringVar(value='0.8')

        params = [
            ('n_estimators',      self._gb_n_est,     'spinbox', {'from_': 10, 'to': 2000}),
            ('learning_rate',     self._gb_lr,         'entry',   {}),
            ('max_depth',         self._gb_max_depth,  'spinbox', {'from_': 1,  'to': 20}),
            ('min_samples_split', self._gb_min_split,  'spinbox', {'from_': 2,  'to': 100}),
            ('subsample',         self._gb_subsample,  'entry',   {}),
        ]
        self._place_hp_grid(f, params)
        return f

    @staticmethod
    def _place_hp_grid(parent, params) -> None:
        """Dispose les widgets d'hyperparamètres en grille 2 colonnes."""
        for i, (label, var, wtype, kwargs) in enumerate(params):
            row      = i // 2
            col_base = (i % 2) * 3
            ttk.Label(parent, text=f'{label} :').grid(
                row=row, column=col_base, sticky='w', padx=(12, 4), pady=4)
            if wtype == 'spinbox':
                ttk.Spinbox(parent, textvariable=var, width=9, **kwargs).grid(
                    row=row, column=col_base + 1, sticky='w')
            else:
                ttk.Entry(parent, textvariable=var, width=9).grid(
                    row=row, column=col_base + 1, sticky='w')

    # ------------------------------------------------------------------
    # Logique de sélection du type de modèle
    # ------------------------------------------------------------------

    def _on_type_change(self) -> None:
        """Cache tous les panneaux HP et affiche celui du type sélectionné."""
        t = self._model_type.get()

        for key, frame in self._hp_frames.items():
            frame.grid_remove()
        self._hp_frames[t].grid(row=0, column=0, sticky='ew')

        # Mettre à jour le nom auto-généré
        prefix = self._NAME_PREFIXES[t]
        count = sum(1 for name in self.controller.trained_models
                    if name.startswith(prefix))
        self._model_name.set(f'{prefix}_{count + 1}')

    # ------------------------------------------------------------------
    # Récupération des hyperparamètres selon le type
    # ------------------------------------------------------------------

    def _get_hyperparams(self) -> dict:
        t = self._model_type.get()

        if t == 'linear':
            return {}

        if t == 'random_forest':
            max_depth_str = self._rf_max_depth.get().strip().lower()
            max_depth = None if max_depth_str == 'none' else int(max_depth_str)
            return {
                'RF_N_ESTIMATORS':   self._rf_n_est.get(),
                'RF_MAX_DEPTH':      max_depth,
                'RF_MIN_SAMPLES_SPLIT': self._rf_min_split.get(),
                'RF_MIN_SAMPLES_LEAF':  self._rf_min_leaf.get(),
            }

        # gradient_boosting
        return {
            'GB_N_ESTIMATORS':   self._gb_n_est.get(),
            'GB_LEARNING_RATE':  float(self._gb_lr.get()),
            'GB_MAX_DEPTH':      self._gb_max_depth.get(),
            'GB_MIN_SAMPLES_SPLIT': self._gb_min_split.get(),
            'GB_SUBSAMPLE':      float(self._gb_subsample.get()),
        }

    # ------------------------------------------------------------------
    # Gestionnaires d'événements
    # ------------------------------------------------------------------

    def _on_train(self) -> None:
        if not self.controller.has_dataset():
            messagebox.showwarning('Dataset manquant',
                                   "Veuillez d'abord construire le dataset dans l'onglet Prétraitement.")
            return

        name = self._model_name.get().strip()
        if not name:
            messagebox.showwarning('Nom manquant', 'Veuillez donner un nom au modèle.')
            return
        if name in self.controller.trained_models:
            messagebox.showwarning('Nom déjà utilisé',
                                   f'Un modèle nommé "{name}" existe déjà. Choisissez un autre nom.')
            return

        model_type = self._model_type.get()
        try:
            hyperparams = self._get_hyperparams()
        except ValueError as exc:
            messagebox.showerror('Hyperparamètres invalides', str(exc))
            return

        display_type = self._TYPE_LABELS[model_type]
        self._train_btn.configure(state='disabled')
        self.controller.set_status(f"Entraînement de '{name}' ({display_type})…")
        self.controller.set_loading(True)

        self._run_thread(
            worker=lambda: self.controller.train_model(name, model_type, hyperparams),
            on_success=lambda model: self._on_train_success(name, display_type),
            on_error=self._on_train_error,
        )

    def _on_train_success(self, name: str, display_type: str) -> None:
        self._tree.insert('', tk.END, iid=name,
                          values=(name, display_type, 'Entraîné ✓'))
        self._train_btn.configure(state='normal')
        self._on_type_change()   # rafraîchit le nom auto-généré
        self.controller.set_loading(False)
        self.controller.set_status(f"Modèle '{name}' entraîné avec succès")
        self.main_window.unlock_tab(3)

    def _on_train_error(self, message: str) -> None:
        self._train_btn.configure(state='normal')
        self.controller.set_loading(False)
        self.controller.set_status("Erreur lors de l'entraînement")
        messagebox.showerror("Erreur d'entraînement", message)

    def _on_delete(self) -> None:
        selected = self._tree.selection()
        if not selected:
            messagebox.showinfo('Aucune sélection',
                                'Sélectionnez un modèle dans la liste pour le supprimer.')
            return
        name = selected[0]
        if messagebox.askyesno('Confirmer la suppression',
                                f'Supprimer le modèle "{name}" ?'):
            self.controller.delete_model(name)
            self._tree.delete(name)
            self.controller.set_status(f"Modèle '{name}' supprimé")

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
