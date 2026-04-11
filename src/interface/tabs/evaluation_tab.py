#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 11/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
Onglet 4 - Évaluation et comparaison des modèles.

Permet de :
  - Calculer et afficher les métriques de tous les modèles entraînés
  - Obtenir une recommandation automatique
  - Afficher les graphiques matplotlib (prédictions, résidus, comparaison)
  - Exporter les résultats (CSV + PNG) dans un répertoire

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


class EvaluationTab:
    """Onglet d'évaluation et de comparaison des modèles de régression."""

    def __init__(self, parent, controller: AppController, main_window):
        self.controller  = controller
        self.main_window = main_window

        self.frame = ttk.Frame(parent)
        self._build()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # --- Section métriques ---
        metrics_frame = ttk.LabelFrame(self.frame, text=' Métriques de performance ', padding=10)
        metrics_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        top_row = ttk.Frame(metrics_frame)
        top_row.pack(fill=tk.X, pady=(0, 6))

        self._adj_r2 = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_row, text='Inclure R² ajusté',
                        variable=self._adj_r2).pack(side=tk.LEFT)
        ttk.Button(top_row, text='Évaluer tous les modèles',
                   command=self._on_evaluate).pack(side=tk.LEFT, padx=16)
        ttk.Button(top_row, text='Recommandation',
                   command=self._on_recommend).pack(side=tk.LEFT)

        # Treeview des métriques
        tree_frame = ttk.Frame(metrics_frame)
        tree_frame.pack(fill=tk.X)

        self._metrics_tree = ttk.Treeview(tree_frame, show='headings', height=5)
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                            command=self._metrics_tree.yview)
        self._metrics_tree.configure(yscrollcommand=vsb.set)
        self._metrics_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Section visualisations ---
        viz_frame = ttk.LabelFrame(self.frame, text=' Visualisations ', padding=6)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        btn_bar = ttk.Frame(viz_frame)
        btn_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_bar, text='Prédictions vs Réel',
                   command=self._plot_predictions).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text='Distribution des résidus',
                   command=self._plot_residuals).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text='Comparaison des métriques',
                   command=self._plot_metrics).pack(side=tk.LEFT, padx=4)

        # Zone d'accueil du canvas matplotlib
        self._plot_frame = ttk.Frame(viz_frame, relief=tk.SUNKEN, borderwidth=1)
        self._plot_frame.pack(fill=tk.BOTH, expand=True)

        # --- Pied de page ---
        footer = ttk.Frame(self.frame)
        footer.pack(fill=tk.X, padx=12, pady=(0, 8))
        ttk.Button(footer, text='Exporter les résultats (CSV + graphiques PNG)',
                   command=self._on_export).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------

    def _on_evaluate(self) -> None:
        if not self.controller.has_models():
            messagebox.showwarning('Aucun modèle',
                                   "Entraînez au moins un modèle avant d'évaluer.")
            return
        try:
            df = self.controller.evaluate_all(include_adjusted_r2=self._adj_r2.get())
            self._refresh_metrics_table(df)
            self.controller.set_status('Évaluation terminée')
        except Exception as exc:
            messagebox.showerror('Erreur', str(exc))

    def _refresh_metrics_table(self, df) -> None:
        """Reconfigure le Treeview et le remplit avec le DataFrame des métriques."""
        cols = ['Modèle'] + list(df.columns)
        self._metrics_tree['columns'] = cols
        for col in cols:
            self._metrics_tree.heading(col, text=col)
            self._metrics_tree.column(col, width=120, anchor='center')
        self._metrics_tree.column('Modèle', width=200, anchor='w')

        for item in self._metrics_tree.get_children():
            self._metrics_tree.delete(item)

        for model_name, row in df.iterrows():
            values = [model_name] + [f'{v:.4f}' for v in row]
            self._metrics_tree.insert('', tk.END, values=values)

    def _on_recommend(self) -> None:
        if not self.controller.has_models():
            messagebox.showwarning('Aucun modèle',
                                   "Entraînez au moins un modèle avant d'obtenir une recommandation.")
            return
        try:
            rec = self.controller.get_recommendation(use_adjusted_r2=self._adj_r2.get())
            lines = [
                f"Meilleur modèle : {rec['best_model']}",
                f"Raison           : {rec['reason']}",
                '',
                'Métriques :',
            ]
            for k, v in rec['metrics'].items():
                lines.append(f'  {k} = {v:.4f}')
            messagebox.showinfo('Recommandation', '\n'.join(lines))
        except Exception as exc:
            messagebox.showerror('Erreur', str(exc))

    # ------------------------------------------------------------------
    # Graphiques matplotlib
    # ------------------------------------------------------------------

    def _display_figure(self, fig) -> None:
        """Remplace le contenu du plot_frame par une nouvelle figure matplotlib."""
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        for widget in self._plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, self._plot_frame)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_predictions(self) -> None:
        self._run_plot(self.controller.get_plot_predictions)

    def _plot_residuals(self) -> None:
        self._run_plot(self.controller.get_plot_residuals)

    def _plot_metrics(self) -> None:
        self._run_plot(lambda: self.controller.get_plot_metrics(self._adj_r2.get()))

    def _run_plot(self, plot_func) -> None:
        if not self.controller.has_models():
            messagebox.showwarning('Aucun modèle',
                                   "Entraînez au moins un modèle avant de tracer.")
            return
        self.controller.set_loading(True)

        def target():
            try:
                fig = plot_func()
                self.frame.after(0, self._on_plot_success, fig)
            except Exception as exc:
                self.frame.after(0, self._on_plot_error, str(exc))

        threading.Thread(target=target, daemon=True).start()

    def _on_plot_success(self, fig) -> None:
        self._display_figure(fig)
        self.controller.set_loading(False)

    def _on_plot_error(self, message: str) -> None:
        self.controller.set_loading(False)
        messagebox.showerror('Erreur de visualisation', message)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        if not self.controller.has_models():
            messagebox.showwarning('Aucun modèle',
                                   "Entraînez au moins un modèle avant d'exporter.")
            return

        output_dir = filedialog.askdirectory(title='Choisir le dossier de destination')
        if not output_dir:
            return

        self.controller.set_status('Export en cours…')
        self.controller.set_loading(True)

        include_adj = self._adj_r2.get()

        def target():
            try:
                self.controller.export_results(output_dir, include_adjusted_r2=include_adj)
                self.frame.after(0, self._on_export_success, output_dir)
            except Exception as exc:
                self.frame.after(0, self._on_export_error, str(exc))

        threading.Thread(target=target, daemon=True).start()

    def _on_export_success(self, output_dir: str) -> None:
        self.controller.set_loading(False)
        self.controller.set_status(f'Résultats exportés dans {output_dir}')
        messagebox.showinfo('Export réussi',
                            f'Les résultats ont été exportés dans :\n{output_dir}')

    def _on_export_error(self, message: str) -> None:
        self.controller.set_loading(False)
        self.controller.set_status("Erreur lors de l'export")
        messagebox.showerror("Erreur d'export", message)
