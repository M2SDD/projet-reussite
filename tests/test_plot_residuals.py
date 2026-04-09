#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour la méthode plot_residuals() de ModelEvaluator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.model_evaluator import ModelEvaluator


def test_plot_residuals():
    """
    Teste la méthode plot_residuals() avec plusieurs modèles.
    """
    print("=" * 80)
    print("Test de la méthode plot_residuals()")
    print("=" * 80)

    # Créer des données synthétiques pour le test
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    # Générer les features
    X = np.random.randn(n_samples, n_features)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # Générer la cible avec une relation linéaire + bruit
    true_coefficients = np.array([2.5, -1.3, 0.8])
    y = X.dot(true_coefficients) + np.random.randn(n_samples) * 0.5
    y_series = pd.Series(y, name='target')

    # Créer et entraîner plusieurs modèles
    print("\n1. Entraînement des modèles...")

    # Modèle 1 : Régression linéaire
    model1 = LinearRegression()
    model1.fit(X_df, y_series)
    print("   ✓ Modèle de régression linéaire entraîné")

    # Modèle 2 : Random Forest
    model2 = RandomForestRegressor(n_estimators=50, random_state=42)
    model2.fit(X_df, y_series)
    print("   ✓ Modèle Random Forest entraîné")

    # Créer l'évaluateur et enregistrer les modèles
    print("\n2. Enregistrement des modèles...")
    evaluator = ModelEvaluator()
    evaluator.add_model('Régression Linéaire', model1, X_df, y_series)
    evaluator.add_model('Forêt Aléatoire', model2, X_df, y_series)
    print("   ✓ 2 modèles enregistrés")

    # Générer le graphique de résidus
    print("\n3. Génération du graphique de résidus...")
    fig = evaluator.plot_residuals()
    print("   ✓ Graphique généré")

    # Vérifications
    print("\n4. Vérifications...")

    # Vérification 1 : Type de retour
    assert isinstance(fig, plt.Figure), "La méthode doit retourner une matplotlib.figure.Figure"
    print("   ✓ (1) Retourne bien une matplotlib.figure.Figure")

    # Vérification 2 : Nombre de sous-graphiques
    axes = fig.get_axes()
    visible_axes = [ax for ax in axes if ax.get_visible()]
    assert len(visible_axes) == 2, f"Devrait avoir 2 sous-graphiques visibles, trouvé {len(visible_axes)}"
    print("   ✓ (2) Grille de sous-graphiques créée avec un histogramme par modèle (2 graphiques)")

    # Vérification 3 : Histogrammes des résidus
    for idx, ax in enumerate(visible_axes):
        patches = ax.patches
        assert len(patches) > 0, f"Le sous-graphique {idx} devrait contenir un histogramme"
    print("   ✓ (3) Chaque graphique contient un histogramme des résidus")

    # Vérification 4 : Courbe de distribution normale
    for idx, ax in enumerate(visible_axes):
        lines = ax.get_lines()
        # Devrait avoir au moins 3 lignes : courbe normale + ligne à x=0 + ligne à la moyenne
        assert len(lines) >= 2, f"Le sous-graphique {idx} devrait contenir au moins 2 lignes"
    print("   ✓ (4) Courbe de distribution normale superposée")

    # Vérification 5 : Ligne verticale à x=0
    for idx, ax in enumerate(visible_axes):
        lines = ax.get_lines()
        # Vérifier qu'il y a au moins une ligne verticale
        has_vertical_line = False
        for line in lines:
            xdata = line.get_xdata()
            # Une ligne verticale a des valeurs x constantes
            if len(xdata) >= 2 and np.allclose(xdata, xdata[0], rtol=1e-10):
                has_vertical_line = True
                break
        assert has_vertical_line, f"Le sous-graphique {idx} devrait contenir une ligne verticale"
    print("   ✓ (5) Ligne verticale à x=0 (moyenne des résidus) présente")

    # Vérification 6 : Labels en français
    french_keywords = ['résidu', 'densité', 'distribution', 'probabilité']
    for idx, ax in enumerate(visible_axes):
        xlabel = ax.get_xlabel().lower()
        ylabel = ax.get_ylabel().lower()
        title = ax.get_title().lower()

        # Vérifier la présence de mots français
        has_french = any(keyword in xlabel or keyword in ylabel or keyword in title
                        for keyword in french_keywords)
        assert has_french, f"Le sous-graphique {idx} devrait avoir des labels/titre en français"
    print("   ✓ (6) Labels et titres en français")

    # Sauvegarder le graphique pour inspection visuelle
    output_path = 'tests/test_plot_residuals_output.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n5. Graphique sauvegardé : {output_path}")

    # Fermer la figure pour libérer la mémoire
    plt.close(fig)

    print("\n" + "=" * 80)
    print("✓ TOUS LES TESTS RÉUSSIS")
    print("=" * 80)


def test_single_model():
    """
    Teste plot_residuals() avec un seul modèle.
    """
    print("\n" + "=" * 80)
    print("Test avec un seul modèle")
    print("=" * 80)

    # Créer des données synthétiques
    np.random.seed(42)
    X = np.random.randn(50, 2)
    X_df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
    y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(50) * 0.3

    # Créer et entraîner un modèle
    model = LinearRegression()
    model.fit(X_df, y)

    # Créer l'évaluateur
    evaluator = ModelEvaluator()
    evaluator.add_model('Modèle Unique', model, X_df, y)

    # Générer le graphique
    fig = evaluator.plot_residuals()

    # Vérifications
    axes = fig.get_axes()
    visible_axes = [ax for ax in axes if ax.get_visible()]
    assert len(visible_axes) == 1, "Devrait avoir 1 sous-graphique avec un seul modèle"

    print("✓ Test avec un seul modèle réussi")
    plt.close(fig)


def test_error_no_models():
    """
    Teste que plot_residuals() lève une erreur si aucun modèle n'est enregistré.
    """
    print("\n" + "=" * 80)
    print("Test de gestion d'erreur : aucun modèle")
    print("=" * 80)

    evaluator = ModelEvaluator()

    try:
        evaluator.plot_residuals()
        assert False, "Devrait lever une ValueError"
    except ValueError as e:
        assert "Aucun modèle" in str(e)
        print(f"✓ Erreur correctement levée : {e}")


if __name__ == '__main__':
    test_plot_residuals()
    test_single_model()
    test_error_no_models()

    print("\n" + "=" * 80)
    print("TOUS LES TESTS TERMINÉS AVEC SUCCÈS !")
    print("=" * 80)
