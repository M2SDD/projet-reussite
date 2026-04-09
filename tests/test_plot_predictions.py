#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour la méthode plot_predictions() de ModelEvaluator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.linear_regressor import LinearRegressor
from src.models.ensemble_regressor import EnsembleRegressor

from src.evaluation.model_evaluator import ModelEvaluator


def test_plot_predictions():
    """
    Teste la méthode plot_predictions() avec plusieurs modèles.
    """
    print("=" * 80)
    print("Test de la méthode plot_predictions()")
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
    model1 = LinearRegressor()
    model1.fit(X_df, y_series)
    print("   ✓ Modèle de régression linéaire entraîné")

    # Modèle 2 : Random Forest
    model2 = EnsembleRegressor()
    model2.fit(X_df, y_series)
    print("   ✓ Modèle Random Forest entraîné")

    # Créer l'évaluateur et enregistrer les modèles
    print("\n2. Enregistrement des modèles...")
    evaluator = ModelEvaluator()
    evaluator.add_model('Régression Linéaire', model1, X_df, y_series)
    evaluator.add_model('Forêt Aléatoire', model2, X_df, y_series)
    print("   ✓ 2 modèles enregistrés")

    # Générer le graphique de prédictions
    print("\n3. Génération du graphique de prédictions...")
    fig = evaluator.plot_predictions()
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
    print("   ✓ (2) Grille de sous-graphiques créée avec un graphique par modèle (2 graphiques)")

    # Vérification 3 : Scatter plots
    for idx, ax in enumerate(visible_axes):
        collections = ax.collections
        assert len(collections) > 0, f"Le sous-graphique {idx} devrait contenir un scatter plot"
    print("   ✓ (3) Chaque graphique contient un scatter plot de prédictions vs valeurs réelles")

    # Vérification 4 : Ligne diagonale de référence
    for idx, ax in enumerate(visible_axes):
        lines = ax.get_lines()
        assert len(lines) > 0, f"Le sous-graphique {idx} devrait contenir une ligne diagonale"
        # Vérifier qu'au moins une ligne est diagonale (y=x)
        has_diagonal = False
        for line in lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) >= 2 and len(ydata) >= 2:
                # Vérifier si c'est une ligne y=x (à epsilon près)
                if np.allclose(xdata, ydata, rtol=1e-5):
                    has_diagonal = True
                    break
        assert has_diagonal, f"Le sous-graphique {idx} devrait contenir une ligne diagonale y=x"
    print("   ✓ (4) Ligne diagonale de référence à y=x présente sur chaque graphique")

    # Vérification 5 : Labels en français
    french_keywords = ['valeur', 'réel', 'prédit', 'prédiction']
    for idx, ax in enumerate(visible_axes):
        xlabel = ax.get_xlabel().lower()
        ylabel = ax.get_ylabel().lower()
        title = ax.get_title().lower()

        # Vérifier la présence de mots français
        has_french = any(keyword in xlabel or keyword in ylabel or keyword in title
                        for keyword in french_keywords)
        assert has_french, f"Le sous-graphique {idx} devrait avoir des labels/titre en français"
    print("   ✓ (5) Labels et titres en français")

    # Sauvegarder le graphique pour inspection visuelle
    output_path = 'tests/test_plot_predictions_output.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n5. Graphique sauvegardé : {output_path}")

    # Fermer la figure pour libérer la mémoire
    plt.close(fig)

    print("\n" + "=" * 80)
    print("✓ TOUS LES TESTS RÉUSSIS")
    print("=" * 80)


def test_single_model():
    """
    Teste plot_predictions() avec un seul modèle.
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
    model = LinearRegressor()
    model.fit(X_df, y)

    # Créer l'évaluateur
    evaluator = ModelEvaluator()
    evaluator.add_model('Modèle Unique', model, X_df, y)

    # Générer le graphique
    fig = evaluator.plot_predictions()

    # Vérifications
    axes = fig.get_axes()
    visible_axes = [ax for ax in axes if ax.get_visible()]
    assert len(visible_axes) == 1, "Devrait avoir 1 sous-graphique avec un seul modèle"

    print("✓ Test avec un seul modèle réussi")
    plt.close(fig)


def test_error_no_models():
    """
    Teste que plot_predictions() lève une erreur si aucun modèle n'est enregistré.
    """
    print("\n" + "=" * 80)
    print("Test de gestion d'erreur : aucun modèle")
    print("=" * 80)

    evaluator = ModelEvaluator()

    try:
        evaluator.plot_predictions()
        assert False, "Devrait lever une ValueError"
    except ValueError as e:
        assert "Aucun modèle" in str(e)
        print(f"✓ Erreur correctement levée : {e}")


if __name__ == '__main__':
    test_plot_predictions()
    test_single_model()
    test_error_no_models()

    print("\n" + "=" * 80)
    print("TOUS LES TESTS TERMINÉS AVEC SUCCÈS !")
    print("=" * 80)
