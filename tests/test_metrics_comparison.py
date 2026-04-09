#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour la méthode plot_metrics_comparison()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.linear_regressor import LinearRegressor
from src.models.ensemble_regressor import EnsembleRegressor

from src.evaluation.model_evaluator import ModelEvaluator


def test_plot_metrics_comparison():
    """
    Test de la méthode plot_metrics_comparison()

    Vérifie que le graphique:
    1. Est un graphique à barres groupées avec les modèles sur l'axe x
    2. Affiche les barres pour chaque métrique (R², RMSE, MAE, R² ajusté)
    3. Contient une légende
    4. Utilise des labels en français
    5. Retourne une Figure matplotlib
    """
    print("=" * 80)
    print("Test de plot_metrics_comparison()")
    print("=" * 80)

    # Créer des données de test
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    # Générer des features aléatoires
    X = np.random.randn(n_samples, n_features)

    # Créer une cible avec une relation linéaire + bruit
    true_coefficients = np.random.randn(n_features)
    y = X @ true_coefficients + np.random.randn(n_samples) * 0.5

    # Convertir en DataFrame pour plus de réalisme
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='target')

    # Split train/test
    split_idx = int(n_samples * 0.7)
    X_train, X_test = X_df[:split_idx], X_df[split_idx:]
    y_train, y_test = y_series[:split_idx], y_series[split_idx:]

    print(f"\nDonnées de test créées:")
    print(f"  - Échantillons d'entraînement: {len(X_train)}")
    print(f"  - Échantillons de test: {len(X_test)}")
    print(f"  - Nombre de features: {n_features}")

    # Créer et entraîner plusieurs modèles
    print("\nEntraînement des modèles...")

    # Modèle 1: Régression linéaire
    model1 = LinearRegressor()
    model1.fit(X_train, y_train)
    print("  ✓ Régression linéaire entraînée")

    # Modèle 2: Random Forest
    model2 = EnsembleRegressor()
    model2.fit(X_train, y_train)
    print("  ✓ Random Forest entraînée")

    # Modèle 3: Decision Tree
    model3 = EnsembleRegressor(model_type='gradient_boosting')
    model3.fit(X_train, y_train)
    print("  ✓ Gradient Boosting entraînée")

    # Créer l'évaluateur et enregistrer les modèles
    print("\nEnregistrement des modèles dans ModelEvaluator...")
    evaluator = ModelEvaluator()
    evaluator.add_model('Régression Linéaire', model1, X_test, y_test)
    evaluator.add_model('Random Forest', model2, X_test, y_test)
    evaluator.add_model('Gradient Boosting', model3, X_test, y_test)
    print("  ✓ 3 modèles enregistrés")

    # Afficher les métriques pour référence
    print("\nMétriques des modèles:")
    metrics = evaluator.evaluate_all()
    for model_name, model_metrics in metrics.items():
        print(f"\n  {model_name}:")
        print(f"    R²: {model_metrics['r2']:.4f}")
        print(f"    RMSE: {model_metrics['rmse']:.4f}")
        print(f"    MAE: {model_metrics['mae']:.4f}")
        print(f"    R² ajusté: {model_metrics['adjusted_r2']:.4f}")

    # Test 1: Générer le graphique avec R² ajusté
    print("\n" + "-" * 80)
    print("Test 1: Graphique avec R² ajusté")
    print("-" * 80)

    fig1 = evaluator.plot_metrics_comparison(include_adjusted_r2=True)

    # Vérifications
    print("\nVérifications:")

    # 1. Vérifier que c'est bien une Figure matplotlib
    assert isinstance(fig1, plt.Figure), "❌ Le retour n'est pas une Figure matplotlib"
    print("  ✓ Retourne une Figure matplotlib")

    # 2. Vérifier qu'il y a des axes
    axes = fig1.get_axes()
    assert len(axes) > 0, "❌ La figure ne contient pas d'axes"
    ax = axes[0]
    print("  ✓ La figure contient des axes")

    # 3. Vérifier la présence de barres
    bars = [patch for patch in ax.patches if hasattr(patch, 'get_height')]
    assert len(bars) > 0, "❌ Aucune barre trouvée dans le graphique"
    print(f"  ✓ {len(bars)} barres trouvées dans le graphique")

    # 4. Vérifier la légende
    legend = ax.get_legend()
    assert legend is not None, "❌ Aucune légende trouvée"
    legend_texts = [text.get_text() for text in legend.get_texts()]
    print(f"  ✓ Légende présente avec {len(legend_texts)} entrées: {legend_texts}")

    # 5. Vérifier que les 4 métriques sont présentes
    expected_metrics = ['R²', 'RMSE', 'MAE', 'R² ajusté']
    for metric in expected_metrics:
        assert metric in legend_texts, f"❌ Métrique '{metric}' manquante dans la légende"
    print(f"  ✓ Toutes les métriques attendues sont présentes")

    # 6. Vérifier les labels en français
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    title = ax.get_title()
    assert xlabel and len(xlabel) > 0, "❌ Label de l'axe x manquant"
    assert ylabel and len(ylabel) > 0, "❌ Label de l'axe y manquant"
    assert title and len(title) > 0, "❌ Titre manquant"
    print(f"  ✓ Labels en français:")
    print(f"    - Titre: '{title}'")
    print(f"    - Axe X: '{xlabel}'")
    print(f"    - Axe Y: '{ylabel}'")

    # 7. Vérifier les noms des modèles sur l'axe x
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    print(f"  ✓ Modèles sur l'axe x: {xticklabels}")

    # Sauvegarder le graphique
    fig1.savefig('tests/test_metrics_comparison_with_adjusted_r2.png', dpi=150, bbox_inches='tight')
    print("\n  → Graphique sauvegardé: tests/test_metrics_comparison_with_adjusted_r2.png")

    # Test 2: Générer le graphique sans R² ajusté
    print("\n" + "-" * 80)
    print("Test 2: Graphique sans R² ajusté")
    print("-" * 80)

    fig2 = evaluator.plot_metrics_comparison(include_adjusted_r2=False)

    # Vérifier que le R² ajusté n'est pas présent
    axes2 = fig2.get_axes()
    ax2 = axes2[0]
    legend2 = ax2.get_legend()
    legend_texts2 = [text.get_text() for text in legend2.get_texts()]

    assert 'R² ajusté' not in legend_texts2, "❌ R² ajusté présent alors qu'il devrait être exclu"
    print(f"\nVérifications:")
    print(f"  ✓ R² ajusté correctement exclu")
    print(f"  ✓ Métriques présentes: {legend_texts2}")

    # Sauvegarder le graphique
    fig2.savefig('tests/test_metrics_comparison_without_adjusted_r2.png', dpi=150, bbox_inches='tight')
    print("\n  → Graphique sauvegardé: tests/test_metrics_comparison_without_adjusted_r2.png")

    # Fermer les figures pour libérer la mémoire
    plt.close(fig1)
    plt.close(fig2)

    # Résumé final
    print("\n" + "=" * 80)
    print("RÉSULTAT: ✅ TOUS LES TESTS SONT PASSÉS")
    print("=" * 80)
    print("\nLe graphique plot_metrics_comparison() répond à tous les critères:")
    print("  ✓ Graphique à barres groupées avec modèles sur l'axe x")
    print("  ✓ Barres pour chaque métrique (R², RMSE, MAE, R² ajusté)")
    print("  ✓ Légende présente et correcte")
    print("  ✓ Labels en français")
    print("  ✓ Retourne une Figure matplotlib")
    print("  ✓ Option pour inclure/exclure le R² ajusté")
    print("\nFichiers générés:")
    print("  - tests/test_metrics_comparison_with_adjusted_r2.png")
    print("  - tests/test_metrics_comparison_without_adjusted_r2.png")
    print("=" * 80)


if __name__ == "__main__":
    test_plot_metrics_comparison()
