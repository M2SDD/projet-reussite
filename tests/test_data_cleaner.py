#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for DataCleaner.

Tests cover:
- remove_duplicates: suppression, warning, reset index, config DUPLICATE_SUBSET
- remove_outliers_iqr: suppression outliers, alignement X/y, ValueError, warning, threshold
- remove_low_variance_features: suppression variance nulle, warning, threshold
- remove_highly_correlated_features: suppression corrélation parfaite, warning, threshold
- select_top_features_linear: k features, subset, k=0, k > n_features
- select_top_features_mutual_info: idem
- select_features_by_importance: idem
- select_features_rfe: idem
- select_features (routeur): string→list, union méthodes, méthode inconnue, préfiltres
"""

import pytest
import warnings
import pandas as pd
import numpy as np

from src import Config
from src.data.data_cleaner import DataCleaner


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cleaner():
    return DataCleaner()


@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame({
        'pseudo': [1, 1, 2],
        'note':   [10.0, 10.0, 15.0],
    })


@pytest.fixture
def df_clean():
    return pd.DataFrame({
        'pseudo': [1, 2, 3],
        'note':   [10.0, 15.0, 12.0],
    })


@pytest.fixture
def X_with_outlier():
    """
    f1 : valeur 100 est un outlier clair (IQR sur [1,2,3,4,100]).
      Q1=2, Q3=4, IQR=2 → borne sup = 4 + 1.5*2 = 7 → 100 > 7
    f2 : pas d'outlier.
    La ligne index=4 (f1=100) doit être supprimée.
    """
    X = pd.DataFrame({
        'f1': [1.0, 2.0, 3.0, 4.0, 100.0],
        'f2': [1.0, 2.0, 3.0, 4.0,   5.0],
    })
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    return X, y


@pytest.fixture
def X_clean_no_outliers():
    """Données uniformément distribuées, aucun outlier attendu."""
    np.random.seed(0)
    X = pd.DataFrame({'f1': np.linspace(1, 10, 20), 'f2': np.linspace(2, 5, 20)})
    y = pd.Series(np.linspace(1, 20, 20))
    return X, y


@pytest.fixture
def X_zero_variance():
    """f_const a une variance nulle, f_varying doit être conservée."""
    return pd.DataFrame({
        'f_const':   [5.0] * 20,
        'f_varying': np.linspace(1.0, 10.0, 20),
    })


@pytest.fixture
def X_correlated():
    """
    f2 est une transformation linéaire parfaite de f1 (corr=1.0).
    f3 est indépendant.
    → f2 doit être supprimée (triangle supérieur → droite du pair).
    """
    base = np.linspace(1.0, 10.0, 30)
    np.random.seed(0)
    return pd.DataFrame({
        'f1': base,
        'f2': 3.0 * base + 1.0,          # parfaitement corrélé à f1
        'f3': np.random.rand(30) * 10.0,  # indépendant
    })


@pytest.fixture
def X_y_for_selection():
    """
    Dataset 50 lignes × 5 features.
    'relevant1' et 'relevant2' expliquent y linéairement.
    'noise1/2/3' sont du bruit pur.
    """
    np.random.seed(42)
    n = 50
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    X = pd.DataFrame({
        'relevant1': r1,
        'relevant2': r2,
        'noise1': np.random.rand(n),
        'noise2': np.random.rand(n),
        'noise3': np.random.rand(n),
    })
    y = pd.Series(2.0 * r1 + 3.0 * r2 + 0.05 * np.random.rand(n))
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Tests : remove_duplicates
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveDuplicates:
    def test_removes_duplicate_rows(self, cleaner, df_with_duplicates):
        result = cleaner.remove_duplicates(df_with_duplicates)
        assert len(result) == 2

    def test_issues_warning_when_duplicates_found(self, cleaner, df_with_duplicates):
        with pytest.warns(UserWarning, match="dupliquées"):
            cleaner.remove_duplicates(df_with_duplicates)

    def test_no_warning_when_no_duplicates(self, cleaner, df_clean):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cleaner.remove_duplicates(df_clean)

    def test_resets_index(self, cleaner, df_with_duplicates):
        result = cleaner.remove_duplicates(df_with_duplicates)
        assert list(result.index) == list(range(len(result)))

    def test_preserves_non_duplicate_rows(self, cleaner, df_clean):
        result = cleaner.remove_duplicates(df_clean)
        assert len(result) == len(df_clean)

    def test_duplicate_subset_config(self):
        """Avec DUPLICATE_SUBSET=['pseudo'], seul le pseudo détermine le doublon."""
        config = Config()
        config.DUPLICATE_SUBSET = ['pseudo']
        cleaner = DataCleaner(config=config)
        df = pd.DataFrame({
            'pseudo': [1, 1],
            'note':   [10.0, 15.0],  # notes différentes, même pseudo
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cleaner.remove_duplicates(df)
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests : remove_outliers_iqr
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveOutliersIqr:
    def test_returns_tuple_of_two(self, cleaner, X_with_outlier):
        result = cleaner.remove_outliers_iqr(*X_with_outlier)
        assert len(result) == 2

    def test_returns_dataframe_and_series(self, cleaner, X_with_outlier):
        X_clean, y_clean = cleaner.remove_outliers_iqr(*X_with_outlier)
        assert isinstance(X_clean, pd.DataFrame)
        assert isinstance(y_clean, pd.Series)

    def test_removes_outlier_row(self, cleaner, X_with_outlier):
        """La ligne avec f1=100 doit être supprimée."""
        X, y = X_with_outlier
        X_clean, _ = cleaner.remove_outliers_iqr(X, y, strategy='any')
        assert 100.0 not in X_clean['f1'].values

    def test_x_and_y_stay_aligned(self, cleaner, X_with_outlier):
        """X_clean et y_clean doivent avoir le même nombre de lignes."""
        X_clean, y_clean = cleaner.remove_outliers_iqr(*X_with_outlier)
        assert len(X_clean) == len(y_clean)

    def test_x_and_y_share_same_index(self, cleaner, X_with_outlier):
        """Les index de X_clean et y_clean doivent correspondre."""
        X_clean, y_clean = cleaner.remove_outliers_iqr(*X_with_outlier)
        assert list(X_clean.index) == list(y_clean.index)

    def test_raises_value_error_on_size_mismatch(self, cleaner):
        X = pd.DataFrame({'f1': [1.0, 2.0, 3.0]})
        y = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError):
            cleaner.remove_outliers_iqr(X, y)

    def test_issues_warning_when_outliers_removed(self, cleaner, X_with_outlier):
        with pytest.warns(UserWarning, match="IQR"):
            cleaner.remove_outliers_iqr(*X_with_outlier, strategy='any')

    def test_no_warning_when_no_outliers(self, cleaner, X_clean_no_outliers):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cleaner.remove_outliers_iqr(*X_clean_no_outliers)

    def test_stricter_threshold_removes_more_rows(self, cleaner):
        """Un seuil plus bas (0.5) doit supprimer plus de lignes que le seuil par défaut (1.5)."""
        X = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0, 6.0]})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_default, _ = cleaner.remove_outliers_iqr(X, y, threshold=1.5)
            X_strict,  _ = cleaner.remove_outliers_iqr(X, y, threshold=0.5)
        assert len(X_strict) <= len(X_default)

    def test_strategy_majority_is_default(self, cleaner, X_with_outlier):
        """Sans préciser strategy, le comportement doit être identique à strategy='any'."""
        X, y = X_with_outlier
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_default, _ = cleaner.remove_outliers_iqr(X, y)
            X_any,     _ = cleaner.remove_outliers_iqr(X, y, strategy='majority')
        pd.testing.assert_frame_equal(X_default, X_any)

    def test_strategy_all_more_permissive_than_any(self, cleaner):
        """strategy='all' ne supprime que si TOUTES les features sont outliers → plus permissif."""
        X = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 100.0],   # f1=100 est outlier
            'f2': [1.0, 2.0, 3.0, 4.0,   5.0],    # f2=5  n'est PAS outlier
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_any, _ = cleaner.remove_outliers_iqr(X, y, strategy='any')
            X_all, _ = cleaner.remove_outliers_iqr(X, y, strategy='all')
        # 'any' supprime la ligne (f1=100 outlier), 'all' la conserve (f2 est dans les bornes)
        assert len(X_all) > len(X_any)

    def test_strategy_majority(self, cleaner):
        """strategy='majority' supprime si > 50% des features sont outliers."""
        X = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 100.0],  # outlier
            'f2': [1.0, 2.0, 3.0, 4.0, 200.0],  # outlier
            'f3': [1.0, 2.0, 3.0, 4.0,   5.0],  # pas outlier
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_majority, _ = cleaner.remove_outliers_iqr(X, y, strategy='majority')
        # 2/3 features outliers (> 50%) → ligne supprimée
        assert len(X_majority) == 4

    def test_strategy_invalid_raises_value_error(self, cleaner, X_with_outlier):
        """Une strategy inconnue doit lever une ValueError."""
        with pytest.raises(ValueError, match="strategy"):
            cleaner.remove_outliers_iqr(*X_with_outlier, strategy='unknown')


# ─────────────────────────────────────────────────────────────────────────────
# Tests : remove_low_variance_features
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveLowVarianceFeatures:
    def test_returns_dataframe(self, cleaner, X_zero_variance):
        result = cleaner.remove_low_variance_features(X_zero_variance)
        assert isinstance(result, pd.DataFrame)

    def test_removes_zero_variance_column(self, cleaner, X_zero_variance):
        result = cleaner.remove_low_variance_features(X_zero_variance)
        assert 'f_const' not in result.columns

    def test_keeps_high_variance_column(self, cleaner, X_zero_variance):
        result = cleaner.remove_low_variance_features(X_zero_variance)
        assert 'f_varying' in result.columns

    def test_issues_warning_when_features_removed(self, cleaner, X_zero_variance):
        with pytest.warns(UserWarning, match="variance"):
            cleaner.remove_low_variance_features(X_zero_variance)

    def test_no_warning_when_all_features_kept(self, cleaner):
        X = pd.DataFrame({'f1': np.linspace(1, 10, 20), 'f2': np.linspace(2, 8, 20)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cleaner.remove_low_variance_features(X)

    def test_custom_threshold(self, cleaner):
        """Avec un seuil élevé, une feature à faible variance doit être supprimée."""
        X = pd.DataFrame({
            'low_var':  np.full(20, 5.0) + np.random.rand(20) * 0.001,
            'high_var': np.linspace(1.0, 100.0, 20),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cleaner.remove_low_variance_features(X, threshold=0.1)
        assert 'low_var' not in result.columns
        assert 'high_var' in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests : remove_highly_correlated_features
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveHighlyCorrelatedFeatures:
    def test_returns_dataframe(self, cleaner, X_correlated):
        result = cleaner.remove_highly_correlated_features(X_correlated)
        assert isinstance(result, pd.DataFrame)

    def test_removes_lower_variance_column(self, cleaner, X_correlated):
        """f2 = 3*f1+1 → var(f2) > var(f1) → f1 doit être supprimée, f2 conservée."""
        result = cleaner.remove_highly_correlated_features(X_correlated)
        # f2 a une variance 9× plus grande que f1 (var(3x+1) = 9*var(x))
        assert 'f1' not in result.columns
        assert 'f2' in result.columns

    def test_keeps_independent_column(self, cleaner, X_correlated):
        """f3 est indépendant → doit toujours être présent."""
        result = cleaner.remove_highly_correlated_features(X_correlated)
        assert 'f3' in result.columns

    def test_issues_warning_when_features_removed(self, cleaner, X_correlated):
        with pytest.warns(UserWarning, match="corrélées"):
            cleaner.remove_highly_correlated_features(X_correlated)

    def test_no_warning_when_no_correlation(self, cleaner):
        np.random.seed(0)
        X = pd.DataFrame({
            'f1': np.random.rand(30),
            'f2': np.random.rand(30),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cleaner.remove_highly_correlated_features(X)

    def test_custom_threshold(self, cleaner):
        """Un seuil de 0.5 doit supprimer des features modérément corrélées."""
        np.random.seed(42)
        base = np.random.rand(30)
        X = pd.DataFrame({
            'f1': base,
            'f2': 0.8 * base + 0.2 * np.random.rand(30),  # corr ≈ 0.9
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cleaner.remove_highly_correlated_features(X, threshold=0.5)
        assert len(result.columns) == 1

    def test_single_column_not_removed(self, cleaner):
        """Un seul DataFrame à 1 colonne ne doit rien supprimer."""
        X = pd.DataFrame({'f1': np.linspace(1.0, 10.0, 20)})
        result = cleaner.remove_highly_correlated_features(X)
        assert list(result.columns) == ['f1']


# ─────────────────────────────────────────────────────────────────────────────
# Tests : select_top_features_linear
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectTopFeaturesLinear:
    def test_returns_tuple_of_two(self, cleaner, X_y_for_selection):
        result = cleaner.select_top_features_linear(*X_y_for_selection, k=2)
        assert len(result) == 2

    def test_returns_dataframe_and_list(self, cleaner, X_y_for_selection):
        X_sel, features = cleaner.select_top_features_linear(*X_y_for_selection, k=2)
        assert isinstance(X_sel, pd.DataFrame)
        assert isinstance(features, list)

    def test_returns_exactly_k_features(self, cleaner, X_y_for_selection):
        _, features = cleaner.select_top_features_linear(*X_y_for_selection, k=3)
        assert len(features) == 3

    def test_selected_features_are_subset(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        _, features = cleaner.select_top_features_linear(X, y, k=2)
        assert set(features).issubset(set(X.columns))

    def test_dataframe_columns_match_features_list(self, cleaner, X_y_for_selection):
        X_sel, features = cleaner.select_top_features_linear(*X_y_for_selection, k=2)
        assert list(X_sel.columns) == features

    def test_k_capped_to_n_features(self, cleaner, X_y_for_selection):
        """k > n_features → retourne toutes les features disponibles."""
        X, y = X_y_for_selection
        _, features = cleaner.select_top_features_linear(X, y, k=999)
        assert len(features) == X.shape[1]

    def test_raises_value_error_for_k_zero(self, cleaner, X_y_for_selection):
        with pytest.raises(ValueError):
            cleaner.select_top_features_linear(*X_y_for_selection, k=0)

    def test_raises_value_error_for_k_negative(self, cleaner, X_y_for_selection):
        with pytest.raises(ValueError):
            cleaner.select_top_features_linear(*X_y_for_selection, k=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : select_top_features_mutual_info
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectTopFeaturesMutualInfo:
    def test_returns_exactly_k_features(self, cleaner, X_y_for_selection):
        _, features = cleaner.select_top_features_mutual_info(*X_y_for_selection, k=2)
        assert len(features) == 2

    def test_selected_features_are_subset(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        _, features = cleaner.select_top_features_mutual_info(X, y, k=2)
        assert set(features).issubset(set(X.columns))

    def test_k_capped_to_n_features(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        _, features = cleaner.select_top_features_mutual_info(X, y, k=999)
        assert len(features) == X.shape[1]

    def test_raises_value_error_for_k_zero(self, cleaner, X_y_for_selection):
        with pytest.raises(ValueError):
            cleaner.select_top_features_mutual_info(*X_y_for_selection, k=0)

    def test_reproducibility_with_random_state(self, cleaner, X_y_for_selection):
        """Deux appels successifs donnent les mêmes features (random_state fixé)."""
        _, features1 = cleaner.select_top_features_mutual_info(*X_y_for_selection, k=3)
        _, features2 = cleaner.select_top_features_mutual_info(*X_y_for_selection, k=3)
        assert features1 == features2


# ─────────────────────────────────────────────────────────────────────────────
# Tests : select_features_by_importance
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectFeaturesByImportance:
    def test_returns_exactly_k_features(self, cleaner, X_y_for_selection):
        _, features = cleaner.select_features_by_importance(*X_y_for_selection, k=2)
        assert len(features) == 2

    def test_selected_features_are_subset(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        _, features = cleaner.select_features_by_importance(X, y, k=2)
        assert set(features).issubset(set(X.columns))

    def test_k_capped_to_n_features(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        _, features = cleaner.select_features_by_importance(X, y, k=999)
        assert len(features) == X.shape[1]

    def test_raises_value_error_for_k_zero(self, cleaner, X_y_for_selection):
        with pytest.raises(ValueError):
            cleaner.select_features_by_importance(*X_y_for_selection, k=0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : select_features_rfe
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectFeaturesRfe:
    def test_returns_exactly_k_features(self, cleaner, X_y_for_selection):
        _, features = cleaner.select_features_rfe(*X_y_for_selection, k=2)
        assert len(features) == 2

    def test_selected_features_are_subset(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        _, features = cleaner.select_features_rfe(X, y, k=2)
        assert set(features).issubset(set(X.columns))

    def test_k_capped_to_n_features(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        _, features = cleaner.select_features_rfe(X, y, k=999)
        assert len(features) == X.shape[1]

    def test_raises_value_error_for_k_zero(self, cleaner, X_y_for_selection):
        with pytest.raises(ValueError):
            cleaner.select_features_rfe(*X_y_for_selection, k=0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : select_features (routeur)
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectFeatures:
    def test_string_method_converted_to_list(self, cleaner, X_y_for_selection):
        """Une méthode passée en string doit fonctionner comme une liste."""
        X_sel, features = cleaner.select_features(*X_y_for_selection, methods='linear', k=2)
        assert len(features) >= 1

    def test_unknown_method_raises_value_error(self, cleaner, X_y_for_selection):
        with pytest.raises(ValueError, match="inconnue"):
            cleaner.select_features(*X_y_for_selection, methods='unknown_method', k=2)

    def test_result_is_subset_of_original_columns(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        X_sel, features = cleaner.select_features(X, y, methods=['linear'], k=2)
        assert set(features).issubset(set(X.columns))

    def test_multiple_methods_union(self, cleaner, X_y_for_selection):
        """L'union de deux méthodes doit retourner au moins autant de features qu'une seule."""
        X, y = X_y_for_selection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, f_single = cleaner.select_features(X, y, methods=['linear'], k=2,
                                                  prefilter_variance=False,
                                                  prefilter_correlation=False)
            _, f_union = cleaner.select_features(X, y, methods=['linear', 'mutual_info'], k=2,
                                                 prefilter_variance=False,
                                                 prefilter_correlation=False)
        assert len(f_union) >= len(f_single)

    def test_result_is_sorted(self, cleaner, X_y_for_selection):
        """Les features sélectionnées doivent être triées alphabétiquement."""
        X, y = X_y_for_selection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, features = cleaner.select_features(X, y, methods=['linear', 'mutual_info'], k=2,
                                                  prefilter_variance=False,
                                                  prefilter_correlation=False)
        assert features == sorted(features)

    def test_dataframe_columns_match_features_list(self, cleaner, X_y_for_selection):
        X, y = X_y_for_selection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_sel, features = cleaner.select_features(X, y, methods='linear', k=2,
                                                      prefilter_variance=False,
                                                      prefilter_correlation=False)
        assert list(X_sel.columns) == features

    def test_prefilter_variance_removes_zero_variance(self, cleaner, X_y_for_selection):
        """Une colonne constante doit être retirée avant la sélection."""
        X, y = X_y_for_selection
        X_with_const = X.copy()
        X_with_const['constant'] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_sel, features = cleaner.select_features(
                X_with_const, y, methods='linear', k=2,
                prefilter_variance=True, prefilter_correlation=False
            )
        assert 'constant' not in features

    def test_prefilter_disabled_keeps_zero_variance(self, cleaner, X_y_for_selection):
        """Sans préfiltre, la colonne constante peut être sélectionnée."""
        X, y = X_y_for_selection
        X_with_const = X.copy()
        X_with_const['constant'] = 0.0
        # On sélectionne assez de features pour potentiellement inclure 'constant'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, features_no_filter = cleaner.select_features(
                X_with_const, y, methods='linear', k=X_with_const.shape[1],
                prefilter_variance=False, prefilter_correlation=False
            )
        # Avec k = n_features et sans préfiltre, toutes les features sont retournées
        assert len(features_no_filter) == X_with_const.shape[1]
