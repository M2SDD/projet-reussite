#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for DatasetBuilder.

Tests cover:
- Initialization and dependency injection
- build_dataset: inner join (drop_inactive_students=True)
- build_dataset: left join (drop_inactive_students=False)
- build_dataset: fillna(0) for inactive students
- build_dataset: feature selection bypass (selection_methods=None)
- build_dataset: remove_outliers=True branch
- build_dataset: return types
- get_train_test_split: correct shapes
- get_train_test_split: ValueError on None inputs
- get_train_test_split: respects TEST_SPLIT_RATIO from config
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src import Config
from src.data.dataset_builder import DatasetBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_logs_df():
    """Logs bruts minimalistes (5 événements, 2 étudiants)."""
    return pd.DataFrame({
        'heure': pd.to_datetime([
            '2024-07-24 09:48:08',
            '2024-07-24 10:00:00',
            '2024-08-19 12:55:34',
            '2024-08-19 14:00:00',
            '2024-09-01 08:00:00',
        ]),
        'pseudo': [436, 436, 841, 841, 436],
        'contexte': ['Cours: PASS - S1'] * 5,
        'composant': ['Système', 'Fichier', 'Système', 'Fichier', 'Système'],
        'evenement': ['Cours consulté', 'Module consulté'] * 2 + ['Cours consulté'],
    })


@pytest.fixture
def sample_notes_df():
    """Notes avec 3 étudiants (pseudo 318 est absent des logs)."""
    return pd.DataFrame({
        'pseudo': [436, 841, 318],
        'note': [11.05, 14.5, 10.0],
    })


@pytest.fixture
def sample_features_df():
    """Features extraites (index = pseudo)."""
    return pd.DataFrame({
        'actions_totales': [3, 2],
        'jours_actifs': [3, 1],
    }, index=pd.Index([436, 841], name='pseudo'))


@pytest.fixture
def builder_with_mocks(sample_logs_df, sample_notes_df, sample_features_df):
    """
    DatasetBuilder dont les 3 dépendances sont mockées.
    Retourne (builder, mock_loader, mock_extractor, mock_cleaner).
    """
    builder = DatasetBuilder()

    builder.loader = MagicMock()
    builder.loader.load_logs.return_value = sample_logs_df.copy()
    builder.loader.load_notes.return_value = sample_notes_df.copy()

    builder.extractor = MagicMock()
    builder.extractor.extract_features.return_value = sample_features_df.copy()

    builder.cleaner = MagicMock()
    # remove_duplicates retourne le df tel quel (pas de doublons dans les fixtures)
    builder.cleaner.remove_duplicates.side_effect = lambda df: df.copy()
    # select_features retourne X inchangé + toutes les colonnes
    builder.cleaner.select_features.side_effect = (
        lambda X, y, **kw: (X, X.columns.tolist())
    )

    return builder, builder.loader, builder.extractor, builder.cleaner


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_config_is_created(self):
        """Sans config, DatasetBuilder crée un Config() par défaut."""
        builder = DatasetBuilder()
        assert isinstance(builder.config, Config)

    def test_custom_config_is_used(self):
        """Avec un Config personnalisé, il est bien conservé."""
        custom_config = Config()
        custom_config.RANDOM_STATE = 99
        builder = DatasetBuilder(config=custom_config)
        assert builder.config.RANDOM_STATE == 99

    def test_sub_components_initialized(self):
        """Les trois composants sont bien instanciés."""
        from src.data.data_loader import DataLoader
        from src.data.feature_extractor import FeatureExtractor
        from src.data.data_cleaner import DataCleaner

        builder = DatasetBuilder()
        assert isinstance(builder.loader, DataLoader)
        assert isinstance(builder.extractor, FeatureExtractor)
        assert isinstance(builder.cleaner, DataCleaner)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : build_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_returns_correct_types(self, builder_with_mocks):
        """build_dataset retourne bien (DataFrame, Series, list)."""
        builder, *_ = builder_with_mocks
        X, y, features = builder.build_dataset('logs.csv', 'notes.csv')

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(features, list)

    def test_inner_join_drops_inactive_students(self, builder_with_mocks):
        """drop_inactive_students=True → inner join → pseudo 318 absent."""
        builder, *_ = builder_with_mocks
        X, y, _ = builder.build_dataset(
            'logs.csv', 'notes.csv', drop_inactive_students=True
        )
        assert 318 not in y.index
        assert len(y) == 2  # seulement 436 et 841

    def test_left_join_keeps_inactive_students(self, builder_with_mocks):
        """drop_inactive_students=False → left join → 3 lignes conservées."""
        builder, *_ = builder_with_mocks
        X, y, _ = builder.build_dataset(
            'logs.csv', 'notes.csv', drop_inactive_students=False
        )
        # Après le merge, l'index est un RangeIndex — on vérifie juste le nombre de lignes
        assert len(y) == 3

    def test_inactive_students_features_filled_with_zero(self, builder_with_mocks):
        """Les étudiants sans logs ont leurs features à 0 (pas NaN)."""
        builder, *_ = builder_with_mocks
        X, *_ = builder.build_dataset(
            'logs.csv', 'notes.csv', drop_inactive_students=False
        )
        # Pas de NaN dans X (le fillna(0) a bien été appliqué)
        assert not X.isnull().any().any()
        # Le pseudo 318 est le 3e dans notes_df → dernière ligne de X (index 2)
        # Ses features sont toutes à 0 car absent des logs
        assert (X.iloc[-1] == 0).all()

    def test_target_column_not_in_X(self, builder_with_mocks):
        """La colonne 'note' (TARGET_COLUMN) ne doit pas apparaître dans X."""
        builder, *_ = builder_with_mocks
        X, y, _ = builder.build_dataset('logs.csv', 'notes.csv')
        assert 'note' not in X.columns

    def test_merge_key_not_in_X(self, builder_with_mocks):
        """La colonne 'pseudo' (MERGE_KEY) ne doit pas apparaître dans X."""
        builder, *_ = builder_with_mocks
        X, y, _ = builder.build_dataset('logs.csv', 'notes.csv')
        assert 'pseudo' not in X.columns

    def test_selection_methods_none_skips_feature_selection(self, builder_with_mocks):
        """selection_methods=None → select_features n'est pas appelé."""
        builder, _, _, mock_cleaner = builder_with_mocks
        X, y, features = builder.build_dataset(
            'logs.csv', 'notes.csv', selection_methods=None
        )
        mock_cleaner.select_features.assert_not_called()
        # features = toutes les colonnes de X
        assert features == X.columns.tolist()

    def test_selection_methods_called_with_correct_params(self, builder_with_mocks):
        """select_features est appelé avec les bons paramètres."""
        builder, _, _, mock_cleaner = builder_with_mocks
        builder.build_dataset(
            'logs.csv', 'notes.csv',
            selection_methods=['linear'],
            k_features=5,
            prefilter_variance=False,
            prefilter_correlation=True
        )
        call_kwargs = mock_cleaner.select_features.call_args
        assert call_kwargs.kwargs['methods'] == ['linear']
        assert call_kwargs.kwargs['k'] == 5
        assert call_kwargs.kwargs['prefilter_variance'] is False
        assert call_kwargs.kwargs['prefilter_correlation'] is True

    def test_loader_called_with_correct_paths(self, builder_with_mocks):
        """load_logs et load_notes sont appelés avec les bons chemins."""
        builder, mock_loader, *_ = builder_with_mocks
        builder.build_dataset('path/to/logs.csv', 'path/to/notes.csv')
        mock_loader.load_logs.assert_called_once_with('path/to/logs.csv')
        mock_loader.load_notes.assert_called_once_with('path/to/notes.csv')

    def test_remove_duplicates_called_twice(self, builder_with_mocks):
        """remove_duplicates est appelé une fois pour les logs, une fois pour les notes."""
        builder, _, _, mock_cleaner = builder_with_mocks
        builder.build_dataset('logs.csv', 'notes.csv')
        assert mock_cleaner.remove_duplicates.call_count == 2

    def test_remove_outliers_false_by_default(self, builder_with_mocks):
        """remove_outliers_iqr ne doit pas être appelé par défaut."""
        builder, _, _, mock_cleaner = builder_with_mocks
        builder.build_dataset('logs.csv', 'notes.csv')
        mock_cleaner.remove_outliers_iqr.assert_not_called()

    def test_remove_outliers_called_when_enabled(self, builder_with_mocks):
        """remove_outliers=True → remove_outliers_iqr est appelé."""
        builder, _, _, mock_cleaner = builder_with_mocks
        # On mock le retour du tuple (X_clean, y_clean)
        mock_cleaner.remove_outliers_iqr.return_value = (
            pd.DataFrame({'actions_totales': [3], 'jours_actifs': [3]},
                         index=pd.Index([436])),
            pd.Series([11.05], index=pd.Index([436]))
        )
        builder.build_dataset('logs.csv', 'notes.csv', remove_outliers=True)
        mock_cleaner.remove_outliers_iqr.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Tests : get_train_test_split
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTrainTestSplit:
    @pytest.fixture
    def sample_X_y(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.rand(100), name='note')
        return X, y

    def test_returns_four_elements(self, sample_X_y):
        """Doit retourner exactement 4 éléments."""
        builder = DatasetBuilder()
        result = builder.get_train_test_split(*sample_X_y)
        assert len(result) == 4

    def test_split_sizes_respect_config(self, sample_X_y):
        """Le split doit respecter TEST_SPLIT_RATIO (0.2 par défaut → 80/20)."""
        builder = DatasetBuilder()
        X_train, X_test, y_train, y_test = builder.get_train_test_split(*sample_X_y)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_custom_split_ratio(self, sample_X_y):
        """Un config avec TEST_SPLIT_RATIO=0.3 doit donner un split 70/30."""
        config = Config()
        config.TEST_SPLIT_RATIO = 0.3
        builder = DatasetBuilder(config=config)
        X_train, X_test, y_train, y_test = builder.get_train_test_split(*sample_X_y)

        assert len(X_test) == 30
        assert len(X_train) == 70

    def test_raises_value_error_on_none_X(self):
        """ValueError si X est None."""
        builder = DatasetBuilder()
        y = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Les matrices X et y doivent être fournies"):
            builder.get_train_test_split(None, y)

    def test_raises_value_error_on_none_y(self):
        """ValueError si y est None."""
        builder = DatasetBuilder()
        X = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValueError, match="Les matrices X et y doivent être fournies"):
            builder.get_train_test_split(X, None)

    def test_reproducibility_with_random_state(self, sample_X_y):
        """Deux appels successifs avec le même random_state donnent le même split."""
        builder = DatasetBuilder()
        split1 = builder.get_train_test_split(*sample_X_y)
        split2 = builder.get_train_test_split(*sample_X_y)

        pd.testing.assert_frame_equal(split1[0], split2[0])  # X_train
        pd.testing.assert_series_equal(split1[2], split2[2])  # y_train