#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for FeatureExtractor.

Tests cover:
- compute_activity_metrics: volume, jours actifs, durée d'engagement, sessions
- compute_temporal_patterns: semaine/weekend, tranches horaires, heure de pointe
- compute_component_features: préfixe comp_, valeurs exactes
- compute_event_type_features: préfixe evt_, valeurs exactes
- compute_consistency_features: streak, écarts inter-sessions, fréquence
- compute_interaction_depth_features: actions/jour, diversité, switch rate
- extract_features: validation entrées, index, colonnes complètes, fillna, nettoyage df_logs
- Cas limites: étudiant unique, action unique, gap exact à la frontière de session
"""

import pytest
import pandas as pd
import numpy as np

from src import Config
from src.data.feature_extractor import FeatureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    return FeatureExtractor()


@pytest.fixture
def sample_logs():
    """
    Logs structurés avec valeurs calculables à la main.

    Étudiant 436 (3 actions) :
      - Lun 02/09 09:00 et 09:10 → même session (gap 10 min < 30 min)
      - Mar 03/09 15:00           → nouvelle session (gap >> 30 min)
      → actions_totales=3, jours_actifs=2, duree_engagement=1j, sessions=2
      → tout en semaine, jours consécutifs=2, heure_pointe=9
      → actions_matin=2, actions_aprem=1

    Étudiant 841 (2 actions) :
      - Mer 04/09 14:00 → semaine, après-midi
      - Sam 07/09 20:00 → weekend, soir
      → actions_totales=2, jours_actifs=2, duree_engagement=3j, sessions=2
      → ratio_activite_weekend=0.5, jours_consecutifs_max=1
    """
    return pd.DataFrame({
        'heure': pd.to_datetime([
            '2024-09-02 09:00:00',
            '2024-09-02 09:10:00',
            '2024-09-03 15:00:00',
            '2024-09-04 14:00:00',
            '2024-09-07 20:00:00',
        ]),
        'pseudo': [436, 436, 436, 841, 841],
        'contexte': [
            'Cours: PASS - S1', 'Fichier: Contrat', 'Cours: PASS - S1',
            'Cours: PASS - S1', 'Forum: Discussion',
        ],
        'composant': ['Système', 'Fichier', 'Système', 'Système', 'Forum'],
        'evenement': [
            'Cours consulté', 'Module consulté', 'Cours consulté',
            'Cours consulté', 'Discussion consultée',
        ],
    })


@pytest.fixture
def single_action_logs():
    """Un seul étudiant, une seule action — cas limite minimum."""
    return pd.DataFrame({
        'heure': pd.to_datetime(['2024-09-02 10:00:00']),
        'pseudo': [999],
        'contexte': ['Cours: PASS - S1'],
        'composant': ['Système'],
        'evenement': ['Cours consulté'],
    })


@pytest.fixture
def session_boundary_logs():
    """
    Deux actions avec un gap exactement à la frontière (SESSION_GAP_MINUTES = 30).
    - Gap de 29 min → même session (< 30 min)
    - Gap de 31 min → nouvelle session (> 30 min)
    """
    return pd.DataFrame({
        'heure': pd.to_datetime([
            '2024-09-02 10:00:00',
            '2024-09-02 10:29:00',  # gap 29 min → même session
            '2024-09-02 11:00:00',  # gap 31 min → nouvelle session
        ]),
        'pseudo': [1, 1, 1],
        'contexte': ['Cours: PASS - S1'] * 3,
        'composant': ['Système'] * 3,
        'evenement': ['Cours consulté'] * 3,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_activity_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeActivityMetrics:
    def test_returns_dataframe(self, extractor, sample_logs):
        result = extractor.compute_activity_metrics(sample_logs)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_pseudo(self, extractor, sample_logs):
        result = extractor.compute_activity_metrics(sample_logs)
        assert result.index.name == 'pseudo'
        assert set(result.index) == {436, 841}

    def test_expected_columns_present(self, extractor, sample_logs):
        result = extractor.compute_activity_metrics(sample_logs)
        expected = {'actions_totales', 'jours_actifs', 'duree_engagement_jours', 'nombre_sessions'}
        assert expected.issubset(result.columns)

    def test_actions_totales(self, extractor, sample_logs):
        result = extractor.compute_activity_metrics(sample_logs)
        assert result.loc[436, 'actions_totales'] == 3
        assert result.loc[841, 'actions_totales'] == 2

    def test_jours_actifs(self, extractor, sample_logs):
        result = extractor.compute_activity_metrics(sample_logs)
        assert result.loc[436, 'jours_actifs'] == 2  # Sep 2 et Sep 3
        assert result.loc[841, 'jours_actifs'] == 2  # Sep 4 et Sep 7

    def test_duree_engagement(self, extractor, sample_logs):
        result = extractor.compute_activity_metrics(sample_logs)
        assert result.loc[436, 'duree_engagement_jours'] == 1  # Sep 3 - Sep 2
        assert result.loc[841, 'duree_engagement_jours'] == 3  # Sep 7 - Sep 4

    def test_duree_engagement_single_action(self, extractor, single_action_logs):
        """Un seul événement → durée d'engagement = 0."""
        result = extractor.compute_activity_metrics(single_action_logs)
        assert result.loc[999, 'duree_engagement_jours'] == 0

    def test_nombre_sessions_two_consecutive_actions(self, extractor, sample_logs):
        """436 : 09:00 et 09:10 → même session, puis 15:00 → nouvelle session."""
        result = extractor.compute_activity_metrics(sample_logs)
        assert result.loc[436, 'nombre_sessions'] == 2

    def test_nombre_sessions_single_action(self, extractor, single_action_logs):
        """Une seule action → 1 session."""
        result = extractor.compute_activity_metrics(single_action_logs)
        assert result.loc[999, 'nombre_sessions'] == 1

    def test_session_boundary_below_threshold(self, extractor, session_boundary_logs):
        """Gap de 29 min < 30 min (seuil) → 2 sessions (pas 3)."""
        result = extractor.compute_activity_metrics(session_boundary_logs)
        assert result.loc[1, 'nombre_sessions'] == 2

    def test_session_boundary_above_threshold(self, extractor):
        """Gap de 31 min > 30 min → chaque action crée une nouvelle session."""
        logs = pd.DataFrame({
            'heure': pd.to_datetime([
                '2024-09-02 10:00:00',
                '2024-09-02 10:31:00',
                '2024-09-02 11:02:00',
            ]),
            'pseudo': [1, 1, 1],
            'contexte': ['Cours'] * 3,
            'composant': ['Système'] * 3,
            'evenement': ['Cours consulté'] * 3,
        })
        result = extractor.compute_activity_metrics(logs)
        assert result.loc[1, 'nombre_sessions'] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_temporal_patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTemporalPatterns:
    def test_returns_dataframe(self, extractor, sample_logs):
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert isinstance(result, pd.DataFrame)

    def test_index_is_pseudo(self, extractor, sample_logs):
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert result.index.name == 'pseudo'

    def test_expected_columns_present(self, extractor, sample_logs):
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        expected = {
            'heure_pointe', 'actions_weekend', 'actions_semaine',
            'ratio_activite_weekend', 'actions_matin', 'actions_aprem',
            'actions_soir', 'actions_nuit',
        }
        assert expected.issubset(result.columns)

    def test_actions_semaine_weekend(self, extractor, sample_logs):
        """436 : 3 actions en semaine, 0 le weekend."""
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert result.loc[436, 'actions_semaine'] == 3
        assert result.loc[436, 'actions_weekend'] == 0

    def test_weekend_action_detected(self, extractor, sample_logs):
        """841 : 1 action en semaine, 1 le weekend (samedi)."""
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert result.loc[841, 'actions_weekend'] == 1
        assert result.loc[841, 'actions_semaine'] == 1

    def test_ratio_activite_weekend(self, extractor, sample_logs):
        """436 : ratio=0 (aucun weekend). 841 : ratio=0.5 (1/2 actions le weekend)."""
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert result.loc[436, 'ratio_activite_weekend'] == 0.0
        assert result.loc[841, 'ratio_activite_weekend'] == pytest.approx(0.5)

    def test_ratio_between_zero_and_one(self, extractor, sample_logs):
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert (result['ratio_activite_weekend'] >= 0).all()
        assert (result['ratio_activite_weekend'] <= 1).all()

    def test_heure_pointe(self, extractor, sample_logs):
        """436 : 9h (2 actions à 09:00 et 09:10) → heure_pointe=9."""
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert result.loc[436, 'heure_pointe'] == 9

    def test_heure_pointe_range(self, extractor, sample_logs):
        """heure_pointe doit être entre 0 et 23."""
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert (result['heure_pointe'] >= 0).all()
        assert (result['heure_pointe'] <= 23).all()

    def test_time_slots(self, extractor, sample_logs):
        """436 : 2 actions le matin (09h), 1 l'aprem (15h)."""
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert result.loc[436, 'actions_matin'] == 2
        assert result.loc[436, 'actions_aprem'] == 1
        assert result.loc[436, 'actions_soir'] == 0
        assert result.loc[436, 'actions_nuit'] == 0

    def test_time_slot_soir(self, extractor, sample_logs):
        """841 : 1 action l'aprem (14h), 1 action le soir (20h)."""
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert result.loc[841, 'actions_aprem'] == 1
        assert result.loc[841, 'actions_soir'] == 1

    def test_no_nan_in_result(self, extractor, sample_logs):
        result = extractor.compute_temporal_patterns(sample_logs.copy())
        assert not result.isnull().any().any()

    def test_does_not_mutate_input_df(self, extractor, sample_logs):
        """La méthode ne doit pas modifier le DataFrame d'entrée."""
        logs = sample_logs.copy()
        original_columns = logs.columns.tolist()
        extractor.compute_temporal_patterns(logs)
        assert logs.columns.tolist() == original_columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_component_features
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeComponentFeatures:
    def test_returns_dataframe(self, extractor, sample_logs):
        result = extractor.compute_component_features(sample_logs)
        assert isinstance(result, pd.DataFrame)

    def test_columns_have_comp_prefix(self, extractor, sample_logs):
        result = extractor.compute_component_features(sample_logs)
        assert all(col.startswith('comp_') for col in result.columns)

    def test_column_names_lowercased(self, extractor, sample_logs):
        result = extractor.compute_component_features(sample_logs)
        for col in result.columns:
            assert col == col.lower()

    def test_column_spaces_replaced_by_underscore(self, extractor):
        logs = pd.DataFrame({
            'heure': pd.to_datetime(['2024-01-01 10:00:00']),
            'pseudo': [1],
            'composant': ['Mon Composant'],
            'evenement': ['evt'],
            'contexte': ['ctx'],
        })
        result = extractor.compute_component_features(logs)
        assert 'comp_mon_composant' in result.columns

    def test_component_counts(self, extractor, sample_logs):
        """436 : Système×2, Fichier×1. 841 : Système×1, Forum×1."""
        result = extractor.compute_component_features(sample_logs)
        assert result.loc[436, 'comp_système'] == 2
        assert result.loc[436, 'comp_fichier'] == 1
        assert result.loc[841, 'comp_forum'] == 1
        assert result.loc[841, 'comp_système'] == 1

    def test_zero_for_absent_component(self, extractor, sample_logs):
        """Un composant absent doit valoir 0 (crosstab, pas NaN)."""
        result = extractor.compute_component_features(sample_logs)
        assert result.loc[436, 'comp_forum'] == 0
        assert result.loc[841, 'comp_fichier'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_event_type_features
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEventTypeFeatures:
    def test_returns_dataframe(self, extractor, sample_logs):
        result = extractor.compute_event_type_features(sample_logs)
        assert isinstance(result, pd.DataFrame)

    def test_columns_have_evt_prefix(self, extractor, sample_logs):
        result = extractor.compute_event_type_features(sample_logs)
        assert all(col.startswith('evt_') for col in result.columns)

    def test_event_counts(self, extractor, sample_logs):
        """
        436 : 'Cours consulté'×2, 'Module consulté'×1.
        841 : 'Cours consulté'×1, 'Discussion consultée'×1.
        """
        result = extractor.compute_event_type_features(sample_logs)
        assert result.loc[436, 'evt_cours_consulté'] == 2
        assert result.loc[436, 'evt_module_consulté'] == 1
        assert result.loc[841, 'evt_discussion_consultée'] == 1

    def test_zero_for_absent_event(self, extractor, sample_logs):
        result = extractor.compute_event_type_features(sample_logs)
        assert result.loc[436, 'evt_discussion_consultée'] == 0
        assert result.loc[841, 'evt_module_consulté'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_consistency_features
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeConsistencyFeatures:
    def test_returns_dataframe(self, extractor, sample_logs):
        result = extractor.compute_consistency_features(sample_logs.copy())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, extractor, sample_logs):
        result = extractor.compute_consistency_features(sample_logs.copy())
        expected = {
            'jours_consecutifs_max', 'ecart_moyen_jours',
            'ecart_type_jours', 'jours_actifs_hebdos',
        }
        assert expected.issubset(result.columns)

    def test_streak_consecutive_days(self, extractor, sample_logs):
        """436 : Sep 2 et Sep 3 sont consécutifs → streak = 2."""
        result = extractor.compute_consistency_features(sample_logs.copy())
        assert result.loc[436, 'jours_consecutifs_max'] == 2

    def test_streak_non_consecutive_days(self, extractor, sample_logs):
        """841 : Sep 4 et Sep 7 ne sont pas consécutifs → streak = 1."""
        result = extractor.compute_consistency_features(sample_logs.copy())
        assert result.loc[841, 'jours_consecutifs_max'] == 1

    def test_streak_single_action(self, extractor, single_action_logs):
        """Un seul jour actif → streak = 1."""
        result = extractor.compute_consistency_features(single_action_logs.copy())
        assert result.loc[999, 'jours_consecutifs_max'] == 1

    def test_ecart_moyen(self, extractor, sample_logs):
        """436 : 1 seul écart (Sep 2→Sep 3 = 1j). 841 : 1 seul écart (Sep 4→Sep 7 = 3j)."""
        result = extractor.compute_consistency_features(sample_logs.copy())
        assert result.loc[436, 'ecart_moyen_jours'] == pytest.approx(1.0)
        assert result.loc[841, 'ecart_moyen_jours'] == pytest.approx(3.0)

    def test_ecart_moyen_single_day(self, extractor, single_action_logs):
        """Un seul jour → pas d'écart calculable → 0."""
        result = extractor.compute_consistency_features(single_action_logs.copy())
        assert result.loc[999, 'ecart_moyen_jours'] == 0

    def test_does_not_mutate_input_df(self, extractor, sample_logs):
        """La méthode ne doit pas modifier le DataFrame d'entrée."""
        logs = sample_logs.copy()
        original_columns = logs.columns.tolist()
        extractor.compute_consistency_features(logs)
        assert logs.columns.tolist() == original_columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_interaction_depth_features
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeInteractionDepthFeatures:
    def test_returns_dataframe(self, extractor, sample_logs):
        result = extractor.compute_interaction_depth_features(sample_logs)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, extractor, sample_logs):
        result = extractor.compute_interaction_depth_features(sample_logs)
        expected = {
            'actions_par_jour', 'diversite_composants',
            'diversite_contextes', 'interactions_moy_par_composant',
            'taux_changement_composant',
        }
        assert expected.issubset(result.columns)

    def test_actions_par_jour(self, extractor, sample_logs):
        """436 : 3 actions / 2 jours = 1.5. 841 : 2 actions / 2 jours = 1.0."""
        result = extractor.compute_interaction_depth_features(sample_logs)
        assert result.loc[436, 'actions_par_jour'] == pytest.approx(1.5)
        assert result.loc[841, 'actions_par_jour'] == pytest.approx(1.0)

    def test_diversite_composants(self, extractor, sample_logs):
        """436 : Système + Fichier = 2. 841 : Système + Forum = 2."""
        result = extractor.compute_interaction_depth_features(sample_logs)
        assert result.loc[436, 'diversite_composants'] == 2
        assert result.loc[841, 'diversite_composants'] == 2

    def test_taux_changement_composant_range(self, extractor, sample_logs):
        """Le taux de changement doit être entre 0 et 1."""
        result = extractor.compute_interaction_depth_features(sample_logs)
        assert (result['taux_changement_composant'] >= 0).all()
        assert (result['taux_changement_composant'] <= 1).all()

    def test_taux_changement_single_action(self, extractor, single_action_logs):
        """Une seule action → taux de changement = 0."""
        result = extractor.compute_interaction_depth_features(single_action_logs)
        assert result.loc[999, 'taux_changement_composant'] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : extract_features (orchestrateur)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_dataframe(self, extractor, sample_logs):
        result = extractor.extract_features(sample_logs.copy())
        assert isinstance(result, pd.DataFrame)

    def test_index_is_pseudo(self, extractor, sample_logs):
        result = extractor.extract_features(sample_logs.copy())
        assert result.index.name == 'pseudo'
        assert set(result.index) == {436, 841}

    def test_no_nan_in_result(self, extractor, sample_logs):
        """Le fillna(0) final doit éliminer tous les NaN."""
        result = extractor.extract_features(sample_logs.copy())
        assert not result.isnull().any().any()

    def test_all_sub_features_present(self, extractor, sample_logs):
        """Le résultat doit contenir les colonnes de chaque sous-méthode."""
        result = extractor.extract_features(sample_logs.copy())
        # Activity metrics
        assert 'actions_totales' in result.columns
        assert 'nombre_sessions' in result.columns
        # Temporal
        assert 'actions_semaine' in result.columns
        assert 'ratio_activite_weekend' in result.columns
        # Components et events
        assert any(col.startswith('comp_') for col in result.columns)
        assert any(col.startswith('evt_') for col in result.columns)
        # Consistency
        assert 'jours_consecutifs_max' in result.columns
        # Depth
        assert 'actions_par_jour' in result.columns

    def test_does_not_mutate_input_df(self, extractor, sample_logs):
        """extract_features ne doit pas modifier le DataFrame d'entrée."""
        logs = sample_logs.copy()
        original_columns = logs.columns.tolist()
        extractor.extract_features(logs)
        assert logs.columns.tolist() == original_columns

    def test_raises_value_error_on_empty_df(self, extractor):
        """Un DataFrame vide doit lever une ValueError."""
        empty_df = pd.DataFrame(columns=['heure', 'pseudo', 'contexte', 'composant', 'evenement'])
        with pytest.raises(ValueError, match="vide"):
            extractor.extract_features(empty_df)

    def test_raises_type_error_if_heure_not_datetime(self, extractor, sample_logs):
        """Une colonne 'heure' non-datetime doit lever une TypeError."""
        logs = sample_logs.copy()
        logs['heure'] = logs['heure'].astype(str)
        with pytest.raises(TypeError, match="datetime"):
            extractor.extract_features(logs)

    def test_result_has_correct_number_of_rows(self, extractor, sample_logs):
        """Une ligne par étudiant unique dans les logs."""
        result = extractor.extract_features(sample_logs.copy())
        assert len(result) == 2

    def test_single_student_single_action(self, extractor, single_action_logs):
        """Cas limite : 1 étudiant, 1 action — ne doit pas planter."""
        result = extractor.extract_features(single_action_logs.copy())
        assert len(result) == 1
        assert not result.isnull().any().any()
