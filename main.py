#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 08/03/2025
# version ='1.0'
# ----------------------------------------------------------------------------------------------------------------------
"""
DataVis sur les exports d'Arche anonymisés

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
import pandas as pd
from src import Config, DataLoader, DataProcessor, Visualizer


# ----------------------------------------------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------------------------------------------
# les constantes par défaut sont définies dans les src.*
LOGS_FILE = 'data/logs_info_25_pseudo.csv'
NOTES_FILE = 'data/notes_info_25_pseudo.csv'


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Initialisation de la configuration
    config = Config()

    # Initialisation des composants principaux
    data_loader = DataLoader()
    data_processor = DataProcessor()
    visualizer = Visualizer()

    # Vérification de l'architecture OOP
    print("✓ Tous les modules sont chargés avec succès")
    print(f"✓ Config: {type(config).__name__}")
    print(f"✓ DataLoader: {type(data_loader).__name__}")
    print(f"✓ DataProcessor: {type(data_processor).__name__}")
    print(f"✓ Visualizer: {type(visualizer).__name__}")
    print()


    # --------------------------------------------------------------------------
    # Data Loading
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Data Loading")
    print("=" * 60)
    print()

    # Initialize DataLoader
    loader = DataLoader()

    # Load logs data
    print("Loading ARCHE logs data...")
    logs_df = loader.load_logs(LOGS_FILE)
    print("Logs loaded successfully")
    print(f"  - Total log entries: {len(logs_df)}")
    print(f"  - Columns: {', '.join(logs_df.columns)}")
    print(f"  - Date range: {logs_df['heure'].min()} to {logs_df['heure'].max()}")
    print()

    # Load notes data
    print("Loading ARCHE notes data...")
    notes_df = loader.load_notes(NOTES_FILE)
    print("Notes loaded successfully")
    print(f"  - Total notes: {len(notes_df)}")
    print(f"  - Columns: {', '.join(notes_df.columns)}")
    print(f"  - Unique students: {notes_df['pseudo'].nunique()}")
    print()

    # Display sample data
    print("Sample log entries:")
    print(logs_df.head(3))
    print()
    print("Sample notes:")
    print(notes_df.head(3))
    print()

    # --------------------------------------------------------------------------
    # Data Cleaning & Preprocessing Pipeline
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Data Cleaning & Preprocessing Pipeline")
    print("=" * 60)
    print()

    # Build the full student dataset
    processor = DataProcessor(config=config)
    student_df = processor.build_student_dataset(logs_df, notes_df)

    # Display cleaning report
    report = processor.get_cleaning_report()
    print("Cleaning Report:")
    for key, value in report.items():
        print(f"  - {key}: {value}")
    print()

    # Display final student dataset
    print(f"Final student dataset shape: {student_df.shape}")
    print()
    print("Sample student data:")
    print(student_df.head(5))
    print()

    # --------------------------------------------------------------------------
    # Engagement Feature Engineering
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Engagement Feature Engineering")
    print("=" * 60)
    print()

    # Build comprehensive engagement features
    print("Computing engagement features from student activity logs...")
    engagement_df = processor.build_engagement_features(logs_df)
    print("Engagement features computed successfully")
    print()

    # Display engagement feature categories
    print("Engagement Feature Categories:")
    print("  - Activity Metrics: total_actions, unique_days_active, actions_per_day, session_count")
    print("  - Component Features: comp_* (interactions per component)")
    print("  - Event Type Features: *_count (views, submissions, forum, quiz, downloads)")
    print("  - Consistency Metrics: streak_days, avg_gap_days, std_gap_days, study_frequency")
    print("  - Interaction Depth: component_diversity, context_diversity, avg_interactions_per_component, component_switch_rate")
    print("  - Temporal Patterns: peak_hour, morning/afternoon/evening/night_activity, weekend_activity_ratio")
    print()

    # Display engagement statistics
    print(f"Total students with engagement features: {len(engagement_df)}")
    print(f"Total engagement features: {len(engagement_df.columns) - 1}")  # -1 for pseudo column
    print()

    # Display sample engagement features
    print("Sample engagement features (first 3 students):")
    display_cols = ['pseudo', 'total_actions', 'unique_days_active', 'actions_per_day',
                    'session_count', 'component_diversity', 'peak_hour', 'weekend_activity_ratio']
    available_cols = [col for col in display_cols if col in engagement_df.columns]
    print(engagement_df[available_cols].head(3))
    print()

    # Display engagement summary statistics
    print("Engagement Metrics Summary:")
    if 'total_actions' in engagement_df.columns:
        print(f"  - Avg total actions: {engagement_df['total_actions'].mean():.2f}")
    if 'unique_days_active' in engagement_df.columns:
        print(f"  - Avg unique days active: {engagement_df['unique_days_active'].mean():.2f}")
    if 'actions_per_day' in engagement_df.columns:
        print(f"  - Avg actions per day: {engagement_df['actions_per_day'].mean():.2f}")
    if 'component_diversity' in engagement_df.columns:
        print(f"  - Avg component diversity: {engagement_df['component_diversity'].mean():.2f}")
    if 'weekend_activity_ratio' in engagement_df.columns:
        print(f"  - Avg weekend activity ratio: {engagement_df['weekend_activity_ratio'].mean():.2f}")
    print()

    # --------------------------------------------------------------------------
    # Statistical Analysis
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Statistical Analysis")
    print("=" * 60)
    print()

    # Merge engagement features with student data for analysis
    print("Merging engagement features with student grades...")
    analysis_df = student_df.merge(engagement_df, on='pseudo', how='inner')
    print(f"Analysis dataset created: {analysis_df.shape[0]} students, {analysis_df.shape[1]} features")
    print()

    # Compute feature correlations with target (note)
    print("Computing feature correlations with target variable (note)...")
    feature_correlations = processor.compute_feature_correlations(analysis_df, target_column='note')
    print("Top 10 features most correlated with final grade:")
    print(feature_correlations.abs().sort_values(ascending=False).head(10))
    print()

    # Compute descriptive statistics for all features
    print("Computing descriptive statistics for engagement features...")
    engagement_cols = [col for col in analysis_df.columns if col not in ['pseudo', 'note', 'note_binaire']]
    descriptive_stats = processor.compute_descriptive_statistics(analysis_df[engagement_cols])
    print("Descriptive statistics summary (first 5 features):")
    print(descriptive_stats.head(5))
    print()

    # Test statistical significance of features
    print("Testing statistical significance of features...")
    significance_results = processor.test_feature_significance(analysis_df, target_column='note')
    print("Features with significant correlation (p < 0.05):")
    significant_features = significance_results[significance_results['p_value'] < 0.05].sort_values('p_value')
    print(f"  - Total significant features: {len(significant_features)}/{len(significance_results)}")
    print("  - Top 5 most significant features:")
    print(significant_features.head(5)[['correlation', 'p_value', 'is_significant']])
    print()

    # Compute correlation matrix for multicollinearity check
    print("Computing feature correlation matrix for multicollinearity analysis...")
    correlation_matrix = processor.compute_feature_feature_correlations(analysis_df[engagement_cols[:10]])  # Limit to first 10 for display
    print("Sample correlation matrix (first 10 features):")
    print(correlation_matrix)
    print()

    print("Statistical analysis completed successfully")
    print()

    # --------------------------------------------------------------------------
    # Feature Selection
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Feature Selection")
    print("=" * 60)
    print()

    # Prepare data for feature selection (exclude target and ID columns)
    feature_cols = [col for col in analysis_df.columns if col not in ['pseudo', 'note', 'note_binaire']]
    features_df = analysis_df[feature_cols + ['note']].copy()

    # Fill NaN values with 0 (missing engagement features mean no activity)
    features_df = features_df.fillna(0)

    print(f"Starting feature selection with {len(feature_cols)} features...")
    print()

    # Step 1: Variance-based selection
    print("Step 1: Removing low-variance features...")
    variance_filtered_df = processor.select_by_variance(features_df, threshold=0.01)
    variance_removed = len(feature_cols) - (len(variance_filtered_df.columns) - 1)  # -1 for note column
    print(f"  - Features removed: {variance_removed}")
    print(f"  - Features remaining: {len(variance_filtered_df.columns) - 1}")
    print()

    # Step 2: Correlation-based selection
    print("Step 2: Removing highly correlated features...")
    correlation_filtered_df = processor.select_by_correlation(variance_filtered_df, threshold=0.90)
    correlation_removed = (len(variance_filtered_df.columns) - 1) - (len(correlation_filtered_df.columns) - 1)
    print(f"  - Features removed: {correlation_removed}")
    print(f"  - Features remaining: {len(correlation_filtered_df.columns) - 1}")
    print()

    # Step 3: SelectKBest for top features
    # Get remaining feature columns (exclude 'note' if it's still in the dataframe)
    remaining_features = [col for col in correlation_filtered_df.columns if col != 'note']
    k_features = min(15, len(remaining_features))  # Ensure k doesn't exceed available features
    print(f"Step 3: Selecting top {k_features} features using statistical tests...")

    # Prepare data for select_k_best: features + note
    selectkbest_df = correlation_filtered_df[remaining_features + ['note']].copy()
    best_features_df = processor.select_k_best(
        selectkbest_df,
        target='note',
        k=k_features
    )
    print(f"  - Features selected: {len(best_features_df.columns)}")
    print(f"  - Selected features: {', '.join(best_features_df.columns.tolist())}")
    print()

    # Display final feature selection summary
    print("Feature Selection Summary:")
    print(f"  - Initial features: {len(feature_cols)}")
    print(f"  - After variance filter: {len(variance_filtered_df.columns) - 1}")
    print(f"  - After correlation filter: {len(remaining_features)}")
    print(f"  - Final selected features: {len(best_features_df.columns)}")
    print(f"  - Reduction rate: {((len(feature_cols) - len(best_features_df.columns)) / len(feature_cols) * 100):.1f}%")
    print()

    print("Feature selection completed successfully")
    print()