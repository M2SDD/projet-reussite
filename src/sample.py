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
import numpy as np
import pandas as pd
import os

from .config import Config
from .data import DatasetBuilder, StatisticsModule
from .models import LinearRegressor, EnsembleRegressor
from .evaluation import ModelEvaluator

# ----------------------------------------------------------------------------------------------------------------------
# Fonction de test / démo
# ----------------------------------------------------------------------------------------------------------------------
def test():
    # Initialisation de la configuration
    # Charger depuis config.json si présent, sinon utiliser les valeurs par défaut
    config_file_path = 'config.json'
    if os.path.exists(config_file_path):
        config = Config(config_file=config_file_path)
        print(f"✓ Configuration chargée depuis: {config_file_path}")
    else:
        config = Config()
        print("✓ Configuration par défaut utilisée (pas de config.json trouvé)")

    # Utiliser les chemins de fichiers de la configuration
    # Fallback vers les fichiers spécifiques du projet si les valeurs par défaut ne sont pas modifiées
    LOGS_FILE = config.LOGS_FILE_PATH if hasattr(config, 'LOGS_FILE_PATH') else 'data/logs_info_25_pseudo.csv'
    NOTES_FILE = config.NOTES_FILE_PATH if hasattr(config, 'NOTES_FILE_PATH') else 'data/notes_info_25_pseudo.csv'

    # Si les chemins de config sont les valeurs par défaut génériques, utiliser les fichiers spécifiques
    if LOGS_FILE == 'data/logs.csv' and os.path.exists('data/logs_info_25_pseudo.csv'):
        LOGS_FILE = 'data/logs_info_25_pseudo.csv'
    if NOTES_FILE == 'data/notes.csv' and os.path.exists('data/notes_info_25_pseudo.csv'):
        NOTES_FILE = 'data/notes_info_25_pseudo.csv'

    print(f"✓ Fichiers de données: logs={LOGS_FILE}, notes={NOTES_FILE}")
    print()

    # --------------------------------------------------------------------------
    # Data Cleaning & Preprocessing Pipeline
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Data Cleaning & Preprocessing Pipeline")
    print("=" * 60)
    print()

    # Build the full student dataset via DatasetBuilder
    builder = DatasetBuilder(config=config)
    X, y, selected_features = builder.build_dataset(
        LOGS_FILE, 
        NOTES_FILE,
        selection_methods=['linear', 'mutual_info', 'rfe']
        )

    # Train/Test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = builder.get_train_test_split(
        X,
        y
    )

    # Reconstitution d'un DataFrame complet pour les analyses suivantes
    analysis_df = pd.concat([X, y], axis=1)

    print(f"Final dataset shape: {analysis_df.shape}")
    print(f"Selected features ({len(selected_features)}): {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
    print()
    print("Sample data:")
    print(analysis_df.head(5))
    print()

    # --------------------------------------------------------------------------
    # Statistical Analysis
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Statistical Analysis")
    print("=" * 60)
    print()

    engagement_cols = [col for col in analysis_df.columns if col != 'note']
    print(f"Analysis dataset: {analysis_df.shape[0]} students, {len(engagement_cols)} features")
    print()

    # Corrélations features → cible
    print("Computing feature correlations with target variable (note)...")
    feature_correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print("Top 10 features most correlated with final grade:")
    print(feature_correlations.head(10))
    print()

    # Statistiques descriptives
    print("Computing descriptive statistics for engagement features...")
    descriptive_stats = X.describe().T
    print("Descriptive statistics summary (first 5 features):")
    print(descriptive_stats.head(5))
    print()

    # Matrice de corrélation inter-features
    print("Computing feature correlation matrix for multicollinearity analysis...")
    correlation_matrix = X[engagement_cols[:10]].corr()
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

    # La sélection de features est déjà effectuée par DatasetBuilder.build_dataset
    best_features_df = X

    print(f"Feature selection already performed by DatasetBuilder.")
    print(f"  - Selected features ({len(selected_features)}): {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
    print()
    print("Feature selection completed successfully")
    print()

    # --------------------------------------------------------------------------
    # Descriptive Statistics Analysis
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Descriptive Statistics Analysis")
    print("=" * 60)
    print()

    # Generate comprehensive statistics report for student dataset
    print("Generating descriptive statistics report for student dataset...")
    stats = StatisticsModule(config=config)
    student_report = stats.generate_report(analysis_df)
    print(student_report)
    print()

    # Generate statistics report for engagement features
    print("Generating descriptive statistics report for engagement features...")
    engagement_report = stats.generate_report(X)
    print(engagement_report)
    print()

    # --------------------------------------------------------------------------
    # Multiple Linear Regression Model
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Multiple Linear Regression Model")
    print("=" * 60)
    print()

    # Initialize regression model
    regression_model = LinearRegressor(config=config)

    print(f"  - Training set: {len(X_train)} samples ({len(X_train)/len(selected_features)*100:.1f}%)")
    print(f"  - Test set: {len(X_test)} samples ({len(X_test)/len(selected_features)*100:.1f}%)")
    print()

    # Train the model
    print("Training multiple linear regression model...")
    regression_model.fit(X_train, y_train)
    print("  ✓ Model trained successfully")
    print()

    # Display model coefficients
    print("Model Coefficients:")
    coefficients = regression_model.get_coefficients()
    print(f"  - Intercept: {coefficients['intercept']:.4f}")
    print()
    print("  Feature Coefficients:")

    # Create a DataFrame for better visualization
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefficients['coefficients']
    }).sort_values('Coefficient', key=abs, ascending=False)

    for idx, row in coef_df.head(10).iterrows():
        print(f"    {row['Feature']:<40} : {row['Coefficient']:>10.4f}")

    if len(coef_df) > 10:
        print(f"    ... and {len(coef_df) - 10} more features")
    print()

    # Evaluate model on training set
    print("Model Performance on Training Set:")
    train_metrics = regression_model.evaluate(X_train, y_train)
    print(f"  - R² Score:           {train_metrics['r2']:.4f}")
    print(f"  - RMSE:               {train_metrics['rmse']:.4f}")
    print(f"  - MAE:                {train_metrics['mae']:.4f}")
    print()

    # Evaluate model on test set
    print("Model Performance on Test Set:")
    test_metrics = regression_model.evaluate(X_test, y_test)
    print(f"  - R² Score:           {test_metrics['r2']:.4f}")
    print(f"  - RMSE:               {test_metrics['rmse']:.4f}")
    print(f"  - MAE:                {test_metrics['mae']:.4f}")
    print()

    # Make sample predictions
    print("Sample Predictions:")
    y_pred_sample = regression_model.predict(X_test.head(5))
    prediction_comparison = pd.DataFrame({
        'Actual': y_test.head(5).values,
        'Predicted': y_pred_sample,
        'Residual': y_test.head(5).values - y_pred_sample
    })
    print(prediction_comparison.to_string(index=False))
    print()

    # Residual analysis
    print("Residual Analysis:")

    # Compute residuals
    residuals = regression_model.compute_residuals(X_test, y_test)
    print(f"  - Mean residual:      {np.mean(residuals):.4f}")
    print(f"  - Std residual:       {np.std(residuals):.4f}")
    print(f"  - Min residual:       {np.min(residuals):.4f}")
    print(f"  - Max residual:       {np.max(residuals):.4f}")
    print()

    # Check normality of residuals
    print("Residual Normality Test (Shapiro-Wilk):")
    normality_test = regression_model.check_residuals_normality(X_test, y_test)
    print(f"  - Test Statistic: {normality_test['test_statistic']:.4f}")
    print(f"  - P-Value:        {normality_test['p_value']:.4f}")

    if normality_test['p_value'] > 0.05:
        print("  - Interpretation: Residuals appear normally distributed (p > 0.05)")
    else:
        print("  - Interpretation: Residuals may not be normally distributed (p ≤ 0.05)")
    print()

    # Model comparison summary
    print("Model Summary:")
    print(f"  - Features used:      {len(selected_features)}")
    print(f"  - Training samples:   {len(X_train)}")
    print(f"  - Test samples:       {len(X_test)}")
    print(f"  - Test R²:            {test_metrics['r2']:.4f}")
    print(f"  - Test RMSE:          {test_metrics['rmse']:.4f}")
    print(f"  - Overfitting check:  {'Minimal' if abs(train_metrics['r2'] - test_metrics['r2']) < 0.1 else 'Possible'}")
    print()

    print("Multiple linear regression model demonstration completed successfully")
    print()


    # --------------------------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Random Forest Regression Model")
    print("=" * 60)
    print()

    # Initialize regression model
    ensemble_model = EnsembleRegressor(config=config)

    # Train the model
    print("Training random forest regression model...")
    ensemble_model.fit(X_train, y_train)
    print("  ✓ Model trained successfully")
    print()

    # Display model coefficients
    print("Feature Importance:")
    feature_importance = ensemble_model.get_feature_importance(feature_names=X_train.columns)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']:<40} : {row['importance']:>10.4f}")

    if len(feature_importance) > 10:
        print(f"    ... and {len(feature_importance) - 10} more features")
    print()

    # Evaluate model on training set
    print("Model Performance on Training Set:")
    train_metrics = ensemble_model.evaluate(X_train, y_train)
    print(f"  - R² Score:           {train_metrics['r2']:.4f}")
    print(f"  - RMSE:               {train_metrics['rmse']:.4f}")
    print(f"  - MAE:                {train_metrics['mae']:.4f}")
    print()

    # Evaluate model on test set
    print("Model Performance on Test Set:")
    test_metrics = ensemble_model.evaluate(X_test, y_test)
    print(f"  - R² Score:           {test_metrics['r2']:.4f}")
    print(f"  - RMSE:               {test_metrics['rmse']:.4f}")
    print(f"  - MAE:                {test_metrics['mae']:.4f}")
    print()

    # Make sample predictions
    print("Sample Predictions:")
    y_pred_sample = ensemble_model.predict(X_test.head(5))
    prediction_comparison = pd.DataFrame({
        'Actual': y_test.head(5).values,
        'Predicted': y_pred_sample,
        'Residual': y_test.head(5).values - y_pred_sample
    })
    print(prediction_comparison.to_string(index=False))
    print()

    # Residual analysis
    print("Residual Analysis:")

    # Compute residuals
    residuals = ensemble_model.compute_residuals(X_test, y_test)
    print(f"  - Mean residual:      {np.mean(residuals):.4f}")
    print(f"  - Std residual:       {np.std(residuals):.4f}")
    print(f"  - Min residual:       {np.min(residuals):.4f}")
    print(f"  - Max residual:       {np.max(residuals):.4f}")
    print()

    # Check normality of residuals
    print("Residual Normality Test (Shapiro-Wilk):")
    normality_test = ensemble_model.check_residuals_normality(X_test, y_test)
    print(f"  - Test Statistic: {normality_test['test_statistic']:.4f}")
    print(f"  - P-Value:        {normality_test['p_value']:.4f}")

    if normality_test['p_value'] > 0.05:
        print("  - Interpretation: Residuals appear normally distributed (p > 0.05)")
    else:
        print("  - Interpretation: Residuals may not be normally distributed (p ≤ 0.05)")
    print()

    # Model comparison summary
    print("Model Summary:")
    print(f"  - Features used:      {len(selected_features)}")
    print(f"  - Training samples:   {len(X_train)}")
    print(f"  - Test samples:       {len(X_test)}")
    print(f"  - Test R²:            {test_metrics['r2']:.4f}")
    print(f"  - Test RMSE:          {test_metrics['rmse']:.4f}")
    print(f"  - Overfitting check:  {'Minimal' if abs(train_metrics['r2'] - test_metrics['r2']) < 0.1 else 'Possible'}")
    print()

    print("Random Forest regression model demonstration completed successfully")
    print()


    # --------------------------------------------------------------------------
    # Models comparison
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Models comparison")
    print("=" * 60)
    evaluator = ModelEvaluator()
    evaluator.add_model('LinearRegression', regression_model, X_test, y_test)
    evaluator.add_model('RandomForestRegressor', ensemble_model, X_test, y_test)
    #evaluator.export_results(output_dir='tests')
    #print("Comparison results exported to 'tests' directory.")
    print("Recommandations based on models' performances:")
    reco = evaluator.get_recommendation()
    print("\t- Best model :", reco['best_model'])
    print("\t- Reason :", reco['reason'])
    print()

    # --------------------------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Pipeline Execution Summary")
    print("=" * 60)
    print()
    print("✓ Data Loading:                    Completed")
    print("✓ Data Cleaning:                   Completed")
    print("✓ Feature Engineering:             Completed")
    print("✓ Statistical Analysis:            Completed")
    print("✓ Feature Selection:               Completed")
    print("✓ Descriptive Statistics:          Completed")
    print("✓ LinearRegression Model:          Completed")
    print("✓ Random Forest Regression Model:  Completed")
    print("✓ Model Evaluation:                Completed")
    print()
    print(f"Final {reco['best_model']} model ready for deployment with {len(selected_features)} features")
    if reco.get('metrics') is not None:
        print(f"Model performance: R²={reco['metrics']['r2']:.4f}, RMSE={reco['metrics']['rmse']:.4f}")
    else:
        print("Model performance: metrics unavailable (insufficient samples)")
    print()
    print("=" * 60)