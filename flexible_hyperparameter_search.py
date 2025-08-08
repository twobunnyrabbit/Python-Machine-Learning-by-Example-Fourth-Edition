#!/usr/bin/env python3
"""
Flexible Hyperparameter Search Script for Tree-Based Classifiers

This script supports hyperparameter tuning for DecisionTreeClassifier, RandomForestClassifier, 
and XGBClassifier with user-selectable parameters and datasets.

TUNABLE PARAMETERS BY CLASSIFIER:

DecisionTreeClassifier:
    - max_depth: Maximum depth of tree [3, 5, 7, 10, 15, 20, None]
    - min_samples_split: Min samples required to split [2, 5, 10, 20, 50]
    - min_samples_leaf: Min samples required in leaf [1, 2, 4, 8, 16]
    - criterion: Function to measure split quality ['gini', 'entropy']
    - max_features: Number of features for best split ['sqrt', 'log2', None]
    - splitter: Strategy to split at each node ['best', 'random']

RandomForestClassifier:
    - n_estimators: Number of trees [50, 100, 200, 300, 500]
    - max_depth: Maximum depth of trees [5, 10, 15, 20, None]
    - min_samples_split: Min samples required to split [2, 5, 10, 20]
    - min_samples_leaf: Min samples required in leaf [1, 2, 4, 8]
    - max_features: Features to consider for best split ['sqrt', 'log2', None]
    - criterion: Function to measure split quality ['gini', 'entropy']
    - bootstrap: Whether bootstrap samples are used [True, False]
    - max_samples: Fraction of samples to draw [0.5, 0.7, 0.8, 1.0]

XGBClassifier:
    - n_estimators: Number of boosting rounds [50, 100, 200, 300, 500]
    - max_depth: Maximum depth of trees [3, 4, 5, 6, 7, 8]
    - learning_rate: Boosting learning rate [0.01, 0.05, 0.1, 0.15, 0.2]
    - subsample: Fraction of samples for training [0.6, 0.8, 0.9, 1.0]
    - colsample_bytree: Fraction of features for training [0.6, 0.8, 0.9, 1.0]
    - reg_alpha: L1 regularization term [0, 0.1, 0.5, 1.0, 2.0]
    - reg_lambda: L2 regularization term [0, 0.1, 0.5, 1.0, 2.0]
    - gamma: Minimum loss reduction for split [0, 0.1, 0.2, 0.5]

Usage Examples:
    # Basic usage with default parameter values
    python flexible_hyperparameter_search.py --classifier decision_tree \
        --dataset click-rate-train.csv --target click \
        --drop id,hour,device_id,device_ip \
        --tune-params max_depth,min_samples_split,criterion

    # Custom parameter values (JSON format)
    python flexible_hyperparameter_search.py --classifier decision_tree \
        --dataset click-rate-train.csv --target click \
        --tune-params max_depth,criterion \
        --param-values '{"max_depth": [5, 10, 15], "criterion": ["gini"]}'

    # Custom parameter values (key-value format)
    python flexible_hyperparameter_search.py --classifier random_forest \
        --dataset my_data.csv --target outcome \
        --tune-params n_estimators,max_depth \
        --param-values "n_estimators=100,200,300 max_depth=10,15,20"

    # Mixed: custom values for some parameters, defaults for others
    python flexible_hyperparameter_search.py --classifier xgboost \
        --dataset my_data.csv --target binary_outcome \
        --tune-params learning_rate,reg_alpha,n_estimators \
        --param-values '{"learning_rate": [0.05, 0.1, 0.15]}'

    # Help and examples
    python flexible_hyperparameter_search.py --help-params decision_tree
    python flexible_hyperparameter_search.py --examples
"""

import argparse
import sys
import time
import json
import warnings
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Suppress multiprocessing warnings on macOS (common issue with n_jobs=-1)
if sys.platform == 'darwin':  # macOS
    warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
    # Set multiprocessing start method to avoid issues
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent))

from config import get_data_path, DATA_ROOT
from model_cache import cached_grid_search, ModelCache
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
                           log_loss, average_precision_score, brier_score_loss, classification_report, 
                           confusion_matrix)

# Import XGBoost with fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False


# Parameter grids for each classifier
PARAMETER_GRIDS = {
    'decision_tree': {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'splitter': ['best', 'random']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False],
        'max_samples': [0.5, 0.7, 0.8, 1.0]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
        'gamma': [0, 0.1, 0.2, 0.5]
    }
}

# Classifier mappings
CLASSIFIERS = {
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'xgboost': XGBClassifier if HAS_XGBOOST else None
}


def validate_classifier(classifier_name):
    """Validate classifier name and availability."""
    if classifier_name not in CLASSIFIERS:
        available = list(CLASSIFIERS.keys())
        raise ValueError(f"Unknown classifier '{classifier_name}'. Available: {available}")
    
    if classifier_name == 'xgboost' and not HAS_XGBOOST:
        raise ImportError("XGBoost not available. Install with: uv add xgboost")
    
    return True


def validate_parameters(classifier_name, tune_params):
    """Validate that tune_params are valid for the given classifier."""
    available_params = set(PARAMETER_GRIDS[classifier_name].keys())
    invalid_params = set(tune_params) - available_params
    
    if invalid_params:
        raise ValueError(f"Invalid parameters for {classifier_name}: {invalid_params}. "
                        f"Available: {sorted(available_params)}")
    
    return True


def load_and_prepare_data(dataset_path, target_column, drop_columns=None, test_size=0.3, random_state=42):
    """Load and prepare dataset for training."""
    print("📊 Loading and preparing data...")
    
    # Handle dataset path - check if it's a full path or needs config resolution
    if Path(dataset_path).exists():
        data_path = Path(dataset_path)
    else:
        # Try to resolve using config
        try:
            # Extract chapter from path if present, otherwise use generic path
            if 'ch' in dataset_path.lower():
                chapter = int(''.join(filter(str.isdigit, dataset_path.split('ch')[1][:2])))
                data_path = get_data_path(chapter, dataset_path)
            else:
                # Try common locations
                data_path = DATA_ROOT / dataset_path
                if not data_path.exists():
                    data_path = Path.cwd() / dataset_path
        except:
            data_path = Path(dataset_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset: {data_path}")
    print(f"   Shape: {df.shape}")
    
    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")
    
    # Prepare features and target
    columns_to_drop = [target_column]
    if drop_columns:
        columns_to_drop.extend(drop_columns)
        
    # Validate drop columns
    invalid_drops = [col for col in drop_columns or [] if col not in df.columns]
    if invalid_drops:
        print(f"⚠️  Warning: Drop columns not found: {invalid_drops}")
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df[target_column]
    
    # Convert categorical variables to dummy variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"   Converting categorical columns: {list(categorical_cols)}")
        X = pd.get_dummies(X, drop_first=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data prepared:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set:  {X_test.shape}")
    print(f"   Features:     {X.shape[1]}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, list(X.columns)


def parse_json_param_values(param_values_str):
    """Parse JSON format parameter values."""
    try:
        custom_values = json.loads(param_values_str)
        if not isinstance(custom_values, dict):
            raise ValueError("JSON parameter values must be a dictionary")
        return custom_values
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in --param-values: {e}")


def parse_keyvalue_param_values(param_values_str):
    """Parse key-value format parameter values (e.g., 'max_depth=3,5,10 criterion=gini,entropy')."""
    custom_values = {}
    
    # Split by spaces to get individual parameter assignments
    param_assignments = param_values_str.strip().split()
    
    for assignment in param_assignments:
        if '=' not in assignment:
            raise ValueError(f"Invalid parameter assignment format: '{assignment}'. Expected 'param=value1,value2'")
        
        param_name, values_str = assignment.split('=', 1)
        param_name = param_name.strip()
        
        # Split values by comma
        value_strings = [v.strip() for v in values_str.split(',')]
        
        # Convert values to appropriate types
        values = []
        for value_str in value_strings:
            # Try to convert to appropriate type
            if value_str.lower() in ['true', 'false']:
                values.append(value_str.lower() == 'true')
            elif value_str.lower() == 'none':
                values.append(None)
            else:
                # Try int, then float, then string
                try:
                    if '.' in value_str:
                        values.append(float(value_str))
                    else:
                        values.append(int(value_str))
                except ValueError:
                    # Keep as string
                    values.append(value_str)
        
        custom_values[param_name] = values
    
    return custom_values


def validate_custom_param_values(custom_values, tune_params, classifier_name):
    """Validate that custom parameter values are appropriate."""
    # Check that all custom parameters are in tune_params
    invalid_params = set(custom_values.keys()) - set(tune_params)
    if invalid_params:
        raise ValueError(f"Custom values provided for parameters not being tuned: {invalid_params}")
    
    # Check that custom parameters are valid for the classifier
    available_params = set(PARAMETER_GRIDS[classifier_name].keys())
    invalid_classifier_params = set(custom_values.keys()) - available_params
    if invalid_classifier_params:
        raise ValueError(f"Custom values provided for invalid parameters for {classifier_name}: {invalid_classifier_params}")
    
    # Validate individual parameter values (basic checks)
    for param, values in custom_values.items():
        if not isinstance(values, list):
            raise ValueError(f"Parameter values for '{param}' must be a list, got {type(values)}")
        
        if len(values) == 0:
            raise ValueError(f"Parameter values for '{param}' cannot be empty")
        
        # Check for reasonable ranges for numerical parameters
        if param in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
            for value in values:
                if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                    raise ValueError(f"Parameter '{param}' must have positive numerical values or None, got {value}")
        
        if param in ['learning_rate', 'subsample', 'colsample_bytree']:
            for value in values:
                if not isinstance(value, (int, float)) or value <= 0 or value > 1:
                    raise ValueError(f"Parameter '{param}' must have values between 0 and 1, got {value}")
    
    return True


def create_parameter_grid(classifier_name, tune_params, custom_param_values=None):
    """Create parameter grid for the specified parameters with optional custom values."""
    full_grid = PARAMETER_GRIDS[classifier_name]
    param_grid = {}
    
    print(f"📋 Parameter grid for {classifier_name}:")
    
    for param in tune_params:
        if custom_param_values and param in custom_param_values:
            # Use custom values
            param_grid[param] = custom_param_values[param]
            print(f"   {param}: {custom_param_values[param]} (custom)")
        else:
            # Use default values
            param_grid[param] = full_grid[param]
            print(f"   {param}: {full_grid[param]} (default)")
    
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    print(f"   Total combinations: {total_combinations}")
    
    return param_grid


def run_hyperparameter_search(classifier_name, param_grid, X_train, y_train, 
                             search_type='grid', n_iter=None, cv=5, scoring='f1', 
                             n_jobs=-1, verbose=1, random_state=42):
    """Run hyperparameter search with caching."""
    
    # Get classifier class and create instance
    classifier_class = CLASSIFIERS[classifier_name]
    
    # Set classifier-specific defaults
    classifier_kwargs = {'random_state': random_state}
    if classifier_name == 'xgboost':
        classifier_kwargs['eval_metric'] = 'logloss'  # Suppress XGBoost warning
    
    estimator = classifier_class(**classifier_kwargs)
    
    # Create unique model name
    param_str = '_'.join(sorted(param_grid.keys()))
    model_name = f"{classifier_name}_{search_type}_{param_str}"
    
    print(f"\n🔍 Running {search_type.upper()} search for {classifier_name}...")
    print(f"   Model name: {model_name}")
    
    start_time = time.time()
    
    if search_type == 'grid':
        # Use cached grid search
        search_result = cached_grid_search(
            estimator=estimator,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    
    elif search_type == 'randomized':
        # Use manual caching for randomized search
        cache = ModelCache()
        
        search_params = {
            'estimator': classifier_name,
            'param_distributions': param_grid,
            'n_iter': n_iter or 50,
            'cv': cv,
            'scoring': scoring,
            'random_state': random_state
        }
        
        # Check cache first
        cached_result = cache.load_search_result(model_name, search_params)
        
        if cached_result is None:
            print(f"   Running RandomizedSearchCV with {search_params['n_iter']} iterations...")
            
            search_result = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=search_params['n_iter'],
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=random_state
            )
            
            search_result.fit(X_train, y_train)
            
            # Save to cache
            cache.save_search_result(model_name, search_result, search_params)
        else:
            search_result = cached_result
    
    else:
        raise ValueError(f"Unknown search type: {search_type}")
    
    elapsed = time.time() - start_time
    print(f"⏱️  Search completed in {elapsed/60:.1f} minutes")
    print(f"🏆 Best {scoring} score: {search_result.best_score_:.4f}")
    print(f"🎯 Best parameters: {search_result.best_params_}")
    
    return search_result


def collect_comprehensive_metrics(search_result, X_test, y_test, X_train, y_train, classifier_name,
                                feature_names, search_type, training_time, custom_params_used, 
                                test_size_ratio, cv_folds, scoring_metric, n_jobs):
    """
    Collect comprehensive metrics and metadata for storage.
    
    Args:
        search_result: Fitted GridSearchCV/RandomizedSearchCV object
        X_test, y_test: Test data
        X_train, y_train: Training data
        classifier_name: Name of the classifier
        feature_names: List of feature names
        search_type: Type of search (grid/randomized)
        training_time: Training time in minutes
        custom_params_used: Whether custom parameter values were used
        test_size_ratio: Test set proportion
        cv_folds: Number of CV folds used
        scoring_metric: Scoring metric used
        n_jobs: Number of parallel jobs used
        
    Returns:
        dict: Comprehensive metrics dictionary
    """
    best_estimator = search_result.best_estimator_
    
    # Get predictions
    y_pred = best_estimator.predict(X_test)
    
    # Handle probability predictions
    try:
        y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_pred_proba = None
        has_proba = False
    
    # Core test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if has_proba else None
    }
    
    # Data information
    data_info = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
        'target_distribution_train': y_train.value_counts().to_dict(),
        'target_distribution_test': y_test.value_counts().to_dict(),
        'test_size_ratio': test_size_ratio
    }
    
    # Training information
    training_info = {
        'search_type': search_type,
        'total_combinations_tested': len(search_result.cv_results_['params']) if hasattr(search_result, 'cv_results_') else None,
        'training_time_minutes': training_time,
        'custom_params_used': custom_params_used,
        'n_jobs': n_jobs,
        'cv_folds': cv_folds,
        'scoring_metric': scoring_metric
    }
    
    # Classification details
    classification_details = {
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Model-specific metrics
    model_specific = {}
    
    # Feature importance (for tree-based models)
    if hasattr(best_estimator, 'feature_importances_'):
        model_specific['feature_importances'] = best_estimator.feature_importances_.tolist()
    else:
        model_specific['feature_importances'] = None
    
    # Decision Tree specific
    if hasattr(best_estimator, 'tree_'):
        model_specific['tree_depth'] = int(best_estimator.tree_.max_depth)
        model_specific['n_leaves'] = int(best_estimator.tree_.n_leaves)
    else:
        model_specific['tree_depth'] = None
        model_specific['n_leaves'] = None
    
    # Ensemble specific (Random Forest, XGBoost)
    if hasattr(best_estimator, 'n_estimators'):
        model_specific['n_estimators'] = int(best_estimator.n_estimators)
    else:
        model_specific['n_estimators'] = None
    
    # Advanced probability metrics
    probability_metrics = {}
    if has_proba:
        try:
            probability_metrics['log_loss'] = log_loss(y_test, y_pred_proba)
        except:
            probability_metrics['log_loss'] = None
            
        try:
            probability_metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
        except:
            probability_metrics['average_precision'] = None
            
        try:
            probability_metrics['brier_score'] = brier_score_loss(y_test, y_pred_proba)
        except:
            probability_metrics['brier_score'] = None
    else:
        probability_metrics = {
            'log_loss': None,
            'average_precision': None,
            'brier_score': None
        }
    
    return {
        'test_metrics': test_metrics,
        'data_info': data_info,
        'training_info': training_info,
        'classification_details': classification_details,
        'model_specific': model_specific,
        'probability_metrics': probability_metrics
    }


def evaluate_model(search_result, X_test, y_test, classifier_name):
    """Evaluate the best model on test set."""
    print(f"\n📈 Evaluating {classifier_name} on test set...")
    
    best_estimator = search_result.best_estimator_
    
    # Predictions
    y_pred = best_estimator.predict(X_test)
    
    # Handle probability predictions
    try:
        y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_pred_proba = None
        has_proba = False
    
    # Calculate metrics
    metrics = {
        'cv_score': search_result.best_score_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'best_params': search_result.best_params_
    }
    
    if has_proba:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Display results
    print(f"📊 Test Set Results:")
    print(f"   CV {search_result.scorer_.__class__.__name__}: {metrics['cv_score']:.4f}")
    print(f"   Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Test Precision: {metrics['precision']:.4f}")
    print(f"   Test Recall:    {metrics['recall']:.4f}")
    print(f"   Test F1 Score:  {metrics['f1']:.4f}")
    if has_proba:
        print(f"   Test ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def show_parameter_help(classifier_name):
    """Show available parameters for a classifier."""
    if classifier_name not in PARAMETER_GRIDS:
        print(f"Unknown classifier: {classifier_name}")
        print(f"Available classifiers: {list(PARAMETER_GRIDS.keys())}")
        return
    
    print(f"\n📋 Available parameters for {classifier_name.replace('_', ' ').title()}:")
    print("=" * 60)
    
    for param, values in PARAMETER_GRIDS[classifier_name].items():
        print(f"{param:20}: {values}")
    
    print(f"\nUsage examples:")
    params = list(PARAMETER_GRIDS[classifier_name].keys())[:3]
    param_str = ','.join(params)
    
    print(f"\n1. Using default values:")
    print(f"python flexible_hyperparameter_search.py \\")
    print(f"    --classifier {classifier_name} \\")
    print(f"    --dataset your_data.csv \\")
    print(f"    --target your_target \\")
    print(f"    --tune-params {param_str}")
    
    print(f"\n2. Using custom values (JSON):")
    print(f"python flexible_hyperparameter_search.py \\")
    print(f"    --classifier {classifier_name} \\")
    print(f"    --dataset your_data.csv \\")
    print(f"    --target your_target \\")
    print(f"    --tune-params {params[0]},{params[1]} \\")
    if classifier_name == 'decision_tree':
        print(f"    --param-values '{{\"max_depth\": [5, 10, 15], \"criterion\": [\"gini\"]}}'")
    elif classifier_name == 'random_forest':
        print(f"    --param-values '{{\"n_estimators\": [100, 200], \"max_depth\": [10, 20]}}'")
    else:  # xgboost
        print(f"    --param-values '{{\"n_estimators\": [100, 200], \"learning_rate\": [0.1, 0.2]}}'")
    
    print(f"\n3. Using custom values (key-value):")
    print(f"python flexible_hyperparameter_search.py \\")
    print(f"    --classifier {classifier_name} \\")
    print(f"    --dataset your_data.csv \\")
    print(f"    --target your_target \\")
    print(f"    --tune-params {params[0]},{params[1]} \\")
    if classifier_name == 'decision_tree':
        print(f"    --param-values \"max_depth=5,10,15 criterion=gini,entropy\"")
    elif classifier_name == 'random_forest':
        print(f"    --param-values \"n_estimators=100,200 max_depth=10,20\"")
    else:  # xgboost
        print(f"    --param-values \"n_estimators=100,200 learning_rate=0.1,0.2\"")


def show_examples():
    """Show usage examples for all classifiers."""
    print("\n🚀 Usage Examples:")
    print("=" * 60)
    
    examples = [
        {
            'name': 'Decision Tree - Default values',
            'cmd': ['python flexible_hyperparameter_search.py',
                   '    --classifier decision_tree',
                   '    --dataset click-rate-train.csv',
                   '    --target click',
                   '    --drop id,hour,device_id,device_ip',
                   '    --tune-params max_depth,criterion']
        },
        {
            'name': 'Decision Tree - Custom values (JSON)',
            'cmd': ['python flexible_hyperparameter_search.py',
                   '    --classifier decision_tree',
                   '    --dataset click-rate-train.csv',
                   '    --target click',
                   '    --tune-params max_depth,min_samples_split',
                   '    --param-values \'{"max_depth": [5, 10, 15], "min_samples_split": [2, 5]}\'']
        },
        {
            'name': 'Random Forest - Custom values (key-value)',
            'cmd': ['python flexible_hyperparameter_search.py',
                   '    --classifier random_forest',
                   '    --dataset my_data.csv',
                   '    --target outcome',
                   '    --tune-params n_estimators,max_depth',
                   '    --param-values "n_estimators=100,200,300 max_depth=10,15,20"']
        },
        {
            'name': 'XGBoost - Mixed (custom + default)',
            'cmd': ['python flexible_hyperparameter_search.py',
                   '    --classifier xgboost',
                   '    --dataset large_data.csv',
                   '    --target binary_outcome',
                   '    --tune-params learning_rate,reg_alpha,n_estimators',
                   '    --param-values \'{"learning_rate": [0.05, 0.1, 0.15]}\'',
                   '    # reg_alpha and n_estimators will use default values']
        },
        {
            'name': 'Random Forest - Randomized search with custom values',
            'cmd': ['python flexible_hyperparameter_search.py',
                   '    --classifier random_forest',
                   '    --dataset large_data.csv',
                   '    --target outcome',
                   '    --tune-params n_estimators,max_depth,min_samples_split',
                   '    --param-values \'{"n_estimators": [50,100,200,300,500]}\'',
                   '    --search-type randomized',
                   '    --n-iter 20']
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print('\\\n'.join(example['cmd']))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Flexible Hyperparameter Search for Tree-Based Classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --classifier decision_tree --dataset data.csv --target y --tune-params max_depth,criterion
  %(prog)s --classifier random_forest --dataset data.csv --target y --tune-params n_estimators,max_depth
  %(prog)s --help-params decision_tree
  %(prog)s --examples
        """
    )
    
    # Classifier and data arguments
    parser.add_argument('--classifier', choices=['decision_tree', 'random_forest', 'xgboost'],
                       help='Classifier to use')
    parser.add_argument('--dataset', type=str,
                       help='Path to CSV dataset')
    parser.add_argument('--target', type=str,
                       help='Target column name')
    parser.add_argument('--drop', type=str,
                       help='Comma-separated columns to drop (optional)')
    
    # Parameter tuning arguments
    parser.add_argument('--tune-params', type=str,
                       help='Comma-separated parameters to tune')
    parser.add_argument('--param-values', type=str,
                       help='Custom parameter values in JSON format or key=value1,value2 format')
    parser.add_argument('--search-type', choices=['grid', 'randomized'], default='grid',
                       help='Type of search (default: grid)')
    parser.add_argument('--n-iter', type=int, default=50,
                       help='Number of iterations for randomized search (default: 50)')
    
    # Search configuration
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--scoring', type=str, default='f1',
                       help='Scoring metric (default: f1)')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Test set proportion (default: 0.3)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    
    # Help arguments
    parser.add_argument('--help-params', type=str,
                       help='Show available parameters for a classifier')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples')
    
    args = parser.parse_args()
    
    # Handle help requests
    if args.help_params:
        show_parameter_help(args.help_params)
        return
    
    if args.examples:
        show_examples()
        return
    
    # Validate required arguments
    if not all([args.classifier, args.dataset, args.target, args.tune_params]):
        parser.error("--classifier, --dataset, --target, and --tune-params are required")
    
    try:
        # Validate inputs
        validate_classifier(args.classifier)
        tune_params = [p.strip() for p in args.tune_params.split(',')]
        validate_parameters(args.classifier, tune_params)
        
        # Parse custom parameter values if provided
        custom_param_values = None
        if args.param_values:
            try:
                # Try JSON format first
                if args.param_values.strip().startswith('{'):
                    custom_param_values = parse_json_param_values(args.param_values)
                else:
                    # Try key-value format
                    custom_param_values = parse_keyvalue_param_values(args.param_values)
                
                # Validate custom values
                validate_custom_param_values(custom_param_values, tune_params, args.classifier)
                
            except Exception as e:
                raise ValueError(f"Error parsing --param-values: {e}")
        
        # Parse drop columns
        drop_columns = [col.strip() for col in args.drop.split(',')] if args.drop else None
        
        print("🚀 Starting Flexible Hyperparameter Search")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Classifier: {args.classifier}")
        print(f"Dataset: {args.dataset}")
        print(f"Target: {args.target}")
        print(f"Parameters to tune: {tune_params}")
        print(f"Search type: {args.search_type}")
        
        # Show multiprocessing info
        if args.n_jobs == -1:
            cpu_count = os.cpu_count()
            print(f"Parallel jobs: {cpu_count} cores (all available)")
            if sys.platform == 'darwin':
                print("ℹ️  Note: Multiprocessing warnings suppressed on macOS")
        else:
            print(f"Parallel jobs: {args.n_jobs}")
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
            args.dataset, args.target, drop_columns, args.test_size, args.random_state
        )
        
        # Create parameter grid
        param_grid = create_parameter_grid(args.classifier, tune_params, custom_param_values)
        
        # Run hyperparameter search
        start_time = time.time()
        search_result = run_hyperparameter_search(
            args.classifier, param_grid, X_train, y_train,
            args.search_type, args.n_iter, args.cv, args.scoring,
            args.n_jobs, args.verbose, args.random_state
        )
        training_time = (time.time() - start_time) / 60  # Convert to minutes
        
        # Evaluate model
        metrics = evaluate_model(search_result, X_test, y_test, args.classifier)
        
        # Collect comprehensive metrics
        comprehensive_metrics = collect_comprehensive_metrics(
            search_result=search_result,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            classifier_name=args.classifier,
            feature_names=feature_names,
            search_type=args.search_type,
            training_time=training_time,
            custom_params_used=custom_param_values is not None,
            test_size_ratio=args.test_size,
            cv_folds=args.cv,
            scoring_metric=args.scoring,
            n_jobs=args.n_jobs
        )
        
        # Update cache with comprehensive metrics
        cache = ModelCache()
        param_str = '_'.join(sorted(param_grid.keys()))
        model_name = f"{args.classifier}_{args.search_type}_{param_str}"
        
        # Create params dict for caching (similar to what's used in run_hyperparameter_search)
        if args.search_type == 'grid':
            cache_params = {
                'estimator': str(type(search_result.estimator).__name__),
                'param_grid': param_grid,
                'cv': args.cv,
                'scoring': args.scoring,
                'estimator_params': search_result.estimator.get_params(),
                'verbose': args.verbose
            }
        else:  # randomized
            cache_params = {
                'estimator': args.classifier,
                'param_distributions': param_grid,
                'n_iter': args.n_iter or 50,
                'cv': args.cv,
                'scoring': args.scoring,
                'random_state': args.random_state
            }
        
        # Save comprehensive metrics to existing cache
        from model_cache import hash_data
        data_hash = hash_data(X_train, y_train)
        cache.save_search_result(model_name, search_result, cache_params, data_hash, comprehensive_metrics)
        
        # Show summary
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"Best {args.scoring} score: {search_result.best_score_:.4f}")
        print(f"Test F1 score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Cache info
        print(f"\n💾 Results and comprehensive metrics cached for analysis:")
        cache.list_cached_models()
        
        print(f"\n✅ Search completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💡 Use analyze_flexible_search_results.qmd to explore comprehensive results!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()