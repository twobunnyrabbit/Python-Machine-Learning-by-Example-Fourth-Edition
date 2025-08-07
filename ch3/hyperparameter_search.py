#!/usr/bin/env python3
"""
Hyperparameter search script for Decision Tree click-rate prediction.

This script runs expensive GridSearchCV and RandomizedSearchCV operations
with full multiprocessing support, then caches results for analysis in notebooks.

Usage:
    cd ch3/
    python hyperparameter_search.py
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config import get_click_rate_file
from model_cache import cached_grid_search, ModelCache
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def load_and_prepare_data():
    """Load and prepare the click-rate dataset."""
    print("📊 Loading and preparing data...")
    
    # Load data
    data_path = get_click_rate_file('click-rate-train.csv')
    df = pd.read_csv(data_path)
    
    # Prepare features
    features_to_drop = ['click', 'id', 'hour', 'device_id', 'device_ip']
    X = df.drop(columns=features_to_drop)
    y = df['click']
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✅ Data prepared:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set:  {X_test.shape}")
    print(f"   Features:     {X.shape[1]}")
    print(f"   Positive class ratio: {y.mean():.4f}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def run_grid_search(X_train, y_train):
    """Run comprehensive GridSearchCV."""
    print("\n🔍 Running GridSearchCV...")
    
    # Comprehensive parameter grid
    param_grid = {
        'max_depth': [5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print(f"Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Use cached grid search with full multiprocessing
    start_time = time.time()
    
    grid_search = cached_grid_search(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        model_name="decision_tree_comprehensive_grid",
        cv=5,
        scoring='f1',
        n_jobs=-1,  # Full multiprocessing - works in scripts!
        verbose=2
    )
    
    elapsed = time.time() - start_time
    print(f"⏱️  GridSearchCV completed in {elapsed/60:.1f} minutes")
    print(f"🏆 Best F1 score: {grid_search.best_score_:.4f}")
    print(f"🎯 Best parameters: {grid_search.best_params_}")
    
    return grid_search

def run_randomized_search(X_train, y_train):
    """Run comprehensive RandomizedSearchCV."""
    print("\n🎲 Running RandomizedSearchCV...")
    
    # Parameter distributions for random search
    param_dist = {
        'max_depth': [3, 5, 7, 10, 15, 20, 25, None],
        'min_samples_split': list(range(2, 101, 2)),  # 2 to 100, step 2
        'min_samples_leaf': list(range(1, 51)),       # 1 to 50
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'splitter': ['best', 'random']
    }
    
    # Create cache instance for manual control
    cache = ModelCache()
    
    search_params = {
        'estimator': 'DecisionTreeClassifier', 
        'param_distributions': param_dist,
        'n_iter': 200,  # Many iterations for thorough search
        'cv': 5,
        'scoring': 'f1',
        'random_state': 42
    }
    
    model_name = "decision_tree_comprehensive_random"
    
    # Check cache first
    cached_result = cache.load_search_result(model_name, search_params)
    
    if cached_result is None:
        print(f"Running RandomizedSearchCV with {search_params['n_iter']} iterations...")
        
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=search_params['n_iter'],
            cv=search_params['cv'],
            scoring=search_params['scoring'],
            n_jobs=-1,  # Full multiprocessing
            verbose=2,
            random_state=search_params['random_state']
        )
        
        random_search.fit(X_train, y_train)
        
        # Save to cache
        cache.save_search_result(model_name, random_search, search_params)
        
        elapsed = time.time() - start_time
        print(f"⏱️  RandomizedSearchCV completed in {elapsed/60:.1f} minutes")
        
    else:
        random_search = cached_result
    
    print(f"🏆 Best F1 score: {random_search.best_score_:.4f}")
    print(f"🎯 Best parameters: {random_search.best_params_}")
    
    return random_search

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test set."""
    print("\n📈 Evaluating models on test set...")
    
    results = {}
    
    for name, model in models.items():
        best_estimator = model.best_estimator_
        
        # Predictions
        y_pred = best_estimator.predict(X_test)
        y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'cv_score': model.best_score_,
            'test_accuracy': accuracy,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'best_params': model.best_params_
        }
        
        print(f"\n{name}:")
        print(f"  CV F1 Score:    {model.best_score_:.4f}")
        print(f"  Test Accuracy:  {accuracy:.4f}")
        print(f"  Test F1 Score:  {f1:.4f}")
        print(f"  Test ROC AUC:   {roc_auc:.4f}")
    
    return results

def main():
    """Main execution function."""
    print("🚀 Starting Decision Tree Hyperparameter Search")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Initialize results dictionary
    models = {}
    
    # Run grid search
    try:
        models['GridSearchCV'] = run_grid_search(X_train, y_train)
    except Exception as e:
        print(f"❌ GridSearchCV failed: {e}")
    
    # Run randomized search  
    try:
        models['RandomizedSearchCV'] = run_randomized_search(X_train, y_train)
    except Exception as e:
        print(f"❌ RandomizedSearchCV failed: {e}")
    
    # Evaluate models
    if models:
        results = evaluate_models(models, X_test, y_test)
        
        # Show summary
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        
        for name, metrics in results.items():
            print(f"\n🏆 {name} - Best Results:")
            print(f"   Test F1 Score: {metrics['test_f1']:.4f}")
            print(f"   Test ROC AUC:  {metrics['test_roc_auc']:.4f}")
        
        # Cache management info
        cache = ModelCache()
        print(f"\n💾 Cached models available for notebook analysis:")
        cache.list_cached_models()
        
    print(f"\n✅ Hyperparameter search completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Results are cached and ready for analysis in notebooks!")
    print("   Use: from model_cache import ModelCache; cache = ModelCache(); cache.list_cached_models()")

if __name__ == "__main__":
    main()