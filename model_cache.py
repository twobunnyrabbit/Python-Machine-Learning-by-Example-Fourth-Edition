"""
Model caching utility for expensive operations like GridSearchCV and RandomizedSearchCV.

This module provides utilities to cache fitted models and search results to avoid 
recomputing expensive operations when parameters haven't changed.
"""

import pickle
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import warnings


class ModelCache:
    """
    Cache for storing and retrieving fitted models and search results.
    
    The cache uses parameter hashing to determine if a model needs to be retrained
    or if cached results can be used.
    """
    
    def __init__(self, cache_dir="model_cache"):
        """
        Initialize the model cache.
        
        Args:
            cache_dir (str): Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _hash_params(self, params_dict):
        """Create a hash from parameters dictionary."""
        # Convert to JSON string with sorted keys for consistent hashing
        params_str = json.dumps(params_dict, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _get_cache_path(self, model_name, params_hash):
        """Get the cache file path for a model."""
        return self.cache_dir / f"{model_name}_{params_hash}.pkl"
    
    def _get_metadata_path(self, model_name, params_hash):
        """Get the metadata file path for a model."""
        return self.cache_dir / f"{model_name}_{params_hash}_meta.json"
    
    def save_search_result(self, model_name, search_object, params_dict, data_hash=None, additional_metrics=None):
        """
        Save a GridSearchCV or RandomizedSearchCV result to cache.
        
        Args:
            model_name (str): Name identifier for the model
            search_object: Fitted GridSearchCV or RandomizedSearchCV object
            params_dict (dict): Dictionary of all parameters that affect the result
            data_hash (str, optional): Hash of the training data to detect data changes
            additional_metrics (dict, optional): Additional metrics and metadata to store
        """
        params_hash = self._hash_params(params_dict)
        cache_path = self._get_cache_path(model_name, params_hash)
        metadata_path = self._get_metadata_path(model_name, params_hash)
        
        # Save the search object
        with open(cache_path, 'wb') as f:
            pickle.dump(search_object, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'params_dict': params_dict,
            'data_hash': data_hash,
            'best_params': search_object.best_params_,
            'best_score': search_object.best_score_
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            metadata.update(additional_metrics)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Cached model '{model_name}' with score {search_object.best_score_:.4f}")
    
    def load_search_result(self, model_name, params_dict, data_hash=None):
        """
        Load a cached GridSearchCV or RandomizedSearchCV result.
        
        Args:
            model_name (str): Name identifier for the model
            params_dict (dict): Dictionary of all parameters that affect the result
            data_hash (str, optional): Hash of the training data to detect data changes
        
        Returns:
            Fitted search object or None if not found/invalid
        """
        params_hash = self._hash_params(params_dict)
        cache_path = self._get_cache_path(model_name, params_hash)
        metadata_path = self._get_metadata_path(model_name, params_hash)
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        # Check metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate data hash if provided
        if data_hash and metadata.get('data_hash') != data_hash:
            print(f"⚠️  Data changed for '{model_name}', cache invalid")
            return None
        
        # Load the search object
        try:
            with open(cache_path, 'rb') as f:
                search_object = pickle.load(f)
            
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            print(f"🔄 Loaded cached model '{model_name}' from {timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Best score: {metadata['best_score']:.4f}")
            print(f"   Best params: {metadata['best_params']}")
            
            return search_object
            
        except Exception as e:
            print(f"❌ Error loading cached model '{model_name}': {e}")
            return None
    
    def clear_cache(self, model_name=None):
        """
        Clear cached models.
        
        Args:
            model_name (str, optional): Clear only this model. If None, clears all.
        """
        if model_name:
            # Clear specific model
            for file in self.cache_dir.glob(f"{model_name}_*.pkl"):
                file.unlink()
            for file in self.cache_dir.glob(f"{model_name}_*_meta.json"):
                file.unlink()
            print(f"Cleared cache for model '{model_name}'")
        else:
            # Clear all
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            for file in self.cache_dir.glob("*_meta.json"):
                file.unlink()
            print("Cleared all cached models")
    
    def list_cached_models(self):
        """List all cached models with their metadata."""
        cached_models = []
        
        for metadata_file in self.cache_dir.glob("*_meta.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                cached_models.append(metadata)
            except Exception as e:
                print(f"Error reading {metadata_file}: {e}")
        
        if cached_models:
            print("Cached Models:")
            print("-" * 80)
            for model in sorted(cached_models, key=lambda x: x['timestamp'], reverse=True):
                timestamp = datetime.fromisoformat(model['timestamp'])
                print(f"📊 {model['model_name']}")
                print(f"   Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Best Score: {model['best_score']:.4f}")
                print(f"   Best Params: {model['best_params']}")
                print()
        else:
            print("No cached models found.")
        
        return cached_models


def hash_data(X, y):
    """
    Create a hash of the training data to detect changes.
    
    Args:
        X: Features dataframe/array
        y: Target series/array
    
    Returns:
        str: Hash of the data
    """
    try:
        import pandas as pd
        if hasattr(X, 'values'):  # DataFrame
            X_bytes = pd.util.hash_pandas_object(X).values.tobytes()
        else:  # numpy array
            X_bytes = X.tobytes()
        
        if hasattr(y, 'values'):  # Series
            y_bytes = pd.util.hash_pandas_object(y).values.tobytes()
        else:  # numpy array
            y_bytes = y.tobytes()
        
        combined = X_bytes + y_bytes
        return hashlib.md5(combined).hexdigest()
    
    except Exception as e:
        warnings.warn(f"Could not hash data: {e}. Data change detection disabled.")
        return None


# Convenience function for easy usage
def cached_grid_search(estimator, param_grid, X_train, y_train, model_name, 
                      cv=5, scoring='accuracy', n_jobs=1, cache_dir="model_cache", additional_metrics=None, **kwargs):
    """
    Perform GridSearchCV with automatic caching.
    
    Args:
        estimator: The estimator to tune
        param_grid: Parameter grid for search
        X_train, y_train: Training data
        model_name: Unique name for this model configuration
        cv, scoring, n_jobs: GridSearchCV parameters
        cache_dir: Directory for cache storage
        additional_metrics: Additional metrics and metadata to store
        **kwargs: Additional GridSearchCV parameters
    
    Returns:
        Fitted GridSearchCV object
    """
    from sklearn.model_selection import GridSearchCV
    
    # Create cache instance
    cache = ModelCache(cache_dir)
    
    # Create parameter dictionary for hashing
    params_dict = {
        'estimator': str(type(estimator).__name__),
        'param_grid': param_grid,
        'cv': cv,
        'scoring': scoring,
        'estimator_params': estimator.get_params(),
        **kwargs
    }
    
    # Hash training data
    data_hash = hash_data(X_train, y_train)
    
    # Try to load from cache
    cached_result = cache.load_search_result(model_name, params_dict, data_hash)
    if cached_result is not None:
        return cached_result
    
    # Run grid search
    print(f"🔍 Running GridSearchCV for '{model_name}'...")
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        **kwargs
    )
    
    grid_search.fit(X_train, y_train)
    
    # Save to cache
    cache.save_search_result(model_name, grid_search, params_dict, data_hash, additional_metrics)
    
    return grid_search