# Python-Machine-Learning-by-Example-Fourth-Edition
Python Machine Learning by Example, Fourth Edition

I forked this from the original repo on 2025-04-15 and added the following:
- converted it to use `uv` for package installation
- most dependencies are installed via `uv add <package_name> ...`
- `pytorch` was not able to be installed this way, instead:
    `uv pip install torch torchvision`

## Dataset Configuration

Datasets are stored externally and accessed via a global configuration system using environment variables.

### Setup Instructions

1. **Copy datasets** to your local machine (from OneDrive or other source)
2. **Create a `.env` file** in the repository root with your dataset paths:
   ```env
   # Dataset paths configuration
   DATA_ROOT=/path/to/your/datasets/Python-Machine-Learning-by-Example-Fourth-Edition/data
   MOVIELENS_PATH=${DATA_ROOT}/ch02/ml-1m
   CLICK_RATE_PATH=${DATA_ROOT}/ch03
   ```
3. **Install dependencies** including python-dotenv and xgboost:
   ```bash
   uv sync
   ```

### Usage in Code

**For Interactive Sessions (Quarto/Jupyter Notebooks) - Copy/Paste Template:**

Use this code block at the start of any new `.qmd` or `.ipynb` file:

```python
# Setup for dataset access in interactive environments
import pandas as pd
import sys
import os
from pathlib import Path

# Add repository root to Python path for config import
# This automatically finds the repo root containing config.py
repo_root = Path.cwd()
while repo_root != repo_root.parent:
    if (repo_root / 'config.py').exists():
        break
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

# Import dataset configuration functions
from config import get_movielens_file, get_click_rate_file, get_data_path

# Examples of loading datasets:
# data_path = get_movielens_file('ratings.dat')           # MovieLens data
# data_path = get_click_rate_file('click-rate-train.csv') # Click rate data  
# data_path = get_data_path(5, 'some-file.csv')          # Chapter 5 data
# df = pd.read_csv(data_path)

# Optional: Import model caching utility for expensive operations
# from model_cache import cached_grid_search, ModelCache
```

**For Python Scripts (.py files):**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import get_movielens_file, get_data_path

data_path = get_movielens_file('ratings.dat')
df = pd.read_csv(data_path)
```

**Available Dataset Functions:**
- `get_movielens_file(filename)` - MovieLens dataset files (ch02)
- `get_click_rate_file(filename)` - Click rate dataset files (ch03)  
- `get_data_path(chapter, filename)` - Any dataset file by chapter number

## Troubleshooting Interactive Environments

**Multiprocessing Warnings in GridSearchCV/RandomizedSearchCV:**

**For the Flexible Hyperparameter Search Script:** The script now automatically handles multiprocessing warnings on macOS. No user action needed!

**For Interactive Environments (Quarto/Jupyter):** If you encounter multiprocessing warnings, add this code before your grid search:

```python
# Fix for multiprocessing warnings in interactive environments (Quarto/Jupyter)
import os
import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Option 1: Disable parallel processing to avoid multiprocessing issues
grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=1,  # Use single core instead of n_jobs=-1
    verbose=1
)

# Option 2: Set multiprocessing start method (add at the top of your notebook)
import multiprocessing as mp
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Option 3: Suppress warnings (not recommended for debugging)
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
```

**Recommended approach:** Use `n_jobs=1` in interactive environments to avoid multiprocessing issues while maintaining functionality.

### Best Practice: Use Python Scripts for Expensive Operations

**For long-running operations like GridSearchCV/RandomizedSearchCV, consider using Python scripts instead of interactive notebooks:**

**Benefits of Python scripts:**
- ✅ Full multiprocessing support (`n_jobs=-1` works properly)
- ✅ Better memory management and performance
- ✅ No risk of losing work if notebook crashes
- ✅ Can run in background while you work on other tasks
- ✅ Better error handling and logging

**Workflow:**
1. **Develop/prototype** in Quarto notebooks with small parameter grids
2. **Run expensive searches** in Python scripts with full parameter grids
3. **Analyze results** back in interactive notebooks

**Example script structure:**
```python
# hyperparameter_search.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_click_rate_file
from model_cache import cached_grid_search
from sklearn.tree import DecisionTreeClassifier
# ... other imports

if __name__ == "__main__":
    # Load and prepare data
    # Run expensive search with n_jobs=-1
    # Results automatically cached for notebook analysis
```

**Run the script:**
```bash
cd ch3/
python hyperparameter_search.py
```

## Model Caching for Expensive Operations

**Save time on repeated GridSearchCV/RandomizedSearchCV runs:**

The repository includes a `model_cache.py` utility to automatically cache expensive search results:

```python
# Easy way - use the convenience function
from model_cache import cached_grid_search

# This will cache results and reuse them if parameters haven't changed
grid_search = cached_grid_search(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid={'max_depth': [3, 5, 7], 'min_samples_split': [2, 5]},
    X_train=X_train, 
    y_train=y_train,
    model_name="decision_tree_clickrate",  # Unique identifier
    cv=5,
    scoring='f1',
    n_jobs=1
)

# Manual way - more control
from model_cache import ModelCache
cache = ModelCache("my_models")  # Cache directory

# Try to load cached result
cached_result = cache.load_search_result("my_model", params_dict)
if cached_result is None:
    # Run expensive search
    grid_search = GridSearchCV(...)
    grid_search.fit(X_train, y_train)
    # Save result
    cache.save_search_result("my_model", grid_search, params_dict)
else:
    grid_search = cached_result
```

**Cache Management:**
```python
cache = ModelCache()
cache.list_cached_models()        # Show all cached models
cache.clear_cache("model_name")   # Clear specific model
cache.clear_cache()               # Clear all models
```

## Working Examples

The repository includes several working examples demonstrating these features:

### Chapter 3 - Decision Tree Examples
- **`ch3/decision-tree-exercise.qmd`** - Interactive development with caching
- **`ch3/decision-tree-with-caching-example.qmd`** - Comprehensive caching demo
- **`ch3/hyperparameter_search.py`** - Production script for expensive searches
- **`ch3/analyze_search_results.qmd`** - Analysis of cached search results

### Usage Workflow Examples

**1. Quick Prototyping (Interactive):**
```python
# In Quarto notebook - fast iteration with small parameter grids
from model_cache import cached_grid_search

grid_search = cached_grid_search(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid={'max_depth': [3, 5], 'min_samples_split': [2, 5]},  # Small grid
    X_train=X_train, y_train=y_train,
    model_name="dt_prototype",
    cv=3, n_jobs=1  # Fast for experimentation
)
```

**2. Production Search (Script):**
```bash
# Run comprehensive search in background
cd ch3/
python hyperparameter_search.py
# This runs extensive parameter grids with n_jobs=-1
# Results automatically cached
```

**3. Results Analysis (Interactive):**
```python
# In Quarto notebook - load cached results for analysis
from model_cache import ModelCache
cache = ModelCache()
cached_model = cache.load_search_result("decision_tree_comprehensive_grid", params)
# Rich visualizations and analysis without re-computation
```

## Repository Structure

```
├── config.py                               # Global dataset configuration
├── model_cache.py                          # Caching utility for expensive operations
├── flexible_hyperparameter_search.py       # NEW: Flexible multi-classifier tuning
├── analyze_flexible_search_results.qmd     # NEW: Analysis for flexible searches
├── FLEXIBLE_SEARCH_GUIDE.md               # NEW: Comprehensive usage guide
├── .env                                   # Local dataset paths (not committed)
├── ch2/                                   # Naive Bayes examples
├── ch3/                                   # Decision Trees with caching examples
│   ├── decision-tree-exercise.qmd              # Interactive development
│   ├── decision-tree-with-caching-example.qmd  # Caching demonstration
│   ├── hyperparameter_search.py               # Production hyperparameter search
│   └── analyze_search_results.qmd             # Results analysis
├── data/                                  # Local datasets (gitignored)
└── model_cache/                           # Cached models (gitignored)
```

## Key Features Summary

✅ **Global Dataset Configuration** - Access OneDrive datasets from anywhere  
✅ **Interactive Environment Support** - Works in Quarto, Jupyter, and Python scripts  
✅ **Model Caching System** - Save expensive GridSearchCV/RandomizedSearchCV results  
✅ **Production-Ready Scripts** - Full multiprocessing support for expensive operations  
✅ **Comprehensive Examples** - Working code for all common scenarios  

## Flexible Hyperparameter Search

**NEW: Advanced hyperparameter tuning for multiple classifiers with custom parameter selection**

### Quick Start
```bash
# Decision Tree with default parameter ranges
python flexible_hyperparameter_search.py \
    --classifier decision_tree \
    --dataset click-rate-train.csv \
    --target click \
    --drop id,hour,device_id,device_ip \
    --tune-params max_depth,criterion

# Random Forest with custom parameter values (JSON format)
python flexible_hyperparameter_search.py \
    --classifier random_forest \
    --dataset my_data.csv \
    --target outcome \
    --tune-params n_estimators,max_depth \
    --param-values '{"n_estimators": [100, 200, 300], "max_depth": [10, 15, 20]}'

# XGBoost with custom values (key-value format)
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset my_data.csv \
    --target outcome \
    --tune-params learning_rate,reg_alpha \
    --param-values "learning_rate=0.05,0.1,0.15 reg_alpha=0,0.5,1.0"
```

### Key Features
- ✅ **3 Classifiers Supported**: DecisionTree, RandomForest, XGBoost
- ✅ **User-Selected Parameters**: Only tune what you specify (no forced defaults)
- ✅ **Custom Parameter Values**: Specify your own ranges or use built-in defaults
- ✅ **Flexible Dataset Support**: Any CSV with configurable target/drop columns
- ✅ **Smart Caching**: Compatible with existing analysis tools
- ✅ **Comprehensive Help**: Built-in parameter reference and examples

### Getting Help
```bash
# Show all available parameters for any classifier
python flexible_hyperparameter_search.py --help-params decision_tree
python flexible_hyperparameter_search.py --help-params random_forest  
python flexible_hyperparameter_search.py --help-params xgboost

# Show usage examples (including custom parameter formats)
python flexible_hyperparameter_search.py --examples
```

### Custom Parameter Values

The script supports custom parameter value ranges in two formats:

**JSON Format:**
```bash
--param-values '{"max_depth": [5, 10, 15], "criterion": ["gini", "entropy"]}'
```

**Key-Value Format:**
```bash
--param-values "max_depth=5,10,15 criterion=gini,entropy"
```

**Mixed Usage (Custom + Default):**
```bash
# Custom learning_rate values, default ranges for reg_alpha and n_estimators
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset my_data.csv \
    --target outcome \
    --tune-params learning_rate,reg_alpha,n_estimators \
    --param-values '{"learning_rate": [0.05, 0.1, 0.15]}'
```

### Analysis Tools
- **`analyze_flexible_search_results.qmd`** - Comprehensive analysis of any cached search
- **`FLEXIBLE_SEARCH_GUIDE.md`** - Complete guide with examples, custom parameter usage, and best practices

**Note:** The `.env` file is not committed to the repository for security and portability reasons.
