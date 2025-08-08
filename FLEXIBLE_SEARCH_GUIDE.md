# Flexible Hyperparameter Search Guide

This comprehensive guide shows you how to use the flexible hyperparameter search script to tune DecisionTreeClassifier, RandomForestClassifier, and XGBClassifier with any dataset and custom parameter selection.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Complete Parameter Reference](#complete-parameter-reference)
3. [Usage Examples](#usage-examples)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Integration with Analysis Tools](#integration-with-analysis-tools)

## Quick Start

### 1. Basic Usage

The most basic usage requires specifying a classifier, dataset, target column, and parameters to tune:

```bash
python flexible_hyperparameter_search.py \
    --classifier decision_tree \
    --dataset my_data.csv \
    --target outcome \
    --tune-params max_depth,criterion
```

### 2. Getting Help

See all available parameters for any classifier:

```bash
# Show parameters for decision trees
python flexible_hyperparameter_search.py --help-params decision_tree

# Show parameters for random forests  
python flexible_hyperparameter_search.py --help-params random_forest

# Show parameters for XGBoost
python flexible_hyperparameter_search.py --help-params xgboost

# Show example commands
python flexible_hyperparameter_search.py --examples
```

### 3. Your First Search

Try this with the included click-rate dataset:

```bash
python flexible_hyperparameter_search.py \
    --classifier decision_tree \
    --dataset click-rate-train.csv \
    --target click \
    --drop id,hour,device_id,device_ip \
    --tune-params max_depth,min_samples_split
```

## Complete Parameter Reference

### DecisionTreeClassifier

| Parameter | Description | Values | Impact |
|-----------|-------------|---------|---------|
| `max_depth` | Maximum depth of the tree | `[3, 5, 7, 10, 15, 20, None]` | Controls overfitting vs underfitting |
| `min_samples_split` | Min samples required to split a node | `[2, 5, 10, 20, 50]` | Higher values prevent overfitting |
| `min_samples_leaf` | Min samples required in a leaf node | `[1, 2, 4, 8, 16]` | Smooths model, reduces overfitting |
| `criterion` | Function to measure split quality | `['gini', 'entropy']` | Different impurity measures |
| `max_features` | Features to consider for best split | `['sqrt', 'log2', None]` | Introduces randomness, speeds up training |
| `splitter` | Strategy to split at each node | `['best', 'random']` | Best finds optimal splits, random is faster |

**Common Parameter Combinations:**
- **Quick tuning**: `max_depth,criterion`
- **Overfitting control**: `max_depth,min_samples_split,min_samples_leaf`
- **Full optimization**: `max_depth,min_samples_split,min_samples_leaf,criterion,max_features`

### RandomForestClassifier

| Parameter | Description | Values | Impact |
|-----------|-------------|---------|---------|
| `n_estimators` | Number of trees in the forest | `[50, 100, 200, 300, 500]` | More trees = better performance, slower training |
| `max_depth` | Maximum depth of trees | `[5, 10, 15, 20, None]` | Controls individual tree complexity |
| `min_samples_split` | Min samples to split a node | `[2, 5, 10, 20]` | Prevents overfitting |
| `min_samples_leaf` | Min samples in leaf nodes | `[1, 2, 4, 8]` | Smooths individual trees |
| `max_features` | Features per split consideration | `['sqrt', 'log2', None]` | Controls randomness and diversity |
| `criterion` | Split quality measure | `['gini', 'entropy']` | Usually minimal impact in forests |
| `bootstrap` | Whether to bootstrap samples | `[True, False]` | True adds randomness, False uses all data |
| `max_samples` | Fraction of samples to draw | `[0.5, 0.7, 0.8, 1.0]` | Controls training data per tree |

**Common Parameter Combinations:**
- **Basic tuning**: `n_estimators,max_depth`
- **Randomness control**: `n_estimators,max_features,bootstrap`
- **Full optimization**: `n_estimators,max_depth,min_samples_split,max_features,bootstrap`

### XGBClassifier

| Parameter | Description | Values | Impact |
|-----------|-------------|---------|---------|
| `n_estimators` | Number of boosting rounds | `[50, 100, 200, 300, 500]` | More rounds = better fit, risk of overfitting |
| `max_depth` | Maximum depth of trees | `[3, 4, 5, 6, 7, 8]` | Deeper trees capture more patterns |
| `learning_rate` | Step size shrinkage | `[0.01, 0.05, 0.1, 0.15, 0.2]` | Lower = more conservative learning |
| `subsample` | Fraction of samples for training | `[0.6, 0.8, 0.9, 1.0]` | Prevents overfitting through sampling |
| `colsample_bytree` | Fraction of features per tree | `[0.6, 0.8, 0.9, 1.0]` | Feature sampling for diversity |
| `reg_alpha` | L1 regularization term | `[0, 0.1, 0.5, 1.0, 2.0]` | Sparsity-inducing regularization |
| `reg_lambda` | L2 regularization term | `[0, 0.1, 0.5, 1.0, 2.0]` | Ridge-like regularization |
| `gamma` | Min loss reduction for split | `[0, 0.1, 0.2, 0.5]` | Makes algorithm conservative |

**Common Parameter Combinations:**
- **Learning control**: `learning_rate,n_estimators`
- **Regularization**: `reg_alpha,reg_lambda,gamma`
- **Sampling strategy**: `subsample,colsample_bytree`
- **Full optimization**: `n_estimators,max_depth,learning_rate,subsample,reg_alpha,reg_lambda`

## Custom Parameter Values

### Overview

By default, the script uses predefined parameter value ranges for each classifier. However, you can specify your own custom values for any parameter while still using defaults for others not specified.

### Supported Formats

#### 1. JSON Format

Specify parameters as a JSON object:

```bash
python flexible_hyperparameter_search.py \
    --classifier decision_tree \
    --dataset my_data.csv \
    --target outcome \
    --tune-params max_depth,criterion \
    --param-values '{"max_depth": [5, 10, 15], "criterion": ["gini"]}'
```

#### 2. Key-Value Format

Specify parameters as space-separated key=value pairs:

```bash
python flexible_hyperparameter_search.py \
    --classifier random_forest \
    --dataset my_data.csv \
    --target outcome \
    --tune-params n_estimators,max_depth \
    --param-values "n_estimators=100,200,300 max_depth=10,15,20"
```

#### 3. Mixed Approach (Recommended)

Specify custom values for some parameters while using defaults for others:

```bash
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset my_data.csv \
    --target outcome \
    --tune-params learning_rate,reg_alpha,n_estimators \
    --param-values '{"learning_rate": [0.05, 0.1, 0.15]}'
    # reg_alpha and n_estimators will use default value ranges
```

### Value Type Conversion

The script automatically handles type conversion for key-value format:

- **Numbers**: `"5"` → `5`, `"0.1"` → `0.1`
- **Booleans**: `"true"` → `True`, `"false"` → `False`
- **None**: `"none"` → `None`
- **Strings**: Anything else remains as string

### Examples by Format

#### JSON Examples

```bash
# Decision Tree with custom depth and criterion
--param-values '{"max_depth": [3, 5, 7], "criterion": ["gini", "entropy"]}'

# Random Forest with custom estimators
--param-values '{"n_estimators": [50, 100, 150], "bootstrap": [true, false]}'

# XGBoost with learning rate and regularization
--param-values '{"learning_rate": [0.01, 0.1, 0.2], "reg_alpha": [0, 0.5, 1.0]}'
```

#### Key-Value Examples

```bash
# Decision Tree
--param-values "max_depth=3,5,7 criterion=gini,entropy"

# Random Forest  
--param-values "n_estimators=50,100,150 bootstrap=true,false"

# XGBoost
--param-values "learning_rate=0.01,0.1,0.2 reg_alpha=0,0.5,1.0"
```

### Validation

The script validates your custom parameter values:

- **Parameter names** must be valid for the chosen classifier
- **Parameter names** must be included in your `--tune-params` list  
- **Value ranges** are checked for reasonableness (e.g., positive numbers where expected)
- **Value types** must be compatible with the parameter

### Best Practices for Custom Values

1. **Start Conservative**: Use narrower ranges than defaults for initial exploration
2. **Focus on Promising Regions**: Use insights from previous searches to guide ranges
3. **Consider Computational Cost**: Fewer values = faster search
4. **Mix Custom and Default**: Specify custom values only for parameters you want to focus on

## Usage Examples

### Example 1: Decision Tree - Quick Depth Tuning

Perfect for initial exploration and understanding your dataset:

```bash
python flexible_hyperparameter_search.py \
    --classifier decision_tree \
    --dataset click-rate-train.csv \
    --target click \
    --drop id,hour,device_id,device_ip \
    --tune-params max_depth,criterion \
    --cv 5
```

**When to use:** First-time analysis, simple datasets, interpretability is key.

### Example 1b: Decision Tree - Custom Parameter Values

Fine-tune with specific depth and split values:

```bash
python flexible_hyperparameter_search.py \
    --classifier decision_tree \
    --dataset click-rate-train.csv \
    --target click \
    --drop id,hour,device_id,device_ip \
    --tune-params max_depth,min_samples_split \
    --param-values '{"max_depth": [5, 10, 15], "min_samples_split": [2, 5]}' \
    --cv 5
```

**When to use:** When you have insights about optimal parameter ranges from previous experiments.

### Example 2: Random Forest - Comprehensive Search

Balance between performance and computational cost:

```bash
python flexible_hyperparameter_search.py \
    --classifier random_forest \
    --dataset large_dataset.csv \
    --target binary_outcome \
    --drop user_id,timestamp \
    --tune-params n_estimators,max_depth,max_features,bootstrap \
    --search-type randomized \
    --n-iter 100 \
    --cv 3 \
    --scoring roc_auc
```

**When to use:** Medium to large datasets, when you need good performance without extensive tuning.

### Example 2b: Random Forest - Custom Estimator Range

Use custom number of estimators while using defaults for other parameters:

```bash
python flexible_hyperparameter_search.py \
    --classifier random_forest \
    --dataset large_dataset.csv \
    --target binary_outcome \
    --drop user_id,timestamp \
    --tune-params n_estimators,max_depth,max_features \
    --param-values "n_estimators=150,250,350" \
    --search-type randomized \
    --n-iter 50 \
    --cv 3 \
    --scoring roc_auc
```

**When to use:** When you want to test specific estimator counts while exploring other parameters broadly.

### Example 3: XGBoost - Learning Rate and Regularization

Fine-tune the most important XGBoost parameters:

```bash
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset competition_data.csv \
    --target target_variable \
    --drop id,date,irrelevant_col \
    --tune-params learning_rate,n_estimators,reg_alpha,reg_lambda \
    --search-type grid \
    --cv 5 \
    --scoring f1 \
    --verbose 2
```

**When to use:** When you need maximum performance and have computational resources.

### Example 3b: XGBoost - Focused Learning Rate Tuning

Custom learning rates with default values for regularization:

```bash
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset competition_data.csv \
    --target target_variable \
    --drop id,date,irrelevant_col \
    --tune-params learning_rate,reg_alpha,reg_lambda \
    --param-values '{"learning_rate": [0.03, 0.07, 0.12]}' \
    --search-type grid \
    --cv 5 \
    --scoring f1 \
    --verbose 2
```

**When to use:** When you want to fine-tune specific parameters while exploring others with default ranges.

### Example 4: Multi-stage Tuning Strategy

Start broad, then narrow down:

```bash
# Stage 1: Broad search with key parameters using defaults
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset my_data.csv \
    --target outcome \
    --tune-params n_estimators,learning_rate,max_depth \
    --search-type randomized \
    --n-iter 50

# Stage 2: Fine-tune around best values from stage 1
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset my_data.csv \
    --target outcome \
    --tune-params learning_rate,reg_alpha,reg_lambda \
    --param-values '{"learning_rate": [0.08, 0.10, 0.12]}' \
    --search-type grid

# Stage 3: Final regularization tuning with custom ranges
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset my_data.csv \
    --target outcome \
    --tune-params reg_alpha,reg_lambda,gamma \
    --param-values "reg_alpha=0.1,0.3,0.5 reg_lambda=0.5,1.0,1.5" \
    --search-type grid
```

### Example 5: Different Scoring Metrics

Choose the right metric for your problem:

```bash
# For imbalanced datasets
python flexible_hyperparameter_search.py \
    --classifier random_forest \
    --dataset imbalanced_data.csv \
    --target rare_event \
    --tune-params n_estimators,max_depth,min_samples_leaf \
    --scoring precision  # or recall, f1, roc_auc

# For multiclass problems
python flexible_hyperparameter_search.py \
    --classifier decision_tree \
    --dataset multiclass_data.csv \
    --target category \
    --tune-params max_depth,min_samples_split \
    --scoring accuracy
```

## Advanced Features

### 1. Custom Scoring Metrics

Supported scoring metrics:
- `accuracy` - Overall correctness
- `precision` - Positive prediction accuracy  
- `recall` - True positive detection rate
- `f1` - Harmonic mean of precision and recall
- `roc_auc` - Area under ROC curve (binary only)

### 2. Cross-Validation Configuration

```bash
# Use different CV strategies
--cv 3   # 3-fold CV (faster)
--cv 5   # 5-fold CV (balanced)  
--cv 10  # 10-fold CV (more reliable, slower)
```

### 3. Search Type Selection

```bash
# Grid search - exhaustive but potentially expensive
--search-type grid

# Randomized search - efficient for large parameter spaces
--search-type randomized --n-iter 100
```

### 4. Parallel Processing

```bash
# Use all available cores (default)
--n-jobs -1

# Use specific number of cores
--n-jobs 4

# Use single core (for debugging)
--n-jobs 1
```

### 4. Custom Parameter Values Integration

```bash
# Combine custom values with other advanced features
python flexible_hyperparameter_search.py \
    --classifier xgboost \
    --dataset large_data.csv \
    --target outcome \
    --tune-params learning_rate,n_estimators,reg_alpha \
    --param-values '{"learning_rate": [0.05, 0.1]}' \
    --search-type randomized \
    --n-iter 50 \
    --cv 3 \
    --scoring roc_auc \
    --n-jobs -1
```

### 5. Dataset Path Flexibility

The script handles various dataset path formats:

```bash
# Absolute path
--dataset /full/path/to/data.csv

# Relative path
--dataset ./data/my_file.csv

# Using config system (for configured datasets)
--dataset click-rate-train.csv  # Automatically resolved

# Current directory
--dataset data.csv
```

## Performance Optimization

### 1. Parameter Selection Strategy

**Start Small, Scale Up:**
1. Begin with 2-3 key parameters
2. Use randomized search for initial exploration
3. Use grid search for final fine-tuning

**Parameter Priority by Classifier:**

**Decision Tree:**
1. `max_depth` (biggest impact)
2. `min_samples_split` (prevents overfitting)
3. `criterion` (minor impact, but fast)

**Random Forest:**
1. `n_estimators` (performance vs time tradeoff)
2. `max_depth` (controls complexity)
3. `max_features` (affects diversity)

**XGBoost:**
1. `learning_rate` + `n_estimators` (together)
2. `max_depth` (model complexity)
3. `reg_alpha` + `reg_lambda` (regularization)

### 2. Search Strategy Recommendations

| Dataset Size | Recommended Approach | Search Type | Parameters |
|--------------|---------------------|-------------|------------|
| Small (<10K) | Comprehensive | Grid | 4-6 parameters |
| Medium (10K-100K) | Balanced | Randomized | 3-5 parameters |
| Large (>100K) | Focused | Randomized | 2-4 parameters |

### 3. Computational Considerations

```bash
# For large datasets - reduce CV folds
--cv 3

# For many parameters - use randomized search
--search-type randomized --n-iter 50

# For quick iteration - focus on key parameters
--tune-params max_depth,learning_rate  # instead of many parameters
```

## Troubleshooting

### Common Issues and Solutions

#### 1. XGBoost Not Available
```
Error: XGBoost not available. Install with: uv add xgboost
```
**Solution:** Install XGBoost dependency
```bash
uv add xgboost
```

#### 2. Dataset Not Found
```
FileNotFoundError: Dataset not found: my_data.csv
```
**Solutions:**
- Use absolute path: `--dataset /full/path/to/data.csv`
- Ensure file exists in current directory
- Check if dataset is configured in config system

#### 3. Invalid Target Column
```
ValueError: Target column 'outcome' not found
```
**Solution:** Check column names in your dataset
```bash
# First check your data columns
python -c "import pandas as pd; print(pd.read_csv('data.csv').columns.tolist())"
```

#### 4. Invalid Parameters
```
ValueError: Invalid parameters for decision_tree: ['n_estimators']
```
**Solution:** Use `--help-params` to see valid parameters
```bash
python flexible_hyperparameter_search.py --help-params decision_tree
```

#### 5. Memory Issues with Large Datasets
- Reduce CV folds: `--cv 3`
- Use fewer parameter combinations
- Consider using `--n-jobs 1` to reduce memory usage
- Use randomized search instead of grid search

#### 6. Long Running Times
- Start with fewer parameters or smaller custom value ranges
- Use randomized search: `--search-type randomized --n-iter 20`
- Reduce CV folds: `--cv 3`
- Use custom parameter values to focus search: `--param-values '{"max_depth": [5, 10]}'`
- Check if results are cached (re-runs should be instant)

#### 7. Invalid Custom Parameter Values
```
ValueError: Parameter 'max_depth' must have positive numerical values or None, got -5
```
**Solution:** Ensure custom values are appropriate for the parameter
```bash
# Check parameter constraints with help
python flexible_hyperparameter_search.py --help-params decision_tree

# Use valid ranges
--param-values '{"max_depth": [5, 10, 15], "criterion": ["gini", "entropy"]}'
```

#### 8. JSON Format Errors
```
ValueError: Invalid JSON format in --param-values
```
**Solutions:**
- Use proper JSON syntax with double quotes: `'{"param": [1, 2, 3]}'`
- Alternative: Use key-value format: `"param=1,2,3"`
- Escape quotes properly in shell: `'{"param": ["value1", "value2"]}'`

#### 9. Multiprocessing Warnings on macOS
```
Exception ignored in: <function ResourceTracker.__del__ at 0x...>
```
**Solution:** The script now automatically handles this issue on macOS. If you still see warnings:
- This is a known Python multiprocessing issue on macOS
- The warnings don't affect functionality - your search will complete successfully
- For troubleshooting, use `--n-jobs 1` to disable parallel processing temporarily

### Performance Debugging

```bash
# Enable verbose output to see progress
--verbose 2

# Check what parameters are being tuned
# The script shows the parameter grid before starting

# Monitor system resources
# Use tools like htop/Activity Monitor to check CPU/memory usage
```

## Integration with Analysis Tools

### 1. Using with Existing Analysis Notebook

The script is compatible with the existing `analyze_search_results.qmd` notebook, but works best with the new `analyze_flexible_search_results.qmd`.

### 2. Accessing Cached Results

```python
# In your analysis notebook
from model_cache import ModelCache

cache = ModelCache()
cache.list_cached_models()  # See all available results

# Load specific results
search_result = cache.load_search_result("decision_tree_grid_max_depth_criterion", params)
```

### 3. Analysis Workflow

1. **Run Search:**
   ```bash
   python flexible_hyperparameter_search.py \
       --classifier random_forest \
       --dataset my_data.csv \
       --target outcome \
       --tune-params n_estimators,max_depth
   ```

2. **Analyze Results:**
   Open `analyze_flexible_search_results.qmd` and run the analysis

3. **Iterate:**
   Based on analysis, run focused searches on promising parameter regions

### 4. Comparing Multiple Searches

You can run multiple searches with different parameter combinations and compare them:

```bash
# Search 1: Focus on tree structure
python flexible_hyperparameter_search.py \
    --classifier random_forest \
    --dataset data.csv \
    --target outcome \
    --tune-params n_estimators,max_depth

# Search 2: Focus on sampling
python flexible_hyperparameter_search.py \
    --classifier random_forest \
    --dataset data.csv \
    --target outcome \
    --tune-params bootstrap,max_samples,max_features
```

Then compare results in the analysis notebook.

## Comprehensive Metrics Storage

### Overview

The flexible hyperparameter search system automatically stores comprehensive metrics and metadata for every search run. This goes far beyond just the cross-validation score, providing a complete picture of model performance, training efficiency, and data characteristics.

### What Metrics Are Automatically Stored

#### 1. **Test Set Performance Metrics**
```python
'test_metrics': {
    'accuracy': 0.8542,      # Overall correctness
    'precision': 0.8123,     # True positive rate among predictions
    'recall': 0.7891,        # True positive detection rate  
    'f1': 0.8005,           # Harmonic mean of precision/recall
    'roc_auc': 0.8734       # Area under ROC curve (binary only)
}
```

**Why useful:** Compare CV performance vs real-world test performance to detect overfitting.

#### 2. **Dataset Information**
```python
'data_info': {
    'train_size': 15000,                    # Training samples
    'test_size': 6429,                      # Test samples  
    'n_features': 47,                       # Number of features
    'feature_names': ['age', 'income', ...], # Feature list
    'target_distribution_train': {0: 8500, 1: 6500},  # Class balance
    'target_distribution_test': {0: 3600, 1: 2829},   # Test distribution
    'test_size_ratio': 0.3                 # Train/test split ratio
}
```

**Why useful:** Understand data characteristics, detect class imbalance, track feature changes over time.

#### 3. **Training Efficiency Metrics**
```python  
'training_info': {
    'search_type': 'randomized',        # Grid or randomized search
    'total_combinations_tested': 50,    # Parameter combinations evaluated
    'training_time_minutes': 12.3,      # Total training time
    'custom_params_used': true,         # Whether custom values were used
    'cv_folds': 5,                     # Cross-validation folds
    'scoring_metric': 'f1',            # Optimization metric
    'n_jobs': -1                       # Parallel processing used
}
```

**Why useful:** Track computational efficiency, optimize search strategies, compare training costs.

#### 4. **Classification Details**
```python
'classification_details': {
    'confusion_matrix': [[1800, 200], [150, 1279]], # Prediction accuracy matrix
    'classification_report': {                       # Detailed per-class metrics
        '0': {'precision': 0.92, 'recall': 0.90, ...},
        '1': {'precision': 0.86, 'recall': 0.89, ...}
    }
}
```

**Why useful:** Understand model behavior per class, identify misclassification patterns.

#### 5. **Model-Specific Architecture**
```python
'model_specific': {
    'feature_importances': [0.15, 0.08, 0.12, ...], # Feature importance scores
    'tree_depth': 8,                                 # Decision tree depth
    'n_leaves': 127,                                # Decision tree leaves
    'n_estimators': 200                             # Forest/boosting trees
}
```

**Why useful:** Understand model complexity, feature contributions, compare architectures.

#### 6. **Advanced Probability Metrics**
```python
'probability_metrics': {
    'log_loss': 0.342,           # Probabilistic accuracy
    'average_precision': 0.867,  # Area under precision-recall curve
    'brier_score': 0.156        # Probability calibration quality
}
```

**Why useful:** Evaluate probability quality, not just classification accuracy.

### How to Access Stored Metrics

#### Programmatic Access
```python
from model_cache import ModelCache

cache = ModelCache()
models = cache.list_cached_models()

# Get specific model metrics
model = models[0]  # Most recent model
print(f"Test F1: {model['test_metrics']['f1']:.3f}")
print(f"Training time: {model['training_info']['training_time_minutes']:.1f} min")
print(f"Feature count: {model['data_info']['n_features']}")
```

#### Analysis Notebook Access
The `analyze_flexible_search_results.qmd` notebook automatically displays all stored metrics:
- Performance comparison charts (CV vs Test)
- Training efficiency visualizations  
- Feature importance plots
- Data distribution analysis
- Model architecture details

### Use Cases and Benefits

#### 1. **Experiment Tracking**
```python
# Compare multiple experiments
for model in models:
    print(f"{model['model_name']}: F1={model['test_metrics']['f1']:.3f}, "
          f"Time={model['training_info']['training_time_minutes']:.1f}min")
```

#### 2. **Performance Monitoring**
- Track model performance over time
- Detect data drift through distribution changes
- Monitor training efficiency trends

#### 3. **Model Selection**
```python
# Find best performing model considering multiple criteria
best_model = max(models, key=lambda x: (
    x['test_metrics']['f1'] * 0.7 +          # Performance weight
    (1 - x['training_info']['training_time_minutes']/60) * 0.3  # Efficiency weight
))
```

#### 4. **Resource Planning**
- Estimate training times for similar searches
- Plan computational resources based on historical data
- Optimize search strategies using efficiency metrics

#### 5. **Feature Engineering Insights**  
```python
# Identify consistently important features across models
important_features = []
for model in models:
    if model['model_specific']['feature_importances']:
        # Get top features for each model
        importances = model['model_specific']['feature_importances']
        feature_names = model['data_info']['feature_names']
        # Analysis of feature importance patterns...
```

### Best Practices for Using Comprehensive Metrics

#### 1. **Regular Analysis**
- Review metrics after each search run
- Look for overfitting (large CV vs Test gaps)
- Monitor training efficiency trends

#### 2. **Comparative Analysis**
- Compare similar models across different parameter ranges
- Analyze trade-offs between performance and training time
- Track improvements over multiple iterations

#### 3. **Documentation**
- Keep notes on successful parameter combinations
- Document insights from feature importance analysis
- Track data changes and their impacts

#### 4. **Performance Optimization**
- Use training time metrics to optimize search strategies
- Focus on promising parameter regions based on historical results
- Balance performance gains with computational costs

### Example Analysis Workflow

```python
# 1. Run hyperparameter search
python flexible_hyperparameter_search.py --classifier xgboost --dataset data.csv --target outcome --tune-params learning_rate,reg_alpha

# 2. Analyze comprehensive results
# Open analyze_flexible_search_results.qmd and examine:
# - CV vs Test performance comparison
# - Feature importance rankings  
# - Training efficiency metrics
# - Data distribution analysis

# 3. Make informed decisions
# - Adjust parameter ranges based on importance patterns
# - Optimize search strategy using efficiency insights
# - Plan next experiments using comprehensive data
```

The comprehensive metrics system transforms the tool from a simple hyperparameter search into a complete ML experiment tracking platform, providing insights that go far beyond basic model performance.

## Best Practices Summary

### 1. Parameter Selection
- Start with 2-3 most impactful parameters
- Use domain knowledge to guide selection
- Consider parameter interactions

### 2. Search Strategy
- Use randomized search for exploration
- Use grid search for final optimization
- Consider computational budget

### 3. Evaluation
- Choose appropriate scoring metric for your problem
- Use stratified CV for imbalanced datasets
- Always validate on held-out test set

### 4. Custom Parameter Usage
- Start with default ranges to understand parameter impact
- Use custom values to focus on promising regions discovered in initial searches
- Combine custom values for critical parameters with defaults for exploration
- Document successful custom value combinations for future use

### 5. Iteration
- Analyze results before expanding parameter search
- Use custom parameter values to focus on promising parameter regions
- Move from broad default ranges to focused custom ranges
- Document your findings and successful parameter combinations

---

**Need Help?**
- Use `--help-params <classifier>` to see available parameters
- Use `--examples` to see usage examples  
- Check the troubleshooting section for common issues
- All results are cached - re-running with same parameters loads instantly!