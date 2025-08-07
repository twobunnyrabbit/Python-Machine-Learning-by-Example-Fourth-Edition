# Decision Tree Techniques Summary

This document provides a comprehensive summary of decision tree techniques implemented in Python using scikit-learn, based on the examples in `decision-tree-scikit.qmd`.

## 1. Basic Decision Tree Implementation

### Key Components:
- **Algorithm**: `DecisionTreeClassifier` from scikit-learn
- **Splitting Criterion**: Gini impurity (default)
- **Key Parameters**:
  - `max_depth`: Maximum depth of the tree (controls overfitting)
  - `min_samples_split`: Minimum samples required to split a node
  - `criterion`: Function to measure quality of a split ('gini' or 'entropy')

### Example Code:
```python
from sklearn.tree import DecisionTreeClassifier

# Initialize and train the model
tree_sk = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
tree_sk.fit(X_train_n, y_train_n)
```

### Visualization:
- Use `export_graphviz` to visualize the tree structure
- Parameters include feature names, impurity display, and class coloring
- Output can be saved as DOT file and converted to PNG

## 2. Data Preparation for Real-World Applications

### Click-Through Rate Prediction Dataset:
- **Source**: Large dataset (6.2GB uncompressed)
- **Sample Size**: 300,000 rows for practical implementation
- **Feature Engineering**:
  - Remove non-predictive columns: 'click', 'id', 'hour', 'device_id', 'device_ip'
  - Convert remaining DataFrame to NumPy array for ML compatibility

### Data Splitting:
```python
n_train = int(n_rows * 0.9)  # 90% for training
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
```

## 3. Categorical Data Handling

### One-Hot Encoding:
- **Purpose**: Convert categorical variables to numerical format
- **Implementation**: `OneHotEncoder` from scikit-learn
- **Key Parameters**:
  - `handle_unknown='ignore'`: Prevents errors with unseen categories
  - Automatically learns categories during `.fit()`
  - Transforms both training and test data consistently

### Usage:
```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')

# Fit and transform training data
X_train_enc = enc.fit_transform(X_train)

# Transform test data (using same encoder)
X_test_enc = enc.transform(X_test)
```

## 4. Hyperparameter Tuning with GridSearchCV

### Purpose:
- Automate cross-validation and hyperparameter optimization
- Find optimal model configuration without manual trial-and-error

### Implementation:
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
parameters = {'max_depth': [3, 10, None]}

# Initialize base model
decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)

# Configure grid search
grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')

# Train and find best parameters
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)

# Use best model for predictions
decision_tree_best = grid_search.best_estimator_
```

### Key Parameters:
- `n_jobs=-1`: Use all available CPU cores
- `cv=3`: 3-fold cross-validation
- `scoring='roc_auc'`: Optimize for ROC AUC metric

## 5. Random Forest (Ensemble Method)

### Concept:
- Ensemble of multiple decision trees
- Reduces overfitting through bagging and feature randomness
- Generally provides better accuracy than single trees

### Implementation:
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    criterion='gini',      # Splitting criterion
    min_samples_split=30,  # Minimum samples to split
    n_jobs=-1,             # Parallel processing
    random_state=42        # Reproducibility
)

# Use GridSearchCV for hyperparameter tuning
grid_search_rf = GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search_rf.fit(X_train_enc, Y_train)

# Evaluate performance
random_forest_best = grid_search_rf.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print(f'ROC AUC: {roc_auc_score(Y_test, pos_prob):.3f}')
```

### Advantages:
- Better generalization through ensemble averaging
- Handles high-dimensional data effectively
- Provides feature importance estimates
- More robust to noise and outliers

## 6. Gradient Boosting with XGBoost

### Concept:
- Sequential ensemble method where each tree corrects errors of previous ones
- Often provides state-of-the-art performance on structured data

### Implementation:
```python
import xgboost as xgb

# Initialize XGBoost model
model = xgb.XGBClassifier(
    learning_rate=0.1,    # Step size shrinkage
    max_depth=10,         # Maximum tree depth
    n_estimators=1000     # Number of boosting rounds
)

# Train and evaluate
model.fit(X_train_enc, Y_train)
pos_prob = model.predict_proba(X_test_enc)[:, 1]
print(f'ROC AUC: {roc_auc_score(Y_test, pos_prob):.3f}')
```

### Key Parameters:
- `learning_rate`: Controls contribution of each tree
- `max_depth`: Maximum depth of individual trees
- `n_estimators`: Number of boosting iterations

## 7. Performance Evaluation

### Metric Used:
- **ROC AUC**: Area Under the Receiver Operating Characteristic Curve
- Measures model's ability to distinguish between classes
- Range: 0.5 (random) to 1.0 (perfect)

### Comparison Results:
- Single Decision Tree: Baseline performance
- Random Forest: Typically improves upon single tree
- XGBoost: Often achieves best performance

## 8. Best Practices and Considerations

### Data Preparation:
- Always handle categorical variables appropriately (one-hot encoding)
- Split data before preprocessing to avoid data leakage
- Use consistent preprocessing for training and test sets

### Model Selection:
- Start with simple models (single decision tree) for baseline
- Progress to ensemble methods (Random Forest, XGBoost) for better performance
- Use cross-validation for reliable performance estimation

### Hyperparameter Tuning:
- Focus on key parameters: max_depth, min_samples_split, n_estimators
- Use GridSearchCV or RandomizedSearchCV for systematic optimization
- Consider computational resources when defining search space

### Interpretability:
- Decision trees offer high interpretability
- Random Forest provides feature importance
- XGBoost offers feature importance but is less interpretable

## 9. Key Takeaways

1. **Decision Trees** are fundamental ML algorithms that create tree-like models of decisions.
2. **Data Preprocessing** is crucial, especially handling categorical variables through one-hot encoding.
3. **Hyperparameter Tuning** using GridSearchCV helps find optimal model configurations.
4. **Ensemble Methods** like Random Forest and XGBoost typically outperform single decision trees.
5. **ROC AUC** is an effective metric for evaluating binary classification models.
6. **Model Selection** should balance performance, interpretability, and computational requirements.

This summary provides a comprehensive reference for decision tree techniques and their implementation in Python using scikit-learn and related libraries.
