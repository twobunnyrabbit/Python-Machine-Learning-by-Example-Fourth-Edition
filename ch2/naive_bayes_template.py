# -*- coding: utf-8 -*-
"""
# Naive Bayes Classifier Template Notebook

This notebook serves as a template for applying Naive Bayes classification using scikit-learn.
It covers the standard steps:
1. Data Loading and Initial Inspection
2. Data Preprocessing and Train/Test Split
3. Model Selection, Training, and Evaluation
4. Model Tuning (if applicable for the specific Naive Bayes variant)

Author: [Your Name/Pseudonym]
Date: [Current Date]
"""

# %% [markdown]
"""
## 1. Import Libraries

We start by importing all the necessary libraries.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.datasets import load_iris # Example dataset
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore') # Optional: To ignore potential warnings

# %% [markdown]
"""
## 2. Data Loading and Initial Inspection

Here we load the dataset and perform some initial checks to understand its structure, features, and target variable.
For this template, we'll use the famous Iris dataset included in scikit-learn, which is a classic example for classification.

**Replace this section with your actual data loading logic if you are using a different dataset.**
"""

# %%
# Load the dataset
# Replace with your data loading:
# df = pd.read_csv('your_dataset.csv')
# Or load from other sources (databases, APIs, etc.)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Display basic info
print("Features (X) Head:")
print(X.head())
print("\nTarget (y) Head:")
print(y.head())

print("\nDataset Info:")
X.info()

print("\nDescriptive Statistics:")
print(X.describe())

print("\nTarget Distribution:")
print(y.value_counts()) # Check class balance

# Optional: Visualize feature distributions (helps understand if GaussianNB is appropriate)
# X.hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

# Optional: Pairplot to visualize relationships (can be slow on large datasets)
# sns.pairplot(X.join(y), hue='target', palette='viridis')
# plt.show()


# %% [markdown]
"""
## 3. Data Preprocessing and Train/Test Split

This section handles data cleaning, transformation, and splitting the data into training and testing sets.

Typical steps include:
- Handling missing values (imputation or removal)
- Encoding categorical variables (One-Hot Encoding, Label Encoding)
- Feature Scaling (Standardization or Normalization - **Note:** Gaussian Naive Bayes is less sensitive to scaling than distance-based models like SVMs or KNN, but other variants might benefit or require specific input types).
- Creating new features (Feature Engineering)

For the Iris dataset, these steps are minimal: no missing values, no categorical features. The primary step is splitting the data.
"""

# %%
# --- Preprocessing Steps ---

# 1. Handling Missing Values (Example - uncomment if needed)
# print("\nMissing values before handling:")
# print(X.isnull().sum())
# X.fillna(X.mean(), inplace=True) # Example: Impute with mean
# print("\nMissing values after handling:")
# print(X.isnull().sum())

# 2. Encoding Categorical Variables (Example - uncomment if needed)
# if 'categorical_column' in X.columns:
#     X = pd.get_dummies(X, columns=['categorical_column'], drop_first=True)

# 3. Feature Scaling (Optional for GaussianNB, but good practice to consider)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=X.columns)
# print("\nFeatures after scaling (if applied):")
# print(X.head())


# --- Train/Test Split ---
# Split data into training and testing sets
# test_size: proportion of the dataset to include in the test split
# random_state: ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # stratify=y is good for classification to maintain class distribution

print(f"\nShape of training features (X_train): {X_train.shape}")
print(f"Shape of training target (y_train): {y_train.shape}")
print(f"Shape of testing features (X_test): {X_test.shape}")
print(f"Shape of testing target (y_test): {y_test.shape}")


# %% [markdown]
"""
## 4. Model Selection, Training, and Evaluation

Naive Bayes is a family of probabilistic classification algorithms based on Bayes' Theorem with the "naive" assumption of conditional independence between features given the class.

Common variants in scikit-learn:
- `GaussianNB`: Assumes features follow a normal distribution. Suitable for continuous data.
- `MultinomialNB`: Suitable for discrete counts. Often used for text classification (e.g., word counts).
- `BernoulliNB`: Suitable for binary/boolean features.

We'll use `GaussianNB` as the Iris dataset has continuous numerical features.

This section covers:
- Instantiating the chosen Naive Bayes model
- Training the model on the training data
- Evaluating the model's performance on the test data
- Using cross-validation for a more robust performance estimate
"""

# %%
# --- Choose and Instantiate the Model ---

# Select the appropriate Naive Bayes model based on your data characteristics
# For Iris (continuous numerical data), GaussianNB is suitable.
# For text data (word counts), use MultinomialNB.
# For binary features, use BernoulliNB.

model = GaussianNB()
# Other options:
# model = MultinomialNB()
# model = BernoulliNB()


print(f"Selected Model: {type(model).__name__}")

# %%
# --- Training the Model ---

# Train the model using the training data
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# %%
# --- Making Predictions ---

# Predict on the test data
y_pred = model.predict(X_test)

# %%
# --- Model Evaluation ---

print("\n--- Model Evaluation ---")

# 1. Accuracy Score: Proportion of correct predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.4f}")

# 2. Confusion Matrix: Shows the counts of true positives, true negatives, false positives, and false negatives
# Rows represent actual classes, columns represent predicted classes
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Optional: Visualize Confusion Matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()


# 3. Classification Report: Provides precision, recall, f1-score for each class
# Precision: Proportion of positive identifications that were actually correct (TP / (TP + FP))
# Recall (Sensitivity): Proportion of actual positives that were identified correctly (TP / (TP + FN))
# F1-score: Harmonic mean of precision and recall (good for imbalanced classes)
# Support: The number of actual occurrences of the class in the specified dataset (test set)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:")
print(class_report)

# %%
# --- Cross-Validation ---
# Use cross-validation for a more robust estimate of the model's performance.
# It splits the data into 'cv' folds, trains on cv-1 folds, and tests on the remaining fold.
# This process is repeated 'cv' times, and the results are averaged.

print("\n--- Cross-Validation ---")
cv_scores = cross_val_score(model, X, y, cv=5) # Using the whole dataset X, y here for CV
print(f"Cross-validation scores (5 folds): {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation of cross-validation accuracy: {cv_scores.std():.4f}")

# %% [markdown]
"""
## 5. Model Tuning

Naive Bayes models typically have few hyperparameters compared to other algorithms.

- `GaussianNB` has `var_smoothing`: a small value added to the variance for numerical stability. Tuning this might sometimes improve performance.
- `MultinomialNB` and `BernoulliNB` have `alpha`: Laplace/Lidstone smoothing parameter. Adjusting `alpha` can help with unseen features in the test data (adds a small count to all features).

We can use `GridSearchCV` to find the best value for the available hyperparameter(s).
"""

# %%
print("--- Model Tuning (using GridSearchCV) ---")

# Define the parameter grid to search
# For GaussianNB, the main parameter is var_smoothing
param_grid = {'var_smoothing': np.logspace(0, -9, 100)} # Search across a range of values from 1e0 to 1e-9

# For MultinomialNB or BernoulliNB, you might tune 'alpha'
# param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]}

# Create the GridSearchCV object
# estimator: the model to tune
# param_grid: the grid of parameters to search
# cv: number of cross-validation folds to use during tuning
# scoring: metric to optimize (e.g., 'accuracy', 'f1_macro')
grid_search = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid, cv=5, scoring='accuracy') # Use GaussianNB estimator

# Fit GridSearchCV to the training data (or full data X, y depending on strategy)
# It's common practice to tune on the training set obtained *before* the final test split.
print("Running GridSearchCV...")
grid_search.fit(X_train, y_train)
print("GridSearchCV complete.")

# Print the best parameters found
print(f"\nBest parameters found: {grid_search.best_params_}")

# Print the best cross-validation score obtained with these parameters
print(f"Best cross-validation score (using training data): {grid_search.best_score_:.4f}")

# Get the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# %%
# --- Evaluate the best model on the held-out test set ---
print("\n--- Evaluating the best model on the test set ---")

y_pred_tuned = best_model.predict(X_test)

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)
class_report_tuned = classification_report(y_test, y_pred_tuned, target_names=iris.target_names)


print(f"Accuracy on the test set (tuned model): {accuracy_tuned:.4f}")
print("\nConfusion Matrix (tuned model):")
print(conf_matrix_tuned)
print("\nClassification Report (tuned model):")
print(class_report_tuned)

# Compare with the initial model's performance on the test set
print(f"\nInitial Model Test Accuracy: {accuracy:.4f}")
print(f"Tuned Model Test Accuracy:   {accuracy_tuned:.4f}")
# Note: For simple datasets like Iris and models like Naive Bayes, tuning might not yield significant improvements,
# or the improvement might be slightly different between the CV score and the final test score due to data split randomness.


# %% [markdown]
"""
## 6. Conclusion and Further Steps

- We have successfully loaded data, preprocessed it, trained a Naive Bayes classifier, evaluated its performance, and performed hyperparameter tuning.
- Naive Bayes is a simple yet effective baseline model, known for its speed and interpretability (due to its probabilistic nature, although the independence assumption is a simplification).
- Its performance can degrade if features are highly correlated, violating the independence assumption.

**Further Steps:**
- Try other Naive Bayes variants (`MultinomialNB`, `BernoulliNB`) if your data characteristics suggest it.
- Explore other classification algorithms (Logistic Regression, SVM, Decision Trees, Random Forests, Gradient Boosting, etc.).
- Perform more in-depth Exploratory Data Analysis (EDA).
- Implement more sophisticated preprocessing or feature engineering techniques.
- Handle imbalanced datasets if applicable (e.g., using techniques like SMOTE).
- Deploy the trained model (e.g., using libraries like joblib or pickle to save the model).
"""

# %%
# Example: Saving the trained model
# import joblib
# filename = 'naive_bayes_model.pkl'
# joblib.dump(best_model, filename)
# print(f"\nModel saved as {filename}")

# Example: Loading the model later
# loaded_model = joblib.load(filename)
# print(f"Loaded model: {loaded_model}")
# loaded_model.predict(X_test) # Use the loaded model

# %%