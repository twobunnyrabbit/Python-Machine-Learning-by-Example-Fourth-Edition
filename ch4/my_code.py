import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, os, pd, plt


@app.cell
def _(os):
    os.getcwd()
    return


@app.cell
def _():
    from sklearn.feature_extraction import DictVectorizer
    return (DictVectorizer,)


@app.cell
def _():
    X_dict = [{'interest': 'tech', 'occupation': 'professional'},
    {'interest': 'fashion', 'occupation': 'student'},
    {'interest': 'fashion', 'occupation': 'professional'},
    {'interest': 'sports', 'occupation': 'student'},
    {'interest': 'tech', 'occupation': 'student'},
    {'interest': 'tech', 'occupation': 'retired'},
    {'interest': 'sports', 'occupation': 'professional' }]
    return (X_dict,)


@app.cell
def _(X_dict, pd):
    pd.DataFrame(X_dict)
    return


@app.cell
def _(DictVectorizer):
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    return (dict_one_hot_encoder,)


@app.cell
def _(X_dict, dict_one_hot_encoder):
    X_encoded = dict_one_hot_encoder.fit_transform(X_dict)
    return (X_encoded,)


@app.cell
def _(X_encoded):
    X_encoded
    return


@app.cell
def _(mo):
    mo.md(r"""Getting started with the logistic function""")
    return


@app.cell
def _(np):
    def sigmoid(input):
        return 1.0 / (1 + np.exp(-input))
    return (sigmoid,)


@app.cell
def _(np, sigmoid):
    def _():
        z = np.linspace(-8, 8, 1000)
        y = sigmoid(z)
        return z, y
    

    z, y = _()
    return y, z


@app.cell
def _(pd, y, z):
    import altair as alt

    # Create DataFrame for plotting
    sigmoid_df = pd.DataFrame({'z': z, 'sigmoid(z)': y})

    # Create Altair line chart
    chart = alt.Chart(sigmoid_df).mark_line(
        color='steelblue',
        strokeWidth=3
    ).encode(
        x=alt.X('z', title='Input (z)', scale=alt.Scale(domain=[-8, 8])),
        y=alt.Y('sigmoid(z)', title='Sigmoid Output', scale=alt.Scale(domain=[0, 1]))
    ).properties(
        title='Sigmoid Function Plot',
        width=600,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16
    )
    # Create horizontal rule at y=0.5
    horizontal_line = alt.Chart(sigmoid_df).mark_rule(
        strokeDash=[5, 5],  # Dashed line pattern
        color='red',
        strokeWidth=2
    ).encode(
        y='y:Q'
    )
    chart
    return


@app.cell
def _(mo):
    mo.md(r"""Define function that computes the prediction $\hat{y}(x)$ with the current weights.""")
    return


@app.cell
def _(np, sigmoid):
    def compute_prediction(X, weights):
        """
        Compute the prediction y_hat based on the current weights
        """
        z = np.dot(X, weights)
        return sigmoid(z)
    return (compute_prediction,)


@app.cell
def _(compute_prediction, np):
    def update_weights_gd(X_train, y_train, weights, learning_rate):
        """
        Update weights by one step
        """
        predictions = compute_prediction(X_train, weights)
        weights_delta = np.dot(X_train.T, y_train - predictions)
        m = y_train.shape[0]
        weights += learning_rate / float(m) * weights_delta
        return weights
    return (update_weights_gd,)


@app.cell
def _(compute_prediction, np):
    def compute_cost(X, y, weights):
        """
         Compute the cost J(w)
        """
        predictions = compute_prediction(X, weights)
        cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
        return cost

    return (compute_cost,)


@app.cell
def _(compute_cost, np, update_weights_gd):
    def train_logistic_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
        """ Train a logistic regression model
        Args:
            X_train, y_train (numpy.ndarray, training data set)
            max_iter (int, number of iterations)
            learning_rate (float)
            fit_intercept (bool, with an intercept w0 or not)
        Returns:
            numpy.ndarray, learned weights
        """
        if fit_intercept:
            intercept = np.ones((X_train.shape[0], 1))
            X_train = np.hstack((intercept, X_train))
        weights = np.zeros(X_train.shape[1])
        for iteration in range(max_iter):
            weights = update_weights_gd(X_train, y_train, weights, learning_rate)
            # Check the cost for every 100 (for example) iterations
            if iteration % 100 == 0:
                print(compute_cost(X_train, y_train, weights))
        return weights
    return (train_logistic_regression,)


@app.cell
def _(compute_prediction, np):
    def predict(X, weights):
        if X.shape[1] == weights.shape[0] - 1:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        return compute_prediction(X, weights)
    return (predict,)


@app.cell
def _(np):
    # A example
    X_train = np.array([[6, 7],
                        [2, 4],
                        [3, 6],
                        [4, 7],
                        [1, 6],
                        [5, 2],
                        [2, 0],
                        [6, 3],
                        [4, 1],
                        [7, 2]])

    y_train = np.array([0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1])
    return X_train, y_train


@app.cell
def _(X_train, train_logistic_regression, y_train):
    weights = train_logistic_regression(X_train, y_train, max_iter=1000, learning_rate=0.1, fit_intercept=True)
    return (weights,)


@app.cell
def _(np, predict, weights):
    X_test = np.array([[6, 1],
                       [1, 3],
                       [3, 1],
                       [4, 5]])

    predictions = predict(X_test, weights)
    print(predictions)
    return X_test, predictions


@app.cell
def _(X_test, X_train, plt, predictions):
    plt.figure()  # Create a new figure
    plt.scatter(X_train[:5,0], X_train[:5,1], c='b', marker='x')
    plt.scatter(X_train[5:,0], X_train[5:,1], c='k', marker='.')
    for i, prediction in enumerate(predictions):
        marker = 'X' if prediction < 0.5 else 'o'
        c = 'b' if prediction < 0.5 else 'k'
        plt.scatter(X_test[i,0], X_test[i,1], c=c, marker=marker)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Predicting ad click-through with logistic regression using gradient descent""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
