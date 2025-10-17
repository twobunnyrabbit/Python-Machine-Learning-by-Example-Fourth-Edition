import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    return os, pd


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
def _():
    return


if __name__ == "__main__":
    app.run()
