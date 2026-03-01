# Author: GitHub Copilot

def prepare_data(train, test):
    X = train.drop('target', axis=1)
    y = train['target']
    X_test = test
    return X, y, X_test

def remove_id(X):
    return X.drop('ID_code', axis=1)