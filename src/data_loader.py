# Author: GitHub Copilot

import pandas as pd

def load_data():
    train = pd.read_csv("data/santander_train.csv")
    test = pd.read_csv("data/santander_test.csv")
    return train, test