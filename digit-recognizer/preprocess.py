import numpy as np
import pandas as pd
from sklearn import model_selection


def load_train_data():
    print("Begin loading train data")
    # read the train and test data
    train_data = pd.read_csv('data/train.csv', nrows=5000)


    # split the train data into X_train and y_train
    X = train_data.drop(columns=['label']).values
    y = train_data['label'].values

    # split the data into training and testing data
    X_train, X_validate, y_train, y_validate = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    print("X_train shape: ", X_train.shape)
    print("X_validate shape: ", X_validate.shape)
    print("y_train shape: ", y_train.shape)
    print("y_validate shape: ", y_validate.shape)

    return X_train, y_train, X_validate, y_validate

def load_test_data():
    print("Begin loading test data")
    test_data = pd.read_csv('data/test.csv', nrows=5000)
    X_test = test_data.values
    print("X_test shape: ", X_test.shape)
    return X_test

