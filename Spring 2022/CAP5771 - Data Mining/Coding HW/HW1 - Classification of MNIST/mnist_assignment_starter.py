from sklearn.model_selection import ShuffleSplit, cross_validate, KFold
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from typing import Dict


def prepare_data(num_train=60000, num_test=10000, normalize=True):

    X, y = datasets.fetch_openml(
        "mnist_784",
        version = 1,
        return_X_y = True,
        as_frame = False
    )

    if normalize:
        X = X / X.max()

    y = y.astype(int)
    Xtrain, Xtest = X[:num_train], X[num_train:num_train + num_test]
    ytrain, ytest = y[:num_train], y[num_train:num_train + num_test]
    return Xtrain, ytrain, Xtest, ytest


def filter_out_7_9s(X, y):
    seven_nine_idx = (y == 7) | (y == 9)
    X_binary = X[seven_nine_idx, :]
    y_binary = y[seven_nine_idx]
    return X_binary, y_binary


def train_simple_classifier_with_cv(Xtrain,
                                    ytrain,
                                    clf,
                                    n_splits = 5,
                                    cv_class = KFold):

    cv = cv_class(n_splits=n_splits)
    scores = cross_validate(clf, Xtrain, ytrain, cv=cv)
    return scores


def print_cv_result_dict(cv_dict: Dict):
    for (key, array) in cv_dict.items():
        print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")


if __name__ == "__main__":
    Xtrain, ytrain, Xtest, ytest = prepare_data()
    Xtrain, ytrain = filter_out_7_9s(Xtrain, ytrain)
    Xtest, ytest = filter_out_7_9s(Xtest, ytest)

    out_dict = train_simple_classifier_with_cv(
        Xtrain,
        ytrain,
        DecisionTreeClassifier()
    )
    print("running cross validation...")
    print_cv_result_dict(out_dict)
