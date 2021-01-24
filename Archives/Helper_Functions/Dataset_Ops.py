import numpy as np


def split_pos_and_neg_set(X, y):
    positive_indices = [i for i, val in enumerate(X >= 0) if val]
    negative_indices = [i for i, val in enumerate(X < 0) if val]

    X_pos = np.array([X[i] for i in positive_indices])
    y_pos = np.array([y[i] for i in positive_indices])

    X_neg = np.array([X[i] for i in negative_indices])
    y_neg = np.array([y[i] for i in negative_indices])

    return X_pos, X_neg, y_pos, y_neg