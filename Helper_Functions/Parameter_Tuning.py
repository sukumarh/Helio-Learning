import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.svm import SVR


def Hyper_parameter_tuning(X, y, model_estimator, configs):
    n_train = len(X)
    validation_fold_ = [-1] * ((int)(0.8 * n_train)) + [0] * ((int)(0.2 * n_train))
    validation_fold = np.random.permutation(validation_fold_)

    # Using GridSearchCV to tune the hyper-parameters
    ps = PredefinedSplit(validation_fold)
    clf = GridSearchCV(model_estimator,
                       configs,
                       return_train_score=True,
                       cv=ps,
                       refit=True,
                       n_jobs=-1,
                       scoring=make_scorer(mean_squared_error, greater_is_better=False))
    clf.fit(X, y)
    return clf

