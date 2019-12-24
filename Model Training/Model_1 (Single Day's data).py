import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.svm import SVR
from Helper_Functions import File_Ops as File_IO
from Helper_Functions import Dataset_Ops as Data
from Helper_Functions import Parameter_Tuning as Tuning
from Helper_Functions import Plotting_Ops as Plotter


def non_linear_reg_using_SVR(X, y, configs):

    # Fitting the SVR regression model
    # svr_rbf = SVR(kernel='rbf', C=1000, gamma=8, epsilon=0.001)
    # svr_rbf_2 = SVR(kernel='rbf', C=100, gamma=4, epsilon=0.001, coef0=1)
    # svr_poly = SVR(kernel='poly', C=100, gamma=5, degree=5, epsilon=.01,
    #                coef0=1)
    # svr_sig = SVR(kernel='sigmoid', C=10, gamma='auto', epsilon=.01,
    #                coef0=1)

    # svrs = [svr_rbf, svr_lin, svr_poly]
    # kernel_label = ['RBF', 'Linear', 'Polynomial']
    # model_color = ['m', 'c', 'g']

    # svrs = [svr_rbf, svr_rbf_2, svr_poly]
    # kernel_label = ['RBF', 'RBF 2', 'Polynomial']
    # model_color = ['m', 'g', 'c']

    # svr_estimator = SVR(coef0=1, gamma='auto')
    svr_estimator = SVR(coef0=1)

    clf = Tuning.Hyper_parameter_tuning(X, y, svr_estimator, configs)

    print('Hyper-parameter tuning completed')

    kernel_label = clf.best_params_['kernel']

    print(f'Best parameters are: {clf.best_params_}')
    print(f'Best score is: {clf.best_score_}')

    svr = SVR(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'],
              gamma=clf.best_params_['gamma'],
              epsilon=clf.best_params_['epsilon'],
              degree=clf.best_params_['degree'], coef0=1)

    y_hat = svr.fit(X, y).predict(X)

    Plotter.plot_svr(X, y_hat, y, kernel_label, 'm', 'r', svr)

    return clf, svr

    # Plotter.plot_svr_multi(X, y, kernel_label, model_color, svrs,
    # n_rows=1, n_cols=3, fig_size=(15, 5))


def non_linear_reg_using_SVR_dataset_splitted(X, y, configs):

    X_pos, X_neg, y_pos, y_neg = Data.split_pos_and_neg_set(X, y)

    svr_estimator = SVR(coef0=1, gamma='auto')
    clf_pos = Tuning.Hyper_parameter_tuning(X_pos, y_pos, svr_estimator, configs)
    clf_neg = Tuning.Hyper_parameter_tuning(X_neg, y_neg, svr_estimator, configs)

    best_pos_params = clf_pos.best_params_
    best_neg_params = clf_neg.best_params_

    print(f'Best parameters for positives are: {best_pos_params}')
    print(f'Best score for positives are: {clf_pos.best_score_}')
    print(f'Best parameters for negatives are: {best_neg_params}')
    print(f'Best score for negatives are: {clf_neg.best_score_}')

    # best_params = {'C': 1000, 'degree': 9, 'epsilon': 1, 'gamma': 20, 'kernel': 'poly'}

    kernel_label = [best_neg_params['kernel'], best_pos_params['kernel']]
    model_color = ['m', 'g']

    svr_pos_best = SVR(kernel=best_pos_params['kernel'], C=best_pos_params['C'],
                       # gamma=best_pos_params['gamma'],
                       epsilon=best_pos_params['epsilon'],
                       degree=best_pos_params['degree'], coef0=1)
    svr_neg_best = SVR(kernel=best_neg_params['kernel'], C=best_neg_params['C'],
                       # gamma=best_neg_params['gamma'],
                       epsilon=best_neg_params['epsilon'],
                       degree=best_neg_params['degree'], coef0=1)

    # plot_svr(X, y, kernel_label, model_color, [svr_pos_best, svr_neg_best])

    Plotter.plot_svr_combined([X_neg, X_pos], [y_neg, y_pos],
                      kernel_label, model_color,
                      [svr_neg_best, svr_pos_best])

    return clf_neg, clf_pos


def main():

    # Read data file
    print("Reading the data file: Trial - AMIE")

    start_time = dt.datetime(2003, 1, 1, 0, 0, 0)  # Start Time
    stop_time = dt.datetime(2003, 1, 2, 0, 0, 0)  # Stop Time

    file_name = 'Data/SingleDay/b20031001n.save'

    # fac_train, hal_train = read_file(16, 0, start_time, stop_time, file_name, mode='Hall')

    fac_train, hal_train = File_IO.read_single_file(16, 0,
                                                    file_name,
                                                    mode='Hall')

    print('Data read complete.')

    # Hyper-parameter tuning configurations
    # SVR_configs = [{'kernel': 'rbf',
    #                 'gamma': [20, 10, 8, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    #                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #                 'epsilon': [0.001, 0.01, 0.1, 1, 10]},
    #                {'kernel': 'sigmoid',
    #                 'gamma': [20, 10, 8, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    #                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #                 'epsilon': [0.001, 0.01, 0.1, 1, 10]},
    #                {'kernel': 'poly',
    #                 'gamma': [20, 10, 8, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    #                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #                 'epsilon': [0.001, 0.01, 0.1, 1, 10],
    #                 'degree': [2, 3, 4, 5, 6, 7, 8, 9]}]

    # SVR_configs_2 = [{'kernel': ['rbf', 'poly'],
    #                   'gamma': [20, 10, 8, 1,  1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    #                   'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                   'epsilon': [0.001, 0.01, 0.1, 1],
    #                   'degree': [1, 2, 3, 4, 5, 6, 7, 8]}]

    # SVR_configs_2 = [{'kernel': ['rbf', 'poly'],
    #                   'gamma': [20, 10, 8, 1, 1e-1, 1e-2],
    #                   'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                   'epsilon': [0.001, 0.01, 0.1, 1],
    #                   'degree': [1, 2, 3, 4, 5, 6, 7, 8]}]

    # SVR_configs_2 = [{'kernel': ['rbf', 'poly'],
    #                   'gamma': [20, 10],
    #                   'C': [10, 100],
    #                   'epsilon': [0.1, 1],
    #                   'degree': [1,2,3,4,5]}]

    SVR_configs_2 = [{'kernel': ['poly', 'rbf'],
                      'gamma': [20, 10, 0.1, 1e-2, 1e-3, 1e-4, 'auto'],
                      'C': [10, 1, 0.1, 0.01],
                      'epsilon': [0.1, 0.01, 0.001],
                      'degree': [1, 2, 3, 4, 5]}]

    clf, svr = non_linear_reg_using_SVR(fac_train, hal_train, SVR_configs_2)

    File_IO.save_model('Trained_Models/svr_single_day.p', [clf, svr])

    # clf_neg, clf_pos = non_linear_reg_using_SVR_dataset_splitted(fac_train, hal_train,
    #                                                              SVR_configs_2)


if __name__ == '__main__':
    main()

