import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.svm import SVR
from Helper_Functions import File_Ops as File_IO
from Helper_Functions import Dataset_Ops as Data
from Helper_Functions import Parameter_Tuning as Tuning
from Helper_Functions import Plotting_Ops as Plotter


def Non_Linear_Reg_SVR_Tuned(X, y, configs):

    print('Tuning Hyper-parameters.')
    svr_estimator = SVR(coef0=1)

    clf = Tuning.Hyper_parameter_tuning(X, y, svr_estimator, configs)

    print('Hyper-parameter tuning completed')
    print(f'Best parameters are: {clf.best_params_}')
    print(f'Best score is: {clf.best_score_}')

    kernel_label = clf.best_params_['kernel']

    svr = SVR(kernel=clf.best_params_['kernel'],
              C=clf.best_params_['C'],
              gamma=clf.best_params_['gamma'],
              epsilon=clf.best_params_['epsilon'],
              degree=clf.best_params_['degree'], coef0=1)

    y_hat = svr.fit(X, y).predict(X)

    Plotter.plot_svr(X, y_hat, y, svr, kernel_label, 'm', 'r', mode='train')

    return clf, svr


def Non_Linear_Reg_SVR_Tuned_Splitted_Dataset(X, y, configs):

    X_pos, X_neg, y_pos, y_neg = Data.split_pos_and_neg_set(X, y)

    svr_estimator = SVR(coef0=1)
    clf_pos = Tuning.Hyper_parameter_tuning(X_pos, y_pos, svr_estimator, configs)
    clf_neg = Tuning.Hyper_parameter_tuning(X_neg, y_neg, svr_estimator, configs)

    best_pos_params = clf_pos.best_params_
    best_neg_params = clf_neg.best_params_

    print(f'Best parameters for positives are: {best_pos_params}')
    print(f'Best score for positives are: {clf_pos.best_score_}')
    print(f'Best parameters for negatives are: {best_neg_params}')
    print(f'Best score for negatives are: {clf_neg.best_score_}')

    kernel_label = [best_neg_params['kernel'], best_pos_params['kernel']]
    model_color = ['m', 'g']
    dataset_color = ['b', 'b']

    svr_pos_best = SVR(kernel=best_pos_params['kernel'],
                       C=best_pos_params['C'],
                       gamma=best_pos_params['gamma'],
                       epsilon=best_pos_params['epsilon'],
                       degree=best_pos_params['degree'], coef0=1)
    svr_neg_best = SVR(kernel=best_neg_params['kernel'],
                       C=best_neg_params['C'],
                       gamma=best_neg_params['gamma'],
                       epsilon=best_neg_params['epsilon'],
                       degree=best_neg_params['degree'], coef0=1)

    y_hat_pos = svr_pos_best.fit(X_pos, y_pos).predict(X_pos)
    y_hat_neg = svr_neg_best.fit(X_neg, y_neg).predict(X_neg)

    Plotter.plot_svr_combined([X_neg, X_pos], [y_hat_neg, y_hat_pos], [y_neg, y_pos],
                              kernel_label, model_color, dataset_color,
                              [svr_neg_best, svr_pos_best],
                              mode='train')

    return clf_neg, clf_pos, svr_pos_best, svr_neg_best


def single_svr(X, y):
    svr = SVR(kernel='rbf', C=6, gamma=12, epsilon=.1, degree=5)
    svr_model = svr.fit(X, y)
    y_hat = svr_model.predict(X)
    Plotter.plot_svr(X[:, 0], y_hat, y, svr, 'SVR', 'm', 'r', mode='train')
    File_IO.save_data('Trained_Models/svr_multi_feature_1.p', ['single_svr', svr_model])
    print('Training complete')


def main():

    # Read data file
    print("Reading the data files - 01/2003")

    start_time = dt.datetime(2003, 1, 1, 0, 0, 0)
    stop_time = dt.datetime(2003, 1, 2, 0, 0, 0)

    directory = 'Data/'
    filename = 'Processed_Data/2003/01_01.p'

    dataset, is_success = File_IO.read_data(filename)

    y_train = dataset[:, 0].T
    X_train = dataset[:, 2:]

    single_svr(X_train[: 20000], y_train[: 20000])

    
if __name__ == '__main__':
    main()

