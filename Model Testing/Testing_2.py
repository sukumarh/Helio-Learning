import datetime as dt
from Helper_Functions import File_Ops as File_IO
from Helper_Functions import Plotting_Ops as Plotter
from Helper_Functions import Dataset_Ops as Data


def main():
    print('Model Testing')

    start_time_train = dt.datetime(2003, 1, 1, 0, 0, 0)
    stop_time_train = dt.datetime(2003, 1, 10, 0, 0, 0)

    directory = 'Processed_Data/2004/01_01.p'

    test_dataset, is_success = File_IO.read_data(directory)

    file_name = 'Trained_Models/svr_multi_feature_1.p'
    data, is_read_successful = File_IO.read_model(file_name)

    if is_read_successful:

        print('Model retrieved')

        if data[0] == 'single_svr':
            y_test = test_dataset[:, 0].T[: 20000]
            X_test = test_dataset[:, 2:][: 20000]
            svr_model = data[1]
            # y_hat = svr.fit(X_train, y_train).predict(X_train)
            # Plotter.plot_svr(X_train, y_hat, y_train, svr, svr.kernel, 'm', 'r',
            #                  [min(X_train) - 0.1, max(X_train) + 0.1, min(y_train) - 1, max(y_train) + 1])

            y_hat = svr_model.predict(X_test)
            test_score = svr_model.score(X_test, y_test)
            print(f'Test Score = {test_score}')

            # Plotter.plot_svr(X_test, y_hat, y_test, svr, svr.kernel, 'm', 'b', mode='test')

        # elif data[0] == 'Non_Linear_Reg_SVR_Tuned_Splitted_Dataset':
        #     [clf_neg, clf_pos, svr_pos, svr_neg] = data[1]
        #     X_test_pos, X_test_neg, y_test_pos, y_test_neg = Data.split_pos_and_neg_set(X_test, y_test)
        #
        #     y_hat_neg = svr_neg.predict(X_test_neg)
        #     y_hat_pos = svr_pos.predict(X_test_pos)
        #
        #     test_score_neg = svr_neg.score(X_test_neg, y_test_neg)
        #     test_score_pos = svr_pos.score(X_test_pos, y_test_pos)
        #
        #     print(f'Positive Test Score = {test_score_pos}')
        #     print(f'Negative Test Score = {test_score_neg}')
        #
        #     Plotter.plot_svr_combined([X_test_neg, X_test_pos], [y_hat_neg, y_hat_pos], [y_test_neg, y_test_pos],
        #                               [svr_neg.kernel, svr_pos.kernel], ['m', 'g'], ['b', 'b'],
        #                               [svr_neg, svr_pos],
        #                               mode='test')


if __name__ == '__main__':
    main()
