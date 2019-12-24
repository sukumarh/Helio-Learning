import numpy as np
import matplotlib.pyplot as plt


def plot_svr(X, y_hat, y, svr, kernel_label,
             model_color, dataset_color,
             mode='train'):

    plt.plot(X, y_hat,
             color=model_color,
             label=f'{kernel_label} model')

    if mode == 'train':
        plt.scatter(X[svr.support_], y[svr.support_], facecolor="none",
                    edgecolor=dataset_color, s=50,
                    label=f'{kernel_label} model')
        plt.scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                    y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                    facecolor="none", edgecolor="k", s=50,
                    label='other training data')

    elif mode == 'test':
        plt.scatter(X, y, facecolor="none",
                    edgecolor=dataset_color, s=50,
                    label=f'Data')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=1, fancybox=True, shadow=True)
    plt.axis([min(X) - 0.1, max(X) + 0.1, min(y) - 1, max(y) + 1], 'equal')
    plt.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


def plot_svr_combined(X, y_hat, y, kernel_label, model_color, dataset_color, svrs,
                      n_rows=1, n_cols=2, fig_size=(10, 5),
                      mode='train'):
    lw = 2
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size, sharey=True)

    for ix, svr in enumerate(svrs):
        axes[ix].plot(X[ix], y_hat[ix], color=model_color[ix], lw=lw,
                      label=f'{kernel_label[ix]} model')

        if mode == 'train':
            axes[ix].scatter(X[ix][svr.support_], y[ix][svr.support_], facecolor="none",
                             edgecolor=dataset_color[ix], s=50,
                             label=f'{kernel_label[ix]} support vectors')
            axes[ix].scatter(X[ix][np.setdiff1d(np.arange(len(X[ix])), svr.support_)],
                             y[ix][np.setdiff1d(np.arange(len(X[ix])), svr.support_)],
                             facecolor="none", edgecolor="k", s=50,
                             label='other training data')

        elif mode == 'test':
            axes[ix].scatter(X[ix], y[ix], facecolor="none",
                             edgecolor=dataset_color[ix], s=50,
                             label=f'Data')

        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

        if ix == 0:
            axes[ix].axis([min(X[ix]) - 0.1, max(X[ix]), min(y[ix]) - 1, max(y[ix]) + 1], 'equal')
        else:
            axes[ix].axis([min(X[ix]), max(X[ix]) + 0.1, min(y[ix]) - 1, max(y[ix]) + 1], 'equal')

    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


def plot_multiple_svr(X, y, kernel_label, model_color, svrs, n_rows=1, n_cols=2, fig_size=(10, 5)):
    lw = 2
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size, sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                      label=f'{kernel_label[ix]} model')
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label=f'{kernel_label[ix]} support vectors')
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)
    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()

