## Implementation of local weighted linear regression
## In this implement, more weight will give to the data closer to the point at which we want to predict
## the weight is given by a Gaussian kernal function
##  Kh(z) = 1/(h*sqrt(2*pi))^p * exp(-(||z||^2/(2*h^2))),
## where z \in R^p, and h is the bandwidth need to be tuned.

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, ParameterGrid
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

import time
import matplotlib.pyplot as plt
import warnings
class LocalWeightedLinearRegression:

    def __init__(self, h=1, fit_intercept=True, copy_X=True):
        ## X, m x p matrix; m samples, p features
        ## y, 1 x m array
        ## Xtest, k x p array.
        ## ytest, 1 x k array

        self.h = h
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.parameters = {'fit_intercept': fit_intercept, 'copy_X': copy_X}

    def _Kh(self, Xc, h, axis=1):
        m, p = Xc.shape
        ss = np.sum(Xc ** 2, axis=axis)
        return np.exp(-ss / (2 * (h ** 2))) / ((np.sqrt(2 * np.pi) * h) ** p)

    def fit(self, X, y):

        if self.copy_X:
            self.X = X.copy()
        else:
            self.X = X
        self.y = y.reshape(-1, 1)  ## reshape to a column vector

        return self

    def predict(self, Xtest):
        m, p = self.X.shape
        n_test = Xtest.shape[0]

        self.predictions_ = np.zeros(n_test)
        if self.fit_intercept:
            self.coefs_ = np.zeros((n_test, p + 1))
        else:
            self.coefs_ = np.zeros((n_test, p))

        for i in range(n_test):
            xtest = Xtest[i, :]
            Xc = np.tile(xtest, (m, 1)) - self.X

            if self.fit_intercept:
                XX = np.hstack((np.ones((m, 1)), Xc))
            else:
                XX = Xc

            W = np.diag(self._Kh(Xc, h=self.h, axis=1))
            coef_ = np.linalg.pinv(XX.T @ W @ XX) @ XX.T @ W @ self.y
            self.coefs_[i, :] = coef_.flatten()
            if self.fit_intercept:
                self.predictions_[i] = self.coefs_[i, 0]  # np.dot(np.insert(xtest,0,1),  self.coefs_[i, :])
            else:
                self.predictions_[i] = np.dot(xtest, self.coefs_[i, :])

        return self

    def score(self, ytest):
        self.ytest = ytest.flatten()
        self.errors_ = (self.ytest - self.predictions_) ** 2
        self.mse = np.mean(self.errors_)
        self.sse = np.sum(self.errors_)

        return self


if __name__ == "__main__":

    data = loadmat("data/LWLR_data.mat")
    X = data['data'][:, 0].reshape(-1, 1)
    y = data['data'][:, 1]
    X.shape

    ## cross validation to find the best h
    n_cv = 5
    kf = KFold(n_splits = n_cv, shuffle = True, random_state = 0)
    num_h = 50
    hs = np.logspace(-4, 4, num_h)
    cv_mse = np.zeros((n_cv, num_h))
    cv_std = np.zeros(num_h)
    cv_sse = np.zeros((n_cv, num_h))

    for i, h in enumerate(hs):
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            Xtrain = X[train_idx, :]
            ytrain = y[train_idx]
            Xtest = X[test_idx, :]
            ytest = y[test_idx]

            LWLR = LocalWeightedLinearRegression(h=h)
            fit = LWLR.fit(Xtrain, ytrain)
            fit.predict(Xtest)
            fit.score(ytest)
            cv_mse[fold, i] = fit.mse
            cv_sse[fold, i] = fit.sse

    cv_mse_std = np.std(cv_mse, axis=0)
    cv_mse_mean = np.mean(cv_mse, axis=0)
    cv_sse_mean = np.mean(cv_sse, axis=0)
    best_h = round(hs[np.argmin(cv_mse_mean)], 4)
    best_mse = round(cv_mse_mean.min(),4)

    print(f'best mse: \t{cv_mse_mean.min()}')
    print(f'best sse: \t{cv_sse_mean.min()}')
    print(f'best h:\t {best_h}')

    # plot MSE curve
    plt.figure(figsize=(5, 2.5))
    plt.errorbar(hs, cv_mse_mean, yerr=cv_mse_std)
    plt.axvline(best_h, c='red', linestyle='--')
    plt.text(best_h * 1.1, 0.15, f'best h = {best_h}', va='bottom', fontsize=8, c='red')
    plt.text(best_h * 1.1, 0.15, f'mse = {best_mse}', va='top', fontsize=8, c='red')
    # plt.text(best_lamb * 1.5, 250, f'alpha: {round(best_lamb, 4)}\nmse:   {round(best_score,1)}', c = 'red')
    plt.xscale('log')
    plt.xlabel('h')
    plt.ylabel('mean square error')
    plt.title('LWLR cross-validation, MSE vs. h')
    plt.tight_layout()
    plt.show()


    ## plot prediction curve, and predict datapoint a = -1.2
    h = best_h
    a = np.array(-1.2).reshape(1,1)

    LWLR =  LocalWeightedLinearRegression(h = best_h)
    fit1 = LWLR.fit(X, y).predict(X)
    LWLR2 = LocalWeightedLinearRegression(h = best_h)
    fit2 = LWLR2.fit(X, y).predict(a)

    plt.figure(figsize = (5.5, 4.5))
    ## plot local fitting curve.
    plt.scatter(X, y, marker = 'o', s = 10, label = 'training data')
    plt.plot(X, fit1.predictions_, c = 'orange', label = 'prediction curve')
    plt.scatter(a, fit2.predictions_, marker = 'x', c = 'red', s = 50, label = 'prediction for X = -1.2')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Local weighted linear regression \nprediction curve \n(h = {round(h,4)})')
    plt.legend()
    plt.show()