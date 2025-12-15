## this script implements the expectation-maximization classification

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt

class EM_classifier():
    def __init__(self,
                 n_clusters = 2,
                 reduce_dim = True,
                 ndim = 4,
                 tol = 1e-6,
                 max_iteration = 1000,
                 random_seed = None):
        self.n_clusters = n_clusters
        self.reduce_dim = reduce_dim
        self.ndim = ndim
        self.max_iteration = max_iteration
        self.tol = tol
        self.seed = np.random.randint(1, 2*23-1, 1) if random_seed is None else random_seed
        self.labels = None
        self.centers = None

    def fit(self, data):
        '''
        args:
            data, m x n array. row as samples, n as features
        '''
        if hasattr(self, "seed"):
            np.random.seed(self.seed)
        else:
            np.random.seed()

        m,n = data.shape

        ## dimension reduction
        ndim = min(n, self.ndim) if self.reduce_dim else n
        pca = PCA(n_components=ndim).fit(data)
        pdata = pca.transform(data)

        ## initialize pi, mu, and sigma
        pi = np.random.random(self.n_clusters)
        pi = pi / np.sum(pi)
        mu = np.random.randn(self.n_clusters, self.ndim)
        sigma = [np.random.randn(self.ndim, self.ndim) for _ in range(self.n_clusters)]
        sigma = [s @ s.T + np.eye(self.ndim) for s in sigma]
        tau = np.full((m, self.n_clusters), fill_value=0.0)

        llh = []
        for iter in range(self.max_iteration):
            ## E-step
            for k in range(self.n_clusters):
                tau[:, k] = pi[k] * mvn.pdf(pdata, mu[k], sigma[k])

            sumk_tau = np.sum(tau, axis=1).reshape(m, 1)
            tau = np.divide(tau, np.tile(sumk_tau, (1, self.n_clusters)))

            ## calculate log-likelihood function
            llh.append(np.sum(np.log(sumk_tau)))

            ## M-step
            mu_old = mu.copy()
            for k in range(self.n_clusters):
                pi[k] = np.mean(tau[:, k], axis=0)
                mu[k] = tau[:, k].T @ pdata / np.sum(tau[:, k], axis=0)
                cdata = pdata - np.tile(mu[k].reshape(1, self.ndim), (m, 1)) ## centered pdata
                sigma[k] = cdata.T @ np.diag(tau[:, k]) @ cdata / np.sum(tau[:, k], axis=0)

            if np.linalg.norm(mu - mu_old) < self.tol:
                # print(f'training converged: {iter} iterations')
                self.labels = np.argmax(tau, axis=1)
                self.centers = mu
                self.sigma = sigma
                self.loglikehood = llh
                self.iterations = iter
                self.reconstructed_centers = pca.inverse_transform(mu)
                break

            if iter >= self.max_iteration:
                raise Exception('maximum iteration reached; EM algorithms did not converge.')


    def score(self, true_labels):
        self.confusion_matrix = cm(true_labels, self.labels)


if __name__ == "__main__":

    ## rows are features, columns are samples
    data = np.loadtxt('data/EM_data.dat').T
    true_labels = loadmat('data/EM_label.mat')['trueLabel'].flatten()


    em_clf = EM_classifier(n_clusters=2, ndim=2, random_seed=6740)
    em_clf.fit(data)

    ## the categories of EM may not be in the same order of original labels
    ## plot the image to check categories.

    fig, ax = plt.subplots(1, em_clf.n_clusters, figsize=(8, 4))
    for k in range(em_clf.n_clusters):
        img = em_clf.reconstructed_centers[k, :].reshape(28, 28).T
        ax[k].imshow(img, cmap='gray')
        ax[k].axis('off')
        ax[k].set_title(f'Average image of\n Gaussian component {k + 1}')
    plt.tight_layout()
    plt.show()

    true_labels_bin = np.array([1 if x == 2 else 0 for x in true_labels])
    em_clf.score(true_labels_bin)
    conf_mat = em_clf.confusion_matrix

    scores = pd.DataFrame({'labels': [6, 2],
                          'accuracy':[conf_mat[0,0]/conf_mat.sum(axis = 0)[0],
                                      conf_mat[1,1]/conf_mat.sum(axis = 0)[1]],
                          'mis_clf_rate':[conf_mat[1, 0]/conf_mat.sum(axis=0)[0],
                                      conf_mat[0, 1]/conf_mat.sum(axis=0)[1]]})

    print(f'confusion matrix:\n {em_clf.confusion_matrix}')
    print(f'accuracy and misclassification rates:\n {scores}')
