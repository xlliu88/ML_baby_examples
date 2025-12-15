from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import networkx as nx

## Implement of epsilon-ISOMAP to cluster images, which will preserve the visual structure of the images
class OrderOfFaces:
    def __init__(self, images_path='data/isomap.mat'):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """
        try:
            self.images = loadmat(images_path)['images']
        except:
            raise ValueError(f'file {images_path} not exists.')

        self.epsilon_ = None
        self.adjacency_matrix = None

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """

        if self.epsilon_ is None:
            self.get_best_epsilon()
        eps = self.epsilon_
        adj_mat = cdist(self.images.T, self.images.T)
        adj_mat[adj_mat>eps] = 0

        self.adjacency_matrix = adj_mat
        return adj_mat
        

    def get_best_epsilon(self):
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.
        """

        A_ = cdist(self.images.T, self.images.T, metric='euclidean')
        LB = A_.min()
        UB = A_.max()

        while True:
            step = (UB-LB)/10.0
            epsL = np.arange(LB, UB, step)
            n_inf = np.zeros(len(epsL))
            for i,eps in enumerate(epsL):
                mat = A_.copy()
                mat[mat>eps] = 0
                D = shortest_path(mat)
                n_inf[i] = np.sum(np.isinf(D))
                
            idxLow = np.where(np.array(n_inf)>0)[0].max()
            idxUpper = np.where(np.array(n_inf)==0)[0].min()

            if idxLow is None or idxUpper is None:
                raise ValueError("Cannot find suitable epsilon range")

            if step > 0.01:
                LB = epsL[idxLow]
                UB = epsL[idxUpper]
            else:
                self.epsilon_ = epsL[idxUpper].round(4)
                break
                # return epsL[idxUpper].round(4)
            
    def isomap(self, dim = 2) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.
        """
        
        A = self.get_adjacency_matrix()
        m = A.shape[0]
        D = shortest_path(A)#, directed = False)
        D[np.isinf(D)] = 0
        H = np.diag(np.ones(m)) - np.ones((m,m))/m
        G = -0.5 * (H @ (D ** 2) @ H)
        e_val, e_vec = np.linalg.eig(G)
        e_val = np.real(e_val)
        e_vec = np.real(e_vec)
        idx = np.argsort(e_val)[::-1]
        Z = e_vec[:,idx][:,:dim] @ np.diag(np.sqrt(e_val[idx][:dim]))
        
        return Z

    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.
        """
        
        B = self.images
        B_norm = B-B.mean(axis=0)
        C = B_norm.T.dot(B_norm)
        e_val, e_vec = eigh(C)
        idx = np.argsort(e_val)[::-1]
        PCs = e_vec[:, idx][:,:num_dim].dot(np.diag(np.sqrt(e_val[idx][:num_dim])))
        
        return PCs

if __name__ == "__main__":
    oof = OrderOfFaces(images_path="data/isomap.mat")
    oof.get_best_epsilon()
    A = oof.get_adjacency_matrix()

    ## plot Adjacency matrix ------------------------------------------------
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=1234)
    pos_arr = np.array(pd.DataFrame(pos).T)

    ## setup for image display
    n_img2show = 20
    axis_ranges = pos_arr.max(axis=0) - pos_arr.min(axis=0)
    imgsize = axis_ranges / 20
    xshift = 0.05

    ## select images to display - KMeans method
    km = KMeans(n_clusters=n_img2show, random_state=0).fit(pos_arr)
    dist = cdist(pos_arr, km.cluster_centers_)
    img_idx = dist.argmin(axis=0)
    img2show = oof.images[:, img_idx].T
    coord = pos_arr[img_idx, :]
    extents = np.vstack((coord[:, 0] + xshift,
                         coord[:, 0] + imgsize[0] * 2 * 0.8 + xshift,
                         coord[:, 1] - imgsize[1],
                         coord[:, 1] + imgsize[1])).T

    canvas_lim = np.vstack((pos_arr.min(axis=0) - 0.1 * axis_ranges,
                            pos_arr.max(axis=0) + 0.1 * axis_ranges)).round()

    ## make the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    nx.draw(G,
            pos,
            with_labels=False,
            node_size=20,
            edge_color='gray')
    ax.scatter(coord[:, 0], coord[:, 1], marker='o', s=30, facecolor='none', edgecolor='r', zorder=2)
    for i in range(n_img2show):
        ax.imshow(img2show[i, :].reshape(64, 64).T, cmap='gray', extent=extents[i, :], zorder=3)
    ax.set_xlim(canvas_lim[:, 0])
    ax.set_ylim(canvas_lim[:, 1])
    plt.title(f"Adjacency matrix connectivity\n (eps = {oof.epsilon_})", x=0.5, y=0.8)
    plt.show()

    ## =======================================================================================
    ## calculate and display the first two dimension of isomap
    ## =======================================================================================
    Z = oof.isomap(dim = 2)
    Z[:,0] = -Z[:,0]  ## flip the x-axis for better visual

    ## image display setup
    n_img2show = 20
    axis_ranges = Z.max(axis=0) - Z.min(axis=0)
    imgsize = axis_ranges / 20
    yshift = imgsize[1] / 2

    ## use KMeans to select images for display
    km = KMeans(n_clusters=n_img2show, random_state=0).fit(Z.real)
    dist = cdist(Z, km.cluster_centers_)
    img_idx = dist.argmin(axis=0)
    img2show = oof.images[:, img_idx].T

    ## image location and canvas limits
    extents = np.vstack((Z[img_idx][:, 0] - imgsize[0] * 0.8,
                         Z[img_idx][:, 0] + imgsize[0] * 0.8,
                         Z[img_idx][:, 1] - imgsize[1] * 2 - yshift,
                         Z[img_idx][:, 1] - yshift)).T

    canvas_lim = np.vstack((Z.min(axis=0) - 0.1 * axis_ranges,
                            Z.max(axis=0) + 0.1 * axis_ranges)).round()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(Z[:, 0], Z[:, 1], zorder=1)
    ax.scatter(Z[img_idx, 0], Z[img_idx, 1], marker='o', s=100, facecolor='none', edgecolor='r', zorder=3)
    ax.set_xlim(canvas_lim[:, 0])
    ax.set_ylim(canvas_lim[:, 1])
    for i in range(0, n_img2show):
        ax.imshow(img2show[i, :].reshape(64, 64).T, cmap='gray', extent=extents[i, :], zorder=3)
    ax.set_title('ISOMAP Dimensional Reduction')
    ax.set_aspect('auto')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.show()


## =======================================================================================
## calculate and display the first two dimension of PCA
## note that the arrange of images does not preserve the visual structure.
## =======================================================================================
    PCs = oof.pca(num_dim=2)

    n_img2show = 20
    axis_ranges = PCs.max(axis=0) - PCs.min(axis=0)
    imgsize = axis_ranges/25
    xshift = imgsize[0]/2.5

    ## select images to display
    km2 = KMeans(n_clusters = n_img2show, random_state = 0).fit(PCs)
    dist2 = cdist(PCs, km2.cluster_centers_)
    img_idx2 = np.argmin(dist2, axis=0)
    img2show = oof.images[:, img_idx2].T

    ## image size and location
    extents2 = np.vstack((PCs[img_idx2][:,0]-imgsize[0]*2*0.8-xshift,
                         PCs[img_idx2][:,0]-xshift,
                         PCs[img_idx2][:,1]-imgsize[1],
                         PCs[img_idx2][:,1]+imgsize[1])).T

    canvas_limit = np.vstack((PCs.min(axis=0)-0.05*axis_ranges,
                            PCs.max(axis=0)+0.05*axis_ranges)).round()

    fig2, ax2 = plt.subplots(1,1, figsize = (10,8))
    ax2.scatter(PCs[:,0], PCs[:,1],zorder = 1)
    ax2.scatter(PCs[img_idx2,0], PCs[img_idx2,1], marker = 'o', s = 100, facecolor = 'none', edgecolor = 'red', zorder=2)
    ax2.set_xlim(canvas_limit[:,0])
    ax2.set_ylim(canvas_limit[:,1])
    for i in range(0,n_img2show):
        ax2.imshow(img2show[i,:].reshape(64,64).T, cmap = 'gray', extent = extents2[i,:], zorder=3)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_aspect('auto')
    ax2.set_title('PCA Dimensional Reduction')
    plt.show()


