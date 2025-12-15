## A Kmean implementation to compress images
## 

import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csc_matrix, find
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# from KMeans import KMeansImpl

class KMeansImpl:
    def __init__(self):
        pass

    def load_image(self, image_name="1.jpeg"):
        """
        Returns the image numpy array.
        It is important that image_name parameter defaults to the choice image name.
        """
        return np.array(Image.open(image_name))

    def initial_centers(self, X, k, sample_size=3):
        ## take average of 3 pixels as the center

        m, d = X.shape
        _idx = np.random.choice(range(m), sample_size * k, replace=False)
        centers = X[_idx, :].reshape(k, sample_size, d).mean(axis=1)
        return centers

    def c_dist(self, X, y, norm=2):
        ## implemented with the cdist function from scipy.
        ## much faster.

        if (norm == 1):
            return cdist(X, y, metric='cityblock')
        elif (norm == 2):
            return cdist(X, y, metric='euclidean')
        else:
            return cdist(X, y, metric='minkowski', p=norm)

    def dist(self, X, y, norm=2):
        ## X, m by d matrix, each row is a data point
        ## y, k by d matrix as centers
        ## norm:  distance norm

        ## returns a m x k matrix representing distance of each dp to each center
        if (norm == 1):
            return np.sum(np.abs(X[:, np.newaxis, :] - y[np.newaxis, :, :]), axis=2)
        elif (norm == 2):
            Y2 = np.sum(np.power(y, 2), axis=1)
            Y1 = -2 * X.dot(y.T)

            return Y1 + Y2[np.newaxis, :]

        elif (norm == np.inf):
            d = X[:, np.newaxis, :] - y[np.newaxis, :, :]
            return np.max(np.abs(d), axis=2)
        else:
            d = X[:, np.newaxis, :] - y[np.newaxis, :, :]
            return (np.sum((np.abs(d)) ** norm, axis=2)) ** (1 / norm)

    def WCSS(self, X, y):
        D = self.c_dist(X, y)
        msd = np.min(D, axis=1)
        return (np.sum(msd))

    def assign_classes(self, X, centers, norm=2):
        ## to assign each data point (row) of X to a centroid in y.

        assert X.shape[1] == centers.shape[1]
        distXy = self.c_dist(X, centers, norm)
        classes = np.argmin(distXy, axis=1)

        return classes

    def update_centers(self, X, classes):
        ## vectorized implementation
        ## X: data points
        ## classes: classes assignment of each data point (row) in X
        ## norm: distance-norm

        ## return a matrix of new centers

        k = len(np.unique(classes))
        m, d = X.shape
        c = csc_matrix((np.ones(m), (np.arange(0, m, 1), classes)), shape=(m, k))
        nzcol = (c.sum(axis=0).A1 != 0)
        c = c[:, nzcol]
        n = c.sum(axis=0)
        centers = (c.T.dot(X)) / (n.T)
        return np.array(centers)

    def assert_eq_matrix(self, X, Y):
        # eps = np.finfo(float).eps
        if np.isclose(X, Y).all():
            return True
        else:
            return False

    def compress(self, pixels, num_clusters, norm_distance=2):
        """
        Compress the image using K-Means clustering.

        Parameters:
            pixels: 3D image for each channel (a, b, 3), values range from 0 to 255.
            num_clusters: Number of clusters (k) to use for compression.
            norm_distance: Type of distance metric to use for clustering.
                            Can be 1 for Manhattan distance or 2 for Euclidean distance.
                            Default is 2 (Euclidean).

        Returns:
            Dictionary containing:
                "class": Cluster assignments for each pixel.
                "centroid": Locations of the cluster centroids.
                "img": Compressed image with each pixel assigned to its closest cluster.
                "number_of_iterations": total iterations taken by algorithm
                "time_taken": time taken by the compression algorithm
        """
        map = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {}
        }

        t0 = time.time()
        w, h, d = pixels.shape
        pixels2d = pixels.reshape(w * h, d)

        centers = self.initial_centers(pixels2d, num_clusters)
        iters = 0
        while True:

            ## update classes assignment and new centers

            cls = self.assign_classes(X=pixels2d, centers=centers, norm=norm_distance)
            new_centers = self.update_centers(X=pixels2d, classes=cls)
            iters += 1

            if (new_centers.shape[0] < centers.shape[0]):
                continue

            converged_centers = np.isclose(new_centers, centers).all()
            if (~converged_centers):
                centers = new_centers
                continue

            ## converged centers;
            ## generated compressed images
            c = csc_matrix((np.ones(w * h), (np.arange(0, w * h, 1), cls)), shape=(w * h, len(set(cls))))
            new_img2d = c.dot(new_centers)

            map['class'] = cls
            map['centroid'] = new_centers
            map['img'] = new_img2d.reshape(w, h, d)
            map['number_of_iterations'] = iters
            map['time_taken'] = time.time() - t0
            map['additional_args'] = {}

            return map

if __name__ == "__main__":
    test_imgs = ['football.bmp','parrots.png', 'tomato_s.jpg']
    norms = [1,2]
    ks = [3,6,12,24,48]
    seed = 1234

    np.random.seed(seed)
    km = KMeansImpl()
    results = [ {f'{img}_{norm}_{k}': km.compress(pixels = km.load_image(f'data/{img}'), num_clusters = k, norm_distance=norm)}
               for img in test_imgs for norm in norms for k in ks]

    cimgs = [v['img'] for r in results for k,v in r.items()]
    centroids = [v['centroid'] for r in results for k,v in r.items() ]
    cls_labs = [v['class'] for r in results for k,v in r.items() ]


    ## compressed result of the first image
    img = test_imgs[0]
    cimgs_0 = cimgs[0:10]

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes.flat):
        ax.imshow(cimgs_0[i].astype(np.uint8))
        ax.axis('off')
    fig.suptitle(f'Kmeans Compressed images for {img} | seed: {seed}.\nTop: norm 1;    bottom: norm2. \nLeft to right: k= 3, 6, 12, 24, 48',
                 x = 0.01,
                 y = 0.9,
                 ha = 'left',
                fontsize = 12)
    plt.tight_layout()
    plt.show()

    ## compress result of the second image
    img = test_imgs[1]
    cimgs_1 = cimgs[10:20]

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes.flat):
        ax.imshow(cimgs_1[i].astype(np.uint8))
        ax.axis('off')
    fig.suptitle(f'Kmeans Compressed images for {img} | seed: {seed}.\nTop: norm 1;    bottom: norm2. \nLeft to right: k= 3, 6, 12, 24, 48',
                 x = 0.01,
                 y = 0.9,
                 ha = 'left')
    plt.tight_layout()
    plt.show()

    ## compress result of the third image
    img = test_imgs[2]
    cimgs_2 = cimgs[20:30]

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes.flat):
        ax.imshow(cimgs_2[i].astype(np.uint8))
        ax.axis('off')
    fig.suptitle(f'Kmeans Compressed images for {img} | seed: {seed}.\nTop: norm 1;    bottom: norm2. \nLeft to right: k= 3, 6, 12, 24, 48',
             x = 0.01,
             y = 0.9,
             ha = 'left',
            fontsize = 12)
    plt.tight_layout()
    plt.show()

    ## running stats
    samples = [(x, y, z) for x in test_imgs for y in norms for z in ks]
    times = [v['time_taken'] for r in results for k, v in r.items()]
    n_iters = [v['number_of_iterations'] for r in results for k, v in r.items()]

    summ = pd.DataFrame(samples, columns=['Image', 'norm', 'k'])
    summ.loc[:, 'Iterations'] = n_iters
    summ.loc[:, 'time_taken'] = times
    summ

    ## calculate WCSS for different k
    samples = [ (x, y, z) for x in test_imgs for y in norms for z in ks]
    WCSS = [0]*len(samples)
    for i, img in enumerate(test_imgs):
        px = km.load_image(f'data/{img}')
        w,h,d = px.shape
        px2d = px.reshape(w*h,d)
        cen_1 = centroids[10*i:10*(i+1)]
        clss = cls_labs[10*i:10*(i+1)]
        WCSS[10*i:10*(i+1)] = [km.c_dist(px2d, cen).min(axis=1).sum().round(2) for cen in cen_1]

    summ['WCSS']  = WCSS

    ## plot WCSS score
    fct = sns.FacetGrid(summ, col='Image')
    fct.map_dataframe(sns.lineplot, x='k', y='WCSS', hue='norm', palette={1:'green', 2:'orange'})
    fct.set(xticks=ks)
    fct.set_xlabels('')
    fct.fig.text(0.5, 0.04, 'number of centers', ha = 'center', fontsize = 12)
    fct.add_legend(title='norm')
    plt.show()