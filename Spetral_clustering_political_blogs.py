import os
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans

import networkx as nx
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "5"

class PoliticalBlogsClustering:
    """
    Spectral clustering for political blogs network data.

    Spectral clustering on a graph from by edges and nodes,
    compares clusters to true labels to compute mismatch rates.
    """
    def __init__(self, edges_path, nodes_path, seed = None):
        '''
        args:
            edges_path: file for edges
            nodes_path: file for nodes;
                        nodes file has 4 columns: 1-based idx, link, real_label, tag
            seed: random seed for Kmeans; will use random seed if None
        '''
        self.edges_path = edges_path
        self.nodes_path = nodes_path
        self.edges = self._load_edges()
        self.nodes = self._load_nodes()
        self.nodes0 = self.nodes.copy() ## save the original nodes
        self.adj_matrix = None
        self._kept_index = None if self.nodes is None else np.full(self.nodes.shape[0],
                                                                   True,
                                                                   dtype = bool)
        self.seed = seed
            
    def _load_edges(self):
        edges = np.loadtxt(self.edges_path)
        return edges-1

    def _load_nodes(self):
        nodes = pd.read_table(self.nodes_path,
                              header=None, 
                              names = ['No', 'Website', 'true_label', 'tags'])
        return nodes

    def _edge2matrix(self, remove_singulars = True):
        '''
        convert edge list to adjacency matrix

        args:
            remove_singulars: if True, remove isolated nodes (not connected to any other node)

        returns:
            tuple: (adj_matrix, kept_index)
        '''

        dim =  self.nodes.shape[0]
        adj_mat = csc_matrix(
                (np.ones(self.edges.shape[0]), (self.edges[:,0], self.edges[:,1])),
                shape = (dim,dim))

        if (adj_mat!=adj_mat.T).nnz > 0:
            adj_mat = adj_mat + adj_mat.T
            
        ## remove isolated nodes (no connection or only self-connected)
        keep = np.full((dim, ), True, dtype=bool)
        if remove_singulars:
            row_sum = adj_mat.sum(axis = 0).A1
            z_mks = row_sum == 0
            
            a_diag = adj_mat.diagonal()
            self_only = (a_diag == 1) & (row_sum == 1)
            
            keep = ~(z_mks | self_only)
            adj_mat = adj_mat[keep,:][:,keep]
        self.adj_matrix = adj_mat
        self._kept_index = keep
        self.nodes = self.nodes[keep].reset_index(drop = True)

        # return self.adj_matrix, self._kept_index, self.nodes

    def _spec_cluster(self, k):
        """
        Spectral clustering using k-means on eigenvectors.
        returns:
            tuple: (cluster_labels, cluster_centers)
        """

        A = self.adj_matrix.todense()
        D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1)
        L = D @ A @ D
        L = np.array(L)

        eig_val, eig_vec = np.linalg.eig(L)
        e_idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, e_idx[:k]]
        
        KM = (KMeans(n_clusters = k, init = 'random', random_state = self.seed)
              .fit(eig_vec.real))
        self.assigned_labels = KM.labels_
        self.cluster_centers = KM.cluster_centers_
        return KM.labels_, KM.cluster_centers_

    def _find_majority_labels(self, cls_assignment, real_class):
        assert len(cls_assignment) == len(real_class), \
            "predicted and real classes need to be the same length"

        m = len(real_class)
        k = max(cls_assignment) + 1
        cls_matrix = csc_matrix(
            (np.ones(m), (np.arange(0,m,1),cls_assignment)),
            shape = (m,k)
        )
        
        rate_1s = (real_class.T @ cls_matrix)/cls_matrix.sum(axis = 0).A1
        mm_rates = np.array([r if r<0.5 else 1-r for r in rate_1s])
        maj_labs = np.array([ int(0) if r < 0.5 else int(1) for r in rate_1s])
        
        overall_mm_rate = sum(mm_rates * cls_matrix.sum(axis = 0).A1)/len(real_class)
        
        return maj_labs, mm_rates, overall_mm_rate

    def fit_predict(self, n_clusters):
        '''
        This method loads the data, performs spectral clustering  and reports the majority labels

        Inputs:
            n_clusters (int): The number of clusters to be created

        Output:
            A map with following attributes
            1. overall_mismatch_rate: <2 decimal places>
            2. mismatch_rates: [{"majority_index": <int>, "mismatch_rate": <2 decimal places>}]
        '''

        map = {
            "overall_mismatch_rate": None,
            "mismatch_rates": []
        }
        
        if self.seed is None:
            self.seed = np.random.seed()
        
        self._edge2matrix()
        real_labs = np.array(self.nodes.iloc[:,2])
        
        cls, centers = self._spec_cluster(k = n_clusters)
        
        maj_labs, mm_rates, oa_mm_rate = self._find_majority_labels(cls, real_labs)

        map['majority_labels'] = maj_labs
        map['overall_mismatch_rate'] = oa_mm_rate.round(2)
        map['mismatch_rates'] = [{'majority_index':int(v1),
                                  'mismatch_rate':v2.round(2)} 
                                  for v1, v2 in zip(maj_labs, mm_rates)]
        self.predict_ = map

if __name__ == "__main__":
    k = 5
    pbc = PoliticalBlogsClustering(nodes_path = 'data/nodes.txt',
                                   edges_path = 'data/edges.txt',
                                   seed = 1234)
    pbc.fit_predict(n_clusters=k)

    print(f"Overall Mismatch Rate: {pbc.predict_['overall_mismatch_rate']}")
    print("\nPer-Cluster Mismatch Rates:")
    for rate_info in pbc.predict_['mismatch_rates']:
        print(f"  Cluster with majority {rate_info['majority_index']}: "
              f"{rate_info['mismatch_rate']}")


    ## plot the network, color nodes by true labels.
    real_labels = pbc.nodes.iloc[:,2]
    G = nx.from_numpy_array(pbc.adj_matrix)
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(10, 6))
    nx.draw(G,
            pos,
            with_labels=False,
            node_size=40,
            node_color=real_labels,
            cmap=plt.cm.viridis,
            edge_color = 'gray')
    plt.title("Political Blogs Network\nColored by real labels")
    plt.show()

    ## plot the network colored by assigned clusters.
    plt.figure(figsize=(10, 6))
    nx.draw(G,
            pos,
            with_labels=False,
            node_size=40,
            node_color=pbc.assigned_labels,
            cmap=plt.cm.viridis,
            edge_color = 'gray')
    plt.title(f"Political Blogs Network clustering (k={k})\nColored by clusters")
    plt.show()
