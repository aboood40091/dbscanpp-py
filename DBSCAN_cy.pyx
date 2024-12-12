# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3str
import numpy as np
cimport numpy as np
from scipy.spatial import KDTree
cimport cython


ctypedef np.double_t DTYPE_t
ctypedef np.int_t ITYPE_t


cdef class DBSCAN:
    cdef double eps
    cdef int min_samples

    def __init__(self, double eps, int min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, np.ndarray[DTYPE_t, ndim=2] x):
        cdef int CONST_UNEXPLORED = -2
        cdef int CONST_NOISE = -1
        cdef int CONST_CLUSTER_START = 0

        cdef double eps = self.eps
        cdef int min_samples = self.min_samples

        # Ensure x is a contiguous array of doubles
        x = np.ascontiguousarray(x, dtype=np.double)
        cdef Py_ssize_t n = x.shape[0]
        cdef Py_ssize_t d = x.shape[1]  # Number of features

        # Create a memoryview of x
        cdef double[:, ::1] x_view = x

        # Initialize labels
        cdef np.ndarray[ITYPE_t, ndim=1] labels_array = np.full(n, CONST_UNEXPLORED, dtype=np.int_)
        cdef int[::1] labels = labels_array

        cdef int cluster = CONST_CLUSTER_START

        # Create a KD-tree for efficient neighbor lookup within radius `eps`
        kdtree = KDTree(x_view)

        cdef Py_ssize_t i, neighbor_i, pos
        cdef list neighbors, extended_neighbors

        for i in range(n):
            if labels[i] != CONST_UNEXPLORED:
                continue

            # Get neighbors within `eps` using KD-tree
            neighbors = kdtree.query_ball_point(x_view[i, :], eps)
            if len(neighbors) >= min_samples:
                # Start a new cluster
                labels[i] = cluster
                pos = 0

                while pos < len(neighbors):
                    neighbor_i = neighbors[pos]
                    pos += 1
                    if labels[neighbor_i] == CONST_NOISE:
                        labels[neighbor_i] = cluster
                    elif labels[neighbor_i] == CONST_UNEXPLORED:
                        labels[neighbor_i] = cluster

                        # Expand neighbors only if it's a core point
                        extended_neighbors = kdtree.query_ball_point(x_view[neighbor_i, :], eps)
                        if len(extended_neighbors) >= min_samples:
                            neighbors.extend(extended_neighbors)

                # Increment cluster label after finishing this cluster
                cluster += 1
            else:
                labels[i] = CONST_NOISE

        # Convert all remaining CONST_UNEXPLORED points to CONST_NOISE
        for i in range(n):
            if labels[i] == CONST_UNEXPLORED:
                labels[i] = CONST_NOISE

        return labels_array
