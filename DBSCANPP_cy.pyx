# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3str
from libc.math cimport ceil
import numpy as np
cimport numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import sqeuclidean
from sklearn.cluster import KMeans
cimport cython


ctypedef np.double_t DTYPE_t
ctypedef np.int_t ITYPE_t
ctypedef np.npy_bool BTYPE_t


cdef enum SamplingType:
    Linspace = 1
    Uniform = 2
    KCenter = 3
    KMeansPP = 4


def k_centers(int m, np.ndarray[DTYPE_t, ndim=2] x):
    """
    Select m centers from x using the k-centers algorithm.
    """
    # Ensure x is contiguous
    x = np.ascontiguousarray(x, dtype=np.double)
    cdef DTYPE_t[:, ::1] x_view = x

    cdef np.ndarray[ITYPE_t, ndim=1] result_array = np.empty(m, dtype=np.int_)
    cdef ITYPE_t[::1] result = result_array

    # Initialize the first center to index 0
    cdef ITYPE_t center_id = 0
    result[0] = center_id

    # Precompute squared norms of all points
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t d = x.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] norms_sq_array = np.empty(n, dtype=np.double)
    cdef DTYPE_t[::1] norms_sq = norms_sq_array
    cdef Py_ssize_t i, j
    cdef DTYPE_t tmp
    for i in range(n):
        tmp = 0.0
        for j in range(d):
            tmp += x_view[i, j] * x_view[i, j]
        norms_sq[i] = tmp

    # Compute squared distances from all points to the first center
    cdef np.ndarray[DTYPE_t, ndim=1] closest_dist_sq_array = np.empty(n, dtype=np.double)
    cdef DTYPE_t[::1] closest_dist_sq = closest_dist_sq_array
    cdef DTYPE_t[::1] x_center_view = x_view[center_id, :]
    for i in range(n):
        tmp = 0.0
        for j in range(d):
            tmp += x_view[i, j] * x_center_view[j]
        closest_dist_sq[i] = norms_sq[i] + norms_sq[center_id] - 2.0 * tmp

    f_argmax = np.argmax

    cdef np.ndarray[DTYPE_t, ndim=1] dist_sq_new_center_array = np.empty(n, dtype=np.double)
    cdef DTYPE_t[::1] dist_sq_new_center = dist_sq_new_center_array
    cdef Py_ssize_t c
    for c in range(1, m):
        # Select the point that is farthest from its closest center
        center_id = f_argmax(closest_dist_sq)  # Manual implementation of np.argmax is actually SLOWER
        result[c] = center_id

        # Compute squared distances from all points to the new center
        x_center_view = x_view[center_id, :]
        for i in range(n):
            tmp = 0.0
            for j in range(d):
                tmp += x_view[i, j] * x_center_view[j]
            dist_sq_new_center[i] = norms_sq[i] + norms_sq[center_id] - 2.0 * tmp

        # Update closest distances
        for i in range(n):
            if dist_sq_new_center[i] < closest_dist_sq[i]:
                closest_dist_sq[i] = dist_sq_new_center[i]

    return result_array


def k_means_pp(int m, np.ndarray[DTYPE_t, ndim=2] x):
    kmeans = KMeans(n_clusters=m, init="k-means++", random_state=42).fit(x)
    centers = kmeans.cluster_centers_
    f_computeDistance = sqeuclidean  # TODO: Replace this with euclidean for better accuracy?
    cdef Py_ssize_t centers_n = len(centers)
    cdef Py_ssize_t n = len(x)
    cdef np.ndarray[ITYPE_t, ndim=1] result = np.empty(centers_n, dtype=np.int_)
    cdef np.ndarray[DTYPE_t, ndim=1] distances = np.empty(n, dtype=np.double)
    cdef Py_ssize_t i, j
    for i in range(centers_n):
        center = centers[i]
        for j in range(n):
            distances[j] = f_computeDistance(x[j], center)
        result[i] = np.argmin(distances)
    return result


cdef class DBSCANPP:
    cdef double eps
    cdef int min_samples

    def __init__(self, double eps, int min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self,
                    np.ndarray[DTYPE_t, ndim=2] x,
                    ratio=None,
                    p=None,
                    sampling_type=SamplingType.Linspace):
        # Ensure x is a contiguous array of doubles
        x = np.ascontiguousarray(x, dtype=np.double)
        cdef Py_ssize_t n = x.shape[0]
        cdef Py_ssize_t d = x.shape[1]  # Number of features

        # Create a memoryview of x
        cdef double[:, ::1] x_view = x

        assert n > 0

        cdef double ratio_value, p_value
        cdef Py_ssize_t m

        if ratio is not None:
            ratio_value = ratio
            m = <Py_ssize_t> ceil(ratio_value * n)
        else:
            assert p is not None
            p_value = p
            m = <Py_ssize_t> (p_value * n ** (d / (d + 4)))
        m = min(max(m, 1), n)  # Clamp m between 1 and n

        cdef np.ndarray[ITYPE_t, ndim=1] subset_indices_array
        cdef ITYPE_t[::1] subset_indices
        cdef np.ndarray[BTYPE_t, ndim=1] is_in_subset_array = np.zeros(n, dtype=np.bool_)
        cdef BTYPE_t[::1] is_in_subset = is_in_subset_array

        if m == n:
            subset_indices_array = np.arange(n, dtype=np.int_)
            subset_indices = subset_indices_array
            is_in_subset_array[:] = True
        else:
            if sampling_type == SamplingType.Uniform:
                subset_indices_array = np.random.choice(np.arange(n, dtype=np.int_), m, replace=False)
                subset_indices = subset_indices_array
            elif sampling_type == SamplingType.KCenter:
                subset_indices_array = k_centers(m, x)
                subset_indices = subset_indices_array
            elif sampling_type == SamplingType.KMeansPP:
                subset_indices_array = k_means_pp(m, x)
                subset_indices = subset_indices_array
            else:  # SamplingType.Linspace
                subset_indices_array = np.linspace(0, n - 1, m, dtype=np.int_)
                subset_indices = subset_indices_array

            for idx in subset_indices:
                is_in_subset[idx] = True

        cdef int CONST_UNEXPLORED = -2
        cdef int CONST_NOISE = -1
        cdef int CONST_CLUSTER_START = 0

        cdef double eps = self.eps
        cdef int min_samples = self.min_samples

        # Initialize labels
        cdef np.ndarray[ITYPE_t, ndim=1] labels_array = np.full(n, CONST_UNEXPLORED, dtype=np.int_)
        cdef ITYPE_t[::1] labels = labels_array

        cdef int cluster = CONST_CLUSTER_START

        # Create a KD-tree for efficient neighbor lookup within radius `eps`
        kdtree = KDTree(x_view)

        cdef ITYPE_t i
        cdef Py_ssize_t neighbor_i, pos
        cdef list neighbors, extended_neighbors

        for i in subset_indices:
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
                        if is_in_subset[neighbor_i]:
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
