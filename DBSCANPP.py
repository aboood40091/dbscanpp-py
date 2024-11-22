import enum
from math import ceil
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from scipy.spatial.distance import sqeuclidean
from sklearn.cluster import KMeans


class SamplingType(enum.Enum):
    Linspace = enum.auto()
    Uniform = enum.auto()
    KCenter = enum.auto()
    KMeansPP = enum.auto()


def k_centers(m: int, x: npt.NDArray) -> npt.NDArray[np.int_]:
    result = np.empty(m, dtype=np.int_)

    # Initialize the first center to index 0
    center_id = 0
    result[0] = center_id

    # Precompute squared norms of all points
    norms_sq = np.einsum('ij,ij->i', x, x)

    # Compute squared distances from all points to the first center
    closest_dist_sq = norms_sq + norms_sq[center_id] - 2 * np.dot(x, x[center_id])

    for c in range(1, m):
        # Select the point that is farthest from its closest center
        center_id = np.argmax(closest_dist_sq)
        result[c] = center_id

        # Compute squared distances from all points to the new center
        dist_sq_new_center = norms_sq + norms_sq[center_id] - 2 * np.dot(x, x[center_id])

        # Update closest distances
        np.minimum(closest_dist_sq, dist_sq_new_center, out=closest_dist_sq)

    return result


def k_means_pp(m: int, x: npt.NDArray) -> npt.NDArray[np.int_]:
    kmeans = KMeans(n_clusters=m, init="k-means++", random_state=42).fit(x)
    centers = kmeans.cluster_centers_
    f_computeDistance = sqeuclidean  # TODO: Replace this with euclidean for better accuracy?
    return np.fromiter((np.argmin([f_computeDistance(point, center) for point in x]) for center in centers), dtype=np.int_, count=len(centers))


class DBSCANPP:
    eps: float
    min_samples: int

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(
        self,
        x: npt.NDArray,
        *,
        ratio: Optional[float] = None,
        p: Optional[float] = None,
        sampling_type: SamplingType = SamplingType.Linspace
    ) -> npt.NDArray[np.int_]:
        n, d = x.shape
        assert n > 0
        if ratio is not None:
            m = int(ceil(ratio * n))
        else:
            assert p is not None
            m = int(p * n ** (d / (d + 4)))
        m = min(max(m, 1), n)  # Clamp m between 1 and n

        if m == n:
            subset_indices = np.arange(n, dtype=np.int_)
            is_in_subset = np.ones(n, dtype=np.bool_)
        else:
            if sampling_type == SamplingType.Uniform:
                subset_indices = np.sort(np.random.choice(np.arange(n, dtype=np.int_), m, replace=False))
            elif sampling_type == SamplingType.KCenter:
                subset_indices = k_centers(m, x)
            elif sampling_type == SamplingType.KMeansPP:
                subset_indices = k_means_pp(m, x)
            else:  # sampling_type == SamplingType.Linspace
                subset_indices = np.linspace(0, n - 1, m, dtype=np.int_)

            is_in_subset = np.zeros(n, dtype=np.bool_)
            is_in_subset[subset_indices] = True

        CONST_UNEXPLORED = -2
        CONST_NOISE = -1
        CONST_CLUSTER_START = 0

        eps = self.eps
        min_samples = self.min_samples

        labels = np.full(n, CONST_UNEXPLORED, np.int_)
        cluster = CONST_CLUSTER_START

        # Create a KD-tree for efficient neighbor lookup within radius `eps`
        kdtree = KDTree(x)

        for i in subset_indices:
            if labels[i] != CONST_UNEXPLORED:
                continue

            # Get neighbors within `eps` using KD-tree
            neighbors: list[int] = kdtree.query_ball_point(x[i], eps)
            if len(neighbors) >= min_samples:
                # Start a new cluster
                labels[i] = cluster

                while neighbors:
                    neighbor_i = neighbors.pop(0)
                    if labels[neighbor_i] == CONST_NOISE:
                        labels[neighbor_i] = cluster
                    elif labels[neighbor_i] == CONST_UNEXPLORED:
                        labels[neighbor_i] = cluster

                        # Expand neighbors only if it's a core point
                        if is_in_subset[neighbor_i]:
                            extended_neighbors = kdtree.query_ball_point(x[neighbor_i], eps)
                            if len(extended_neighbors) >= min_samples:
                                neighbors.extend(extended_neighbors)
                
                # Increment cluster label after finishing this cluster
                cluster += 1
            else:
                labels[i] = CONST_NOISE

        # Convert all remaining CONST_UNEXPLORED points to CONST_NOISE
        labels[labels == CONST_UNEXPLORED] = CONST_NOISE

        return labels
