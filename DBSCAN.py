import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


class DBSCAN:
    eps: float
    min_samples: int

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(
        self,
        x: npt.NDArray
    ) -> npt.NDArray[np.int_]:
        CONST_UNEXPLORED = -2
        CONST_NOISE = -1
        CONST_CLUSTER_START = 0

        eps = self.eps
        min_samples = self.min_samples

        # Ensure x is a contiguous array of doubles
        x = np.ascontiguousarray(x, dtype=np.double)
        n = len(x)

        # Initialize labels
        labels = np.full(n, CONST_UNEXPLORED, np.int_)
        cluster = CONST_CLUSTER_START

        # Create a KD-tree for efficient neighbor lookup within radius `eps`
        kdtree = KDTree(x)

        for i in range(n):
            if labels[i] != CONST_UNEXPLORED:
                continue

            # Get neighbors within `eps` using KD-tree
            neighbors: list[int] = kdtree.query_ball_point(x[i], eps)
            if len(neighbors) >= min_samples:
                # Start a new cluster
                labels[i] = cluster
                pos = 0  # Initialize position index

                while pos < len(neighbors):
                    neighbor_i = neighbors[pos]
                    pos += 1
                    if labels[neighbor_i] == CONST_NOISE:
                        labels[neighbor_i] = cluster
                    elif labels[neighbor_i] == CONST_UNEXPLORED:
                        labels[neighbor_i] = cluster

                        # Expand neighbors only if it's a core point
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
