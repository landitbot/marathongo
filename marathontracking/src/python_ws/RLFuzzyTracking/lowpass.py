import numpy as np


class LowPass2nd:
    """
    二阶低通：两级一阶低通级联
    y1 = alpha * x  + (1-alpha) * y1_prev
    y  = alpha * y1 + (1-alpha) * y_prev
    """

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.y1 = None  # 第一级记忆
        self.y = None  # 第二级记忆

    def compute(self, x: float) -> float:
        if self.y1 is None:
            self.y1 = x
            self.y = x
            return x

        self.y1 = self.alpha * x + (1 - self.alpha) * self.y1
        self.y = self.alpha * self.y1 + (1 - self.alpha) * self.y
        return self.y


class LowPass:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.last_v = None

    def compute(self, v: float):
        if self.last_v is None:
            return v
        else:
            return v * self.alpha + self.last_v * (1 - self.alpha)



def mean_shift(X, bandwidth, max_iters=100, tol=1e-4):
    """
    Mean Shift Clustering Algorithm

    :param X: Input data points [N, features]
    :param bandwidth: Bandwidth for mean shift
    :param max_iters: Maximum number of iterations
    :param tol: Tolerance for convergence
    """
    n_samples, n_features = X.shape
    centroids = np.copy(X)
    cluster_size = np.zeros_like(X)

    for _ in range(max_iters):
        new_centroids = np.zeros_like(centroids)
        new_cluster_size = np.zeros_like(cluster_size)

        for i in range(n_samples):
            # Compute distances to all points
            distances = np.linalg.norm(X - centroids[i], axis=1)
            # Select points within the bandwidth
            in_bandwidth = X[distances < bandwidth]
            # Update centroid
            new_cluster_size[i] = len(in_bandwidth)
            if len(in_bandwidth) > 0:
                new_centroids[i] = in_bandwidth.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids
        cluster_size = new_cluster_size

    # Assign labels based on final centroids
    unique_centroids, index = np.unique(centroids, axis=0, return_index=True)
    unique_cluster_size = cluster_size[index, :]
    return unique_centroids, unique_cluster_size