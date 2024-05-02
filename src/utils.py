import numpy as np


def mean(x: np.ndarray) -> np.ndarray:
    """
    Mean vector calculation function
    :param x: (n_samples, n_features) data matrix
    :return: (1, n_features) mean vector
    """
    n_samples, n_features = x.shape[0], x.shape[1]

    mu = np.zeros((1, n_features))
    for feature in range(n_features):
        for data in x[:, feature]:
            mu[0, feature] += data

    mu = mu / n_samples
    return mu


def covariance1(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Covariance matrix calculation function 1
    References:
    Alpaydin, E. (2010). Introduction to Machine Learning (2nd ed.) Section 5.2.
    :param x: (n_samples, n_features) data matrix
    :param mu: (1, n_features) mean matrix of the data
    :return: (n_features, n_features) covariance matrix
    """
    n_samples, n_features = x.shape[0], x.shape[1]

    x_c = x - mu
    return x_c.T @ x_c / (n_samples - 1)


def covariance2(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Covariance matrix calculation function 2
    References:
    Alpaydin, E. (2010). Introduction to Machine Learning (2nd ed.) Section 5.2.
    :param x: (n_samples, n_features) data matrix
    :param mu: (1, n_features) mean matrix of the data
    :return: (n_features, n_features) covariance matrix
    """
    n_samples, n_features = x.shape[0], x.shape[1]

    return x.T @ x / (n_samples - 1) - mu.T @ mu
