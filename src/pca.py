from utils import mean, covariance1, variance

import matplotlib.pyplot as plt
import numpy as np


class PCA:
    def __init__(self, n_components: int):
        """
        Constructor for PCA
        :param n_components: Number of components that will used
        """
        self.n_components = n_components
        self.x_pca = None

    @staticmethod
    def standardize(x):
        """
        Standardize the given input data
        :param x: (n_samples, n_features) data matrix
        :return: (n_samples, n_features) standardized data matrix
        """
        return (x - mean(x)) / np.sqrt(variance(x, mean(x)))

    def fit_transform(self, x):
        """
        Step by step applies the PCA to the input
        :param x: (n_samples, n_features) data matrix
        :return: None
        """
        # Step 1 - Standardize:
        x = self.standardize(x)

        # Step 2 - Compute covariance:
        cov_matrix = covariance1(x, np.zeros((1, 64)))

        # Step 3 - Calculate eigenvalues and eigenvectors:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 4 - Sort and select principal components
        selected_idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        components = eigenvectors[:, selected_idx]

        # Step 5 - Transform
        self.x_pca = x @ components

    def visualize(self, y):
        """
        Visualizer for the pca components
        :param y: (n_samples) labels
        :return:
        """
        np.random.seed(42)
        selected_indices = np.random.choice(range(self.x_pca.shape[0]), size=200, replace=False)

        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(self.x_pca[:, 0], self.x_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, ticks=range(10))

        # Annotate randomly selected 200 points
        for i in selected_indices:
            plt.annotate(str(int(y[i])), (self.x_pca[i, 0], self.x_pca[i, 1]), fontsize=9)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Projection of opdigits Dataset')
        plt.grid(True)
        plt.show()
