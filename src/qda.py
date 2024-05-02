from base_discriminant_analysis import BaseDiscriminantAnalysis

import numpy as np


class QDA(BaseDiscriminantAnalysis):
    def __init__(self, mean_vectors: dict, cov_matrices: dict):
        """
        Constructor for QDA
        :param mean_vectors: Dictionary that includes the mean vector for each classes
        :param cov_matrices: Dictionary that includes the mean vector for each classes
        """
        super().__init__(mean_vectors.keys())
        self.name = 'QDA'
        self.mean_vectors = mean_vectors
        self.cov_matrices = cov_matrices

    def _calc_probs(self, x: np.ndarray, class_label: int) -> np.ndarray:
        """
        Calculates the class probabilities for each datapoint
        :param x: (n_samples, n_features) data matrix
        :param class_label: Class label that will be predicted
        :return: (n_samples) class label prediction probabilities
        """
        n_samples = x.shape[0]  # Extract n_samples

        # Calculate the variables that will be used for the calculation
        mean_vec = self.mean_vectors[class_label]
        cov_mat = self.cov_matrices[class_label]

        inv_cov_mat = np.linalg.inv(cov_mat)
        det_cov_mat = np.linalg.det(cov_mat)

        probs = np.zeros(n_samples)  # Allocate memory
        for i in range(n_samples):
            x_c = x[i, :] - mean_vec  # Calculate the centered mean for the current data point

            first = (-1 / 2) * np.log(det_cov_mat)  # First equation
            second = (1 / 2) * (x_c @ inv_cov_mat @ x_c.T).item()  # Second equation
            probs[i] = first - second + np.log(self.prior_probs[class_label])  # Final value

        return probs
