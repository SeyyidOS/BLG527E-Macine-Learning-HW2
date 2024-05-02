from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class BaseDiscriminantAnalysis(ABC):
    def __init__(self, classes: dict.keys):
        self.name = ''
        self.classes = classes
        self.prior_probs = {}

    def fit(self, y_train: np.ndarray) -> None:
        """
        Calculates the prior probabilities for the classes using training data
        :param y_train: (n_samples) class labels
        :return: None
        """
        for c in self.classes:
            self.prior_probs[c] = np.mean(y_train == c)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the predicted labels
        :param x: (n_samples, n_features) data matrix
        :return: (n_samples) class predictions
        """
        n_samples = x.shape[0]  # Extract n_samples

        probs = np.zeros((n_samples, len(self.classes)))  # Allocate memory for the predicted class labels

        # Loop through the classes and calculate class probabilities for each datapoint
        for i, c in enumerate(self.classes):
            probs[:, i] = self._calc_probs(x, c)
        return np.argmax(probs, axis=1)

    @abstractmethod
    def _calc_probs(self, x: np.ndarray, class_label: int) -> np.ndarray:
        """
        Calculates the class probabilities for each datapoint
        :param x: (n_samples, n_features) data matrix
        :param class_label: Class label that will be predicted
        :return: (n_samples) class label prediction probabilities
        """
        pass

    def visualize(self, x: np.ndarray, y: np.array, seed: int) -> None:
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))

        axs[0].contourf(xx, yy, z, alpha=0.3)
        axs[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='RdBu', edgecolor='k', s=20, alpha=0.7)
        axs[0].set_title(self.name + ' - Training Data')
        axs[0].set_xlabel('Feature 1')
        axs[0].set_ylabel('Feature 2')
        axs[0].grid(True)

        axs[1].contourf(xx, yy, z, alpha=0.3)
        axs[1].scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='RdBu', edgecolor='k', s=20, alpha=0.7)
        axs[1].set_title(self.name + ' - Test Data')
        axs[1].set_xlabel('Feature 1')
        axs[1].set_ylabel('Feature 2')
        axs[1].grid(True)

        plt.show()
