import numpy as np


class ExpectationMaximization:
    def __init__(self, max_iters=10000, threshold=1e-6):
        """
        Constructor for the expectation maximization
        :param max_iters: Maximum number of iterations
        :param threshold: Threshold value for the convergence
        """
        self.max_iters = max_iters
        self.threshold = threshold

    def __str__(self):
        """
        String representation of the found parameters
        :return: String
        """
        return (f'QA: {self.Q_A:.4f}, QB: {self.Q_B:.4f}, '
                f'pi_A: {self.pi_A:.4f}, pi_B: {self.pi_B:.4f}')

    def _initialize_parameters(self):
        """
        Parameter initialization function
        :return: None
        """
        self.pi_A, self.pi_B, self.prev_pi_A, self.prev_pi_B = 0.5, 0.5, np.inf, np.inf
        self.Q_A, self.Q_B, self.prev_Q_A, self.prev_Q_B = np.random.rand(), np.random.rand(), np.inf, np.inf

    def _expectation_step(self, data):
        """
        Expectation step for the expectation maximization algorithm
        :param data: (n_experimenst, n_coin_tosses) shaped data
        :return: (n_experiments, 2) shaped data
        """
        expectations = np.zeros((data.shape[0], 2))
        for i, x in enumerate(data):
            p_A = self.pi_A * self.Q_A ** np.sum(x) * (1 - self.Q_A) ** (data.shape[1] - np.sum(x))
            p_B = self.pi_B * self.Q_B ** np.sum(x) * (1 - self.Q_B) ** (data.shape[1] - np.sum(x))
            total = p_A + p_B
            expectations[i, :] = [p_A / total, p_B / total]

        self.prev_pi_A, self.prev_pi_B = self.pi_A, self.pi_B
        self.pi_A = expectations[:, 0].sum() / len(data)
        self.pi_B = expectations[:, 1].sum() / len(data)

        return expectations

    def _maximization_step(self, data, expectations):
        """
        Maximization step for the expectation maxmization algorithm
        :param data:
        :param expectations:
        :return: None
        """
        number_of_heads_each_exp = [sum(obs) for obs in data]
        self.prev_Q_A, self.prev_Q_B = self.Q_A, self.Q_B
        self.Q_A = (expectations[:, 0] @ number_of_heads_each_exp) / (expectations[:, 0].sum() * 10)
        self.Q_B = (expectations[:, 1] @ number_of_heads_each_exp) / (expectations[:, 1].sum() * 10)

    def _is_converged(self):
        """
        Convergence checker funtion
        :return: Boolean
        """
        if abs(self.prev_Q_A - self.Q_A) < self.threshold and \
                abs(self.prev_Q_B - self.Q_B) < self.threshold:
            return True
        return False

    def fit(self, data):
        """
        Expectatoin maximization algorithm runner function
        :param data:
        :return:
        """
        self._initialize_parameters()
        for _ in range(self.max_iters):
            responsibilities = self._expectation_step(data)
            self._maximization_step(data, responsibilities)

            if self._is_converged():
                break


if __name__ == "__main__":
    # 0->T, 1->H
    data = [
        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    ]

    em = ExpectationMaximization()
    em.fit(np.array(data))
    print(em)
