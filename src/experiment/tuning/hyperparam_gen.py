import numpy as np


class OptimParamGenerator:
    """
    Generate parameters like number of hidden units, learning rate, N for finite diff, etc
    """

    def __init__(self, seed):
        # for reproducibility
        self.seed = seed

    def learning_rate_gen(self):
        """
        sample a learning rate
        """
        log_learning_rate = np.random.uniform(-7.5, -4.5)
        learning_rate = np.exp(log_learning_rate)
        return learning_rate

