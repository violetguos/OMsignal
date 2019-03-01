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


class AeParamGenerator():
    def __init__(self):
        pass
    # TODO: kernel size cannot be < 0 after 6 max pools from the CNN params

    def record(self):
        # TODO: write the numbers to a config file
        pass

class CnnParamGenerator():
    def __init__(self):
        pass
    # TODO :
    # [optimizer]
    # learning_rate = 0.001
    # momentum = 0.9
    # nepoch = 1000
    # batch_size = 16
    #
    # [model]
    # name = CONV1DBN
    # hidden_size = 16
    # dropout = 0.1
    # n_layers = 1
    # kernel_size = 8
    # pool_size = 4
    #
    # [loss]
    # weight1 = 1
    # weight2 = 1
    # weight3 = 1
    # weight4 = 1
    #
    # [path]
    # model = Models / baseline_final
    # tensorboard = Tensorboard / baseline_final
