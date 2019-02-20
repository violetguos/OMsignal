import numpy as np
import random

class CircShift(object):
    """Shift the sample for the desired amount.

    Args:
        shift (int): Desired shift. A positive value shifts the sample forward in time.
    """

    def __init__(self, shift):
        assert isinstance(shift, int), "Incorrect shift type. Shift must be an integer."
        self.shift = shift
        
    def __call__(self, sample):
        return np.roll(sample, self.shift)

    def reroll(self, new_shift):
        """Replace the existing shift value of CircShift

        Args:
            new_shift (int): New desired shift.
        """
        assert isinstance(new_shift, int), "Incorrect shift type. new_shift must be an integer."
        self.shift = new_shift

class RandomCircShift(object):
    """Randomly shift a sample between -125 and 125 positions (+/- 1 second).
    
    Args:
        prob: Probability of shift, between 0 and 1.
    """
    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):
        if random.uniform(0,1) < self.p:
            sample = np.roll(sample, random.randint(-125,125),axis=1)
        return sample


class Negate(object):
    """Negate the values of the signal.
    """

    def __call__(self, sample):
        return np.negative(sample)

class RandomNegate(object):
    """Randomly negate the values of the signal.

    Args:
        prob: Probability of Negate, between 0 and 1.
    """
    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):
        if random.uniform(0,1) < self.p:
            sample = np.negative(sample)
        return sample


class ReplaceNoise(object):
    """Replace a sub-window with controlled noise matching mean and variance.

    Args:
        window_start (int): Position of the start of the replaced window in the sample.
        window_width (int): Width of the replaced window. Suggested between 0 and 250.
    """

    def __init__(self, window_start, window_width):
        assert isinstance(window_start, int), "Incorrect start type. window_start must be an integer."
        assert isinstance(window_width, int), "Incorrect width type. window_width must be an integer."
        self.start = window_start
        self.width = window_width

    def __call__(self, sample):
        if (self.start + self.width) > len(sample):
            raise ValueError('The window exceeds the sample.')
        else:
            mean = np.mean(sample[self.start:self.start+self.width])
            std = np.nanstd(sample[self.start:self.start+self.width])

            noise = np.random.normal(mean, std, self.width)
            sample[self.start:self.start+self.width] = noise

        return sample
    
    def reroll(self, new_start, new_window):
        """Replace the existing window start and width values of ReplaceNoise

        Args:
            new_start (int): Position of the start of the replaced window in the sample.
            new_window (int): Width of the replaced window. Suggested between 0 and 250.
        """
        assert isinstance(new_start, int), "Incorrect start type. new_start must be an integer."
        assert isinstance(new_window, int), "Incorrect width type. new_window must be an integer."
        self.start = new_start
        self.width = new_window

class RandomReplaceNoise(object):
    """Replace a random sub-window with controlled noise matching mean and variance.

    Args:
        prob: Probability of Negate, between 0 and 1.
    """

    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):
        if random.uniform(0,1) < self.p:
            start = random.randint(0,sample.shape[1]-250)
            width = random.randint(1,250)
            mean = np.nanmean(sample[0,start:start+width])
            std = np.nanstd(sample[0,start:start+width])
            noise = np.random.normal(mean, std, width)
            sample[0,start:start+width] = noise
        return sample

class DropoutBurst(object):
    """Replace the signal in a 48ms window around a time instants by 0, simulating periods of weak signal.
    As suggested in https://arxiv.org/pdf/1710.06122.pdf

    Args:
        instants: Middle position of the dropout window
    """

    def __init__(self, instants):
        assert isinstance(instants, int), "Incorrect instants type. instants must be an integer."
        if instants < 3:
            instants = 3
        elif instants > 3746: #magic numbers!!
            instants = 3746
        
        self.instants = instants

    def __call__(self, sample):
        sample[1, self.instants-3:self.instants+3] = 0
        return sample

class RandomDropoutBurst(object):
    """Replace the signal in a random 48ms window by 0, simulating periods of weak signal.
    As suggested in https://arxiv.org/pdf/1710.06122.pdf

    Args:
        prob: Probability of Dropout, between 0 and 1.
    """

    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):
        if random.uniform(0,1) < self.p:
            instants = random.randint(3, sample.shape[1]-4)
            sample[0,instants-3:instants+3] = 0
        return sample
