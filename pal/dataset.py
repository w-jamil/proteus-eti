import os
import glob
import numpy as np
from numpy import genfromtxt


class Dataset:
    """This class handles data loading
    """
    def __init__(self, file_path):
        # read the whole data file
        self.instances = np.loadtxt(file_path, delimiter=' ')

    def next(self):
        """Return next datapoint
        """
        if self.instances.any():
            x = self.instances[0]
            # shift to next element
            self.instances = self.instances[1:]
            return x[:-1], x[-1]
        else:
            return [], None 