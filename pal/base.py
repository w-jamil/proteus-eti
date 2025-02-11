import pandas as pd
import numpy as np


class BaseModel(object):
    """This class implements a base model
    """

    def __init__(self, a,n):
        self.n = n
        self.a = a 
        self.w = np.matrix(np.ones(self.n))


    def get_params(self):
        """return current parameters
        """
        return {
            'w': self.w
            }
    
    def apply_updates(self, updates):
        """apply new param updates to the current model
        """
        if 'w' in updates:
            self.w = np.matrix(updates['w'])
    
    @property
    def delta(self, x, y):
        """calculate model updates
        """
        raise NotImplementedError

    @property
    def predict(self, x):
        """make prediction
        """
        raise NotImplementedError