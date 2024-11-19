import numpy as np


class BaseModel(object):
    """This class implements a base model
    """

    def __init__(self, min_, max_, eta, c, n,alpha):
        self.min_ = min_
        self.max_ = max_
        self.eta = eta
        self.c  = c
        self.n = n
        self.w = np.matrix(np.ones(self.n))
        self.g = 1.0
        self.alpha=alpha

    def get_params(self):
        """return current parameters
        """
        return {'w': self.w}
    
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