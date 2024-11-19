import numpy as np


class BaseModel(object):
    """This class implements a base model for regression algorithms
    """

    def __init__(self, a, n):
        self.a = a
        self.n = n
        self.A = np.zeros((n, n))
        self.b = np.zeros(self.n)
        self.w = np.matrix(np.ones(self.n))
        self.InvA = np.zeros((n, n))

    def get_params(self):
        """return current parameters
        """
        return {
            'A': self.A,
            'b': self.b,
            'w': self.w,
            'InvA': self.InvA}
    
    def apply_updates(self, updates):
        """apply new param updates to the current model
        """
        if 'A' in updates:
            self.A = updates['A']
        if 'b' in updates:
            self.b = updates['b']
        if 'w' in updates:
            self.w = np.matrix(updates['w'])
        if 'InvA' in updates:
            self.InvA = updates['InvA']
    
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