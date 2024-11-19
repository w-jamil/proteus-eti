import numpy as np
import argparse
import matplotlib.pyplot as plt

from base import BaseModel


parser = argparse.ArgumentParser(description="Run the OSLOG algorithm in sequential mode")

parser.add_argument("--data-file", default='data/temp1.txt', type=str,
                    help="path of the data file")

parser.add_argument("--tuning-parameter", default=0.6, type=float,
                    help="The value of the tuning parameter.")


class OSLOG(BaseModel):
    """This class implements Online Shrinkage via Limit of Gibbs Sampler.
    """

    def __init__(self, a, n):
        super(OSLOG, self).__init__(a, n)

    def predict(self, x):
        pred = np.matrix(x)*self.w.T 
        Dt = np.diag(np.abs(self.w.A1))
        at = np.matrix(x).T * np.matrix(x)
        self.A = self.A + at 
        AAt =  np.linalg.inv(np.matrix(self.a*np.array(np.eye(self.n)))+np.matrix(np.sqrt(Dt))*np.matrix(self.A)*np.matrix(np.sqrt(Dt)))
        self.InvA = np.sqrt(Dt)*AAt*np.sqrt(Dt)
        return pred.item(0)

    def delta(self, x, y):
        self.b = np.array(self.b) + (y*x)
        self.w = self.b * self.InvA 
        return {
            'A': self.A,
            'b': self.b,
            'w': self.w,
            'InvA': self.InvA}


if __name__ == "__main__":
    args = parser.parse_args()

    array = np.loadtxt(args.data_file, delimiter=' ')
    n = len(array[0][:-1])
    model = OSLOG(args.tuning_parameter, n)
    for i, a in enumerate(array):
        x, y = a[:-1], a[-1]
        new_y = model.predict(x)
        print(new_y, y)
        if i != 0:
            plt.scatter(y, new_y)
        model.delta(x, y)
    plt.show()

