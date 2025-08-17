import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from base import BaseModel

parser = argparse.ArgumentParser(description="Run the COIRR algorithm in sequential mode")

parser.add_argument("--data-file", default='data/temp2.txt', type=str,
                    help="path of the data file")

parser.add_argument("--tuning-parameter", default=0.6, type=float,
                    help="The value of the tuning parameter.")


class COIRR(BaseModel):
    """This class implements Competitive Online Regression. 
    """

    def __init__(self, a, n):
        super(COIRR, self).__init__(a, n)

    def delta(self, x, y):
        self.b = np.array(self.b) + (y*x)
        self.w = self.b * self.InvA 
        return {
            'A': self.A,
            'b': self.b,
            'w': self.w,
            'InvA': self.InvA}

    def predict(self, x):
        at = np.matrix(x).T * np.matrix(x)
        Dt = np.diag(np.abs(self.w.A1))
        self.A = self.A + at 
        AAt = np.linalg.inv(np.matrix(self.a*np.array(np.eye(self.n)))+np.matrix(np.sqrt(Dt))*np.matrix(self.A)*np.matrix(np.sqrt(Dt)))
        self.InvA = np.sqrt(Dt)*AAt*np.sqrt(Dt)
        self.w = self.b * self.InvA 
        pred = np.matrix(x)*self.w.T
        return pred.item(0)

    
if __name__ == "__main__":
    args = parser.parse_args()

    lst = pd.DataFrame()

    array = np.loadtxt(args.data_file, delimiter='\t')
    n = len(array[0][:-1])
    model = COIRR(args.tuning_parameter, n)
    for i, a in enumerate(array):
        x, y = a[:-1], a[-1]
        new_y = model.predict(x)
        print(new_y, y)
        if i != 0:
            plt.scatter(y, new_y)
        model.delta(x, y)
        a = pd.concat([pd.Series(y), pd.Series(new_y)],axis=1)
        lst = pd.concat([lst,a],axis=0)

    plt.show()

