import argparse
import numpy as np
import matplotlib.pyplot as plt

from base import BaseModel


parser = argparse.ArgumentParser(description="Run the ONLMSR algorithm in sequential mode")

parser.add_argument("--data-file", default='data/temp2.txt', type=str,
                    help="path of the data file")

parser.add_argument("--tuning-parameter", default=0.6, type=float,
                    help="The value of the tuning parameter.")

class ONLMSR(BaseModel):
    """This class implements Online Normalised Least Mean Square Regression.
    """

    def __init__(self, a, n):
        super(ONLMSR, self).__init__(a, n)

    def delta(self,x,y):
       l =  (y - (np.matrix(x)*np.matrix(self.w).T)) / (self.a + np.matrix(x) * np.matrix(x).T) 
       self.w = self.w + l*x
       return {'w': self.w}

    def predict(self, x):
       pred = np.matrix(x)*np.matrix(self.w).T
       return pred.item(0) 

if __name__ == "__main__":
    args = parser.parse_args()

    array = np.loadtxt(args.data_file, delimiter=' ')
    n = len(array[0][:-1])
    model = ONLMSR(args.tuning_parameter, n)
    for i, a in enumerate(array):
        x, y = a[:-1], a[-1]
        new_y = model.predict(x)
        print(new_y, y)
        if i != 0:
            plt.scatter(y, new_y)
        model.delta(x, y)
    plt.show()
