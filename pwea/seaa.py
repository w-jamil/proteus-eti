import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from base import BaseModel

parser = argparse.ArgumentParser(description="Run the AA algorithm in sequential mode")

parser.add_argument("--data-file", default='data/gas.txt', type=str,
                    help="path of the data file")

parser.add_argument("--min-val", default=-0.044337620177300535, type=float,
                    help="The min value of the label.")

parser.add_argument("--max-val", default=0.03995909036217061, type=float,
                    help="The max value of the label.")


parser.add_argument("--tuning-parameter", default=281.4548522087015, type=float,
                    help="The value of the tuning parameter.")

parser.add_argument("--switch-rate", default=0.0014, type=float,
                    help="The rate of switching.")


parser.add_argument("--a-a", default=0.05, type=float,
                    help="The rate of switching.")


class SEAA(BaseModel):
    """This class implements Aggregation Algorithm. 
    """

    def __init__(self, min_,max_, eta,c,n,alpha):
        super(SEAA, self).__init__(min_,max_,eta,c, n,alpha)

    def delta(self, x, y):
        self.w = self.w/self.w.sum()
        self.w = ((1.0-self.alpha) * self.w) + ((self.alpha/(n-1.0)) * (1.0-self.w))
        self.w = np.multiply(self.w,np.exp(-1.0/self.eta * (x-y)**2))    
        return {'w': self.w}
                

    def predict(self, x):
        gmin = -(1/self.eta) * np.log(np.dot(self.w, (np.exp(-self.eta * (x - self.min_)**2))))
        gmax = -(1/self.eta) * np.log(np.dot(self.w, (np.exp(-self.eta * (x - self.max_)**2))))
        pred = (0.5 * (self.min_ + self.max_)) - (gmax - gmin)/(2 *(self.max_ - self.min_))
        return pred.item(0)

if __name__ == "__main__":
    args = parser.parse_args()

    array = np.loadtxt(args.data_file, delimiter=' ')
    n = len(array[0][:-1])
    model = SEAA(args.min_val,args.max_val,args.tuning_parameter,args.switch_rate, n,args.a_a)
    for i, a in enumerate(array):
        x, y = a[:-1], a[-1]
        new_y = model.predict(x)
        print(new_y, y)
        if i != 0:
            plt.scatter(y, new_y)
        model.delta(x, y)
    plt.show()

