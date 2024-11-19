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
class WAA(BaseModel):
    """This class implements Weighted Average Algorithm. 
    """

    def __init__(self, min_,max_, eta,c,n):
        super(WAA, self).__init__(min_,max_,eta,c, n)

    def delta(self, x, y):
        self.w = self.w/self.w.sum()
        self.w = np.multiply(self.w, np.exp(-1.0/self.eta * (x-y)**2)) 
        return {'w': self.w}
            

    def predict(self, x):
        pred = np.dot(self.w,x)
        return pred.item(0)

