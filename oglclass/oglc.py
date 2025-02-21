import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from base import BaseModel
from sklearn.metrics import classification_report
# from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


parser = argparse.ArgumentParser(description="Run the PAL algorithm in sequential mode")

parser.add_argument("--data-file", default='data/Phishing_smartphone.txt', type=str,
                    help="path of the data file")

parser.add_argument("--tuning-parameter", default=0.00001, type=float,
                    help="The value of the tuning parameter.")

class OGL(BaseModel):
    """This class implements Online PAL with no tuning parameter. 
    """

    def __init__(self, a,n):
        super(OGL, self).__init__(a,n)

    def delta(self, x, y):
        self.w = self.w + (1-y*np.sign(self.w.dot(x)))/(x.dot(x)**.5+self.a)*y*x
        return {'w': self.w}

    def predict(self, x):
        pred = np.sign(self.w.dot(x))
        return pred.item(0)

    
if __name__ == "__main__":
    args = parser.parse_args()

    array = np.loadtxt(args.data_file, delimiter='\t')
    n = len(array[0][:-1])
    f1_score = []
    model = OGL(args.tuning_parameter,n)
    y_pred = np.ones(len(array))
    y_vec = np.ones(len(array))
    for i, a in enumerate(array):
        x, y = a[:-1], a[-1]
        new_y = model.predict(x)
        y_pred[i] = new_y
        y_vec[i] = y
        model.delta(x, y)
        print(i)
        # if i>502:
        #     report = classification_report(pd.Series(y_vec).iloc[1:i].astype(float),pd.Series(y_pred).iloc[1:i].astype(float), output_dict=True)
        #     categories = list(report.keys())[:-3]
        #     f1_score.append([report[category]['f1-score'] for category in categories])

    # loss = pd.DataFrame(f1_score)
    # loss = loss.dropna(axis=0).reset_index(drop=True)
    # loss.columns = ["Benign","Malicious"]
    # plt.plot(loss)
    # plt.show()
    print(classification_report(y_vec[1:].astype(float),pd.Series(y_pred[1:]), labels=[-1, 1]))

    report = classification_report(y_vec[1:].astype(float),pd.Series(y_pred[1:]), output_dict=True)
    # Categories for the classification
    # categories = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'

    # # Extracting precision, recall, and f1-score
    # precision = [report[category]['precision'] for category in categories]
    # recall = [report[category]['recall'] for category in categories]
    # f1_score = [report[category]['f1-score'] for category in categories]

    # # Setting the positions and width for the bars
    # pos = list(range(len(categories)))
    # width = 0.25

    # # Setting the positions and width for the bars
    # pos = list(range(len(categories)))
    # width = 0.25

    # # Plotting each metric
    # plt.bar(pos, precision, width, alpha=0.5, color='red', label='Precision')
    # plt.bar([p + width for p in pos], recall, width, alpha=0.5, color='blue', label='Recall')
    # plt.bar([p + width*2 for p in pos], f1_score, width, alpha=0.5, color='green', label='F1-Score')

    # # Adding the aesthetics
    # plt.xlabel('Category')
    # plt.ylabel('Score')
    # plt.title('Classification Report')
    # plt.xticks([p + width for p in pos], categories)
    # plt.xticks(pos, ["Benign","Malicious"])
    # plt.legend(['Precision', 'Recall', 'F1-Score'], loc='upper left')
    # plt.grid()
    # plt.show()

    # cm = confusion_matrix(y_vec[1:].astype(float),pd.Series(y_pred[1:]),labels=[-1,1])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Benign","Malicious"])

    # print(cm)

    # disp.plot()

# import matplotlib.pyplot as plt 
  
# # lets plot decision boundary for this 
# w = model.coef_[0] 
# b = model.intercept_[0]  
# x = np.linspace(1, 4) 
# y = -(w[0] / w[1]) * x - b / w[1] 
# plt.plot(x, y, 'k-') 
  
# # plot data points 
# plt.scatter(X[:, 0], X[:, 1], c=Y) 
# plt.xlabel('Feature 1') 
# plt.ylabel('Feature 2') 
# plt.show()



# # importing libraries
# import numpy as np
# import time
# import matplotlib.pyplot as plt
 
# # creating initial data values
# # of x and y
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
 
# # to run GUI event loop
# plt.ion()
 
# # here we are creating sub plots
# figure, ax = plt.subplots(figsize=(10, 8))
# line1, = ax.plot(x, y)
 
# # setting title
# plt.title("Geeks For Geeks", fontsize=20)
 
# # setting x-axis label and y-axis label
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
 
# # Loop
# for _ in range(50):
#     # creating new Y values
#     new_y = np.sin(x-0.5*_)
 
#     # updating data values
#     line1.set_xdata(x)
#     line1.set_ydata(new_y)
 
#     # drawing updated values
#     figure.canvas.draw()
 
#     # This will run the GUI event
#     # loop until all UI events
#     # currently waiting have been processed
#     figure.canvas.flush_events()
 
#     time.sleep(0.1)