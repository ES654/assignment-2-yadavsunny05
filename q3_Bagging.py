"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# Or use sklearn decision tree
from linearRegression.linearRegression import LinearRegression

########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")


tree = DecisionTreeClassifier(criterion="entropy")
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


y_list = list()
data = {'Feature 1':list(), 'Feature 2': list()}
for i in range(1,9):
    for j in range(1,9):
        data['Feature 1'].append(i)
        data['Feature 2'].append(j)
        if(i==5 and j==8):
            y_list.append(1)
        elif(i==3 and j==3):
            y_list.append(0)
        elif(i<=5 and j<=5):
            y_list.append(1)
        else:
            y_list.append(0)
X = pd.DataFrame(data=data)
y = pd.Series(y_list, dtype="category")


criteria = 'entropy'
tree = DecisionTreeClassifier(criterion=criteria)
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=5 )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot()
print('Criteria :', criteria)
print('Train Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))