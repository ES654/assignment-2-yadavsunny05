"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
y = y.replace(0,-1)
Classifier_AB.fit(X, y)
Classifier_AB.classifier

y_hat = Classifier_AB.predict(X)
print(y_hat,y)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features



from sklearn.datasets import  load_iris
data=pd.DataFrame(load_iris()['data'])
n=len(data)
y = pd.DataFrame(load_iris()["target"])
data[data.columns[-1] + 1] = y

data = np.array(data)

np.random.shuffle(data)

data = data[:,[1,3,4]]
for i in range(len(data)):
    if(data[i][2] == 2):
        data[i][2] = 1
    else:
        data[i][2] = -1

X_train = data[:int(len(data)*0.6),[0,1]]
X_test = data[int(len(data)*0.6):,[0,1]]
Y_train = data[:int(len(data)*0.6),2]
Y_test = data[int(len(data)*0.6):,2]

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
Y_train = pd.Series(Y_train)
Y_test = pd.Series(Y_test)


criteria = 'information_gain'
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X_train,Y_train)
Classifier_AB.classifier

y_hat = Classifier_AB.predict(X_test)
print(y_hat,Y_test)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, Y_test))
for cls in set(y_hat):
    print('Precision: ', precision(np.array(y_hat),np.array(Y_test), cls))
    print('Recall: ', recall(np.array(y_hat),  np.array(Y_test), cls))
