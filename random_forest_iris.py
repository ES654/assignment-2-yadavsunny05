import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

###Write code here
from sklearn.datasets import  load_iris
data=pd.DataFrame(load_iris()['data'])
n=len(data)
y = pd.DataFrame(load_iris()["target"])
data[data.columns[-1] + 1] = y

data = np.array(data)

np.random.shuffle(data)

X_train = data[:int(len(data)*0.6),[1,3]]
X_test = data[int(len(data)*0.6):,[1,3]]
Y_train = data[:int(len(data)*0.6),4]
Y_test = data[int(len(data)*0.6):,4]

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
Y_train = pd.Series(Y_train)
Y_test = pd.Series(Y_test)


criteria = 'entropy'
Classifier_AB = RandomForestClassifier(10,criteria)
Classifier_AB.fit(X_train,Y_train)

y_hat = Classifier_AB.predict(X_test)
print(y_hat,Y_test)
Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, Y_test))
for cls in set(y_hat):
    print('Precision: ', precision(np.array(y_hat),np.array(Y_test), cls))
    print('Recall: ', recall(np.array(y_hat),  np.array(Y_test), cls))