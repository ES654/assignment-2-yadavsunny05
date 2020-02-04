import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from sklearn.model_selection import train_test_split



data = pd.read_csv(r'C:\Users\Anshuman Yadav\Documents\Real.csv')
X_train,X_test,Y_train,Y_test = train_test_split(data[data.columns[1:-1]],data[data.columns[-1]])
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

y =  data[data.columns[-1]]

data = data[data.columns[:-1]]

weights = []
train_mae = []
test_mae = []

def k_fold(data,y, k =5):
    for i in range(5):
        val = len(data)//k
        x_test = data[val*i:val*(i+1)]
        x_train = np.append(data[0:val*i],data[val*(i+1):],axis = 0)
        y_test = y[val*i:val*(i+1)]
        y_train = np.append(y[0:val*i],y[val*(i+1):],axis = 0)
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        LR = LinearRegression(fit_intercept=True)
        LR.fit(x_train,y_train)
        temp = []
        for i in LR.theta:
            temp.append(i[0])
        weights.append(temp)
        train_mae.append(mae(LR.predict(x_train),y_train)[0])
        test_mae.append(mae(LR.predict(x_test),y_test)[0])


k_fold(data,y,5)

print(weights)
print(train_mae)
print(test_mae)

fig = plt.figure(figsize=(20,10))


for i in range(5):
    plt.subplot(2,3,i+1)
    plt.bar(list(i for i in range(len(weights[0]))), weights[i])
    plt.ylabel("Value of co-effecients in log scale")
    plt.yscale("log")
    plt.xlabel("coefficients")
    plt.title(str(i + 1)  + " Fold coefficients vs value")
plt.show()


fig2 = plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.bar(x = [1,2,3,4,5],height=train_mae)
plt.xlabel("Folds")
plt.ylabel("mae")
plt.title(" Folds vs train mae")
plt.subplot(1,2,2)
plt.bar(x = [1,2,3,4,5] ,height=test_mae)
plt.title("Folds vs test mae")
plt.xlabel("Folds")
plt.ylabel("mae")

plt.show()




