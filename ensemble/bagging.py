import random as rd
from itertools import combinations
import copy
import pandas as pd
import operator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.classifiers = [copy.deepcopy(base_estimator) for i in range(n_estimators)]
        self.X = None
        self.Y = None
        self.data = [None for i in range(n_estimators)]

        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''

        pass

    def fit(self, X, y):
        assert(X.shape[0]==y.size)
        total = pd.concat([X,y],axis = 1,ignore_index=True)
        self.X = X
        self.Y = y
        for i in range(self.n_estimators):
            temp = total.sample(n=y.size, replace=True).reset_index(drop=True)
            self.data[i] = temp
            self.classifiers[i].fit(temp.iloc[:,:-1],temp.iloc[:,-1])

        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass

    def predict(self, X):
        trees = self.classifiers
        out = []
        for i in range(self.n_estimators):
            temptree =  trees[i]
            out.append(temptree.predict(X))
        ans = []
        for i in range(X.shape[0]):
            tempdict = dict()
            for j in out:
                if(j[i] not in tempdict):
                    tempdict[j[i]] = 1
                else:
                    tempdict[j[i]] += 1
            ans.append(max(tempdict.items(), key=operator.itemgetter(1))[0])
        return(ans)
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        The code is mostly from the above referece
        This function should return [fig1, fig2]

        """
        h=0.02
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig1 = plt.figure(figsize=(6*self.n_estimators,7))
        for i in range(self.n_estimators):
            plt.subplot(1,self.n_estimators,i+1)
            plt.scatter(self.data[i].iloc[:,0], self.data[i].iloc[:,1], c=self.data[i][2], cmap=cm_bright, edgecolors='k')
            plt.xlabel("Feature : " +str(self.data[i].columns[0]))
            plt.ylabel("Feature : "+str(self.data[i].columns[1]))
            plt.title("Bagging Round "+str(i+1))
        plt.show()

        fig2 = plt.figure(figsize=(6*self.n_estimators,7))

        for i in range(self.n_estimators):
            x_min, x_max = self.data[i].iloc[:, 0].min() - .5, self.data[i].iloc[:, 0].max() + .5
            y_min, y_max = self.data[i].iloc[:, 1].min() - .5, self.data[i].iloc[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
            if hasattr(self.classifiers[i], "decision_function"):
                Z = self.classifiers[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = self.classifiers[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.data[i].iloc[:,0], self.data[i].iloc[:,1], c=self.data[i].iloc[:,2], cmap=cm_bright, edgecolors='k')
            plt.xlabel("Feature : " + str(self.data[i].columns[0]))
            plt.ylabel("Feature : " + str(self.data[i].columns[1]))
            plt.title("Bagging Round "+str(i+1))
        plt.show()
        return [fig1, fig2]
        pass
