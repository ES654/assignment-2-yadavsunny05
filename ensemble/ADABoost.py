from tree.utils import *
from tree.base import *
import random as rd
from itertools import combinations
import copy
import pandas as pd
import operator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3,depth = 3): # Optional Arguments: Type of estimator
        
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.alpha = None
        self.classifier = None
        self.depth = depth
        self.X = None
        self.Y = None
        self.data = [None for i in range(n_estimators)]

        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        pass
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        alphas = []
        X_columns = list(X.columns)
        m = self.n_estimators
        weights = np.ones(len(Y))/len(Y)
        temp = DecisionTree(max_depth = self.depth,sample_weight=None)
        classifier = [copy.deepcopy(temp) for i in range(self.n_estimators)]
        for i in range(m):
            classifier[i].sample_weight = weights
            classifier[i].fit(X[X_columns],Y)
            pred = np.array(classifier[i].predict(X))
            self.data[i] = pd.concat([X[X_columns],pd.Series(pred)],axis = 1,ignore_index=True)
            y_true = np.array(Y)
            error = np.sum(weights[y_true!=pred])
            print(error)
            alph = 0.5*np.log((1-error)/error)
            alphas.append(alph)
            weights[y_true == pred] = weights[y_true == pred]*np.exp(-weights[y_true == pred])
            weights[y_true != pred] = weights[y_true != pred]*np.exp(weights[y_true != pred])
            weights = weights/sum(weights)
        self.alpha = alphas
        self.classifier = classifier
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass

    def predict(self, X):
        ans = np.array(self.classifier[0].predict(X))
        ans = ans*self.alpha[0]
        for i in range(1,self.n_estimators):
            ans += (np.array(self.classifier[i].predict(X)))*self.alpha[i]
        ans = np.sign(ans)
        return ans

        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

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
            print(x_max,x_min,y_max,y_min)
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
            if hasattr(self.classifier[i], "decision_function"):
                Z = self.classifier[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = self.classifier[i].predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=self.X.columns))
            Z = np.array(Z).reshape(xx.shape)
            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.data[i].iloc[:,0], self.data[i].iloc[:,1], c=self.data[i].iloc[:,2], cmap=cm_bright, edgecolors='k')
            plt.xlabel("Feature : " + str(self.data[i].columns[0]))
            plt.ylabel("Feature : " + str(self.data[i].columns[1]))
            plt.title("Bagging Round "+str(i+1))
        plt.show()
        return [fig1, fig2]
        pass
