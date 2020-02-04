from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import copy
import operator
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import graphviz
from sklearn.externals.six import StringIO
import matplotlib.image as mpimg
import sklearn.tree as sktree

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=3):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.trees = None
        self.features = None
        self.X = None
        self.data = [None for i in range(n_estimators)]
        self.Y = None
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        pass

    def fit(self, X, y):
        assert(X.shape[0]==y.size)
        self.Y = y
        self.X = X
        m = min(2, X.shape[1])
        tree = DecisionTreeClassifier(criterion = self.criterion,max_depth = self.max_depth)
        trearr = []
        features_comp = []
        for i in range(self.n_estimators):
            temptree = copy.deepcopy(tree)
            features = random.sample(list(X.columns),k=(X.shape[1]-m))
            temptree.fit(X.drop(features, axis=1),y)
            self.data[i] =  pd.concat([X.drop(features, axis=1),y],axis = 1,ignore_index=True)
            trearr.append(temptree)
            features_comp.append(features)
        self.trees = trearr
        self.features = features_comp
        return
        
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass

    def predict(self, X):
        trees = self.trees
        out = []
        for i in range(self.n_estimators):
            temptree =  trees[i]
            out.append(temptree.predict(X.drop(self.features[i], axis=1)))
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
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        fig1 = plt.figure(figsize=(self.n_estimators*7,5))
        for i in range(self.n_estimators):
            plt.subplot(1,self.n_estimators,i+1)
            temp = []
            for j in self.X.columns:
                temp.append(j)
            for j in temp:
                if(j in self.features[i]):
                    temp.remove(j)
            sktree.export_graphviz(self.trees[i], out_file='tree.dot', 
                class_names=[str(i) for i in self.Y.unique()],  
                filled=True, rounded=True, special_characters=True)
        plt.show() 

        h=0.02
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig2 = plt.figure(figsize=(self.n_estimators*5,2))

    
        fig2 = plt.figure(figsize=(6*self.n_estimators,7))
        for i in range(self.n_estimators):
            x_min, x_max = self.data[i].iloc[:, 0].min() - .5, self.data[i].iloc[:, 0].max() + .5
            y_min, y_max = self.data[i].iloc[:, 1].min() - .5, self.data[i].iloc[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
            if hasattr(self.trees[i], "decision_function"):
                Z = self.trees[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:   
                Z = self.trees[i].predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.data[i][0], self.data[i][1], c=self.data[i][2], cmap=cm_bright, edgecolors='k')
            plt.xlabel("Feature : " + str(self.data[i].columns[0]))
            plt.ylabel("Feature : " + str(self.data[i].columns[1]))
            plt.title("Dtree Classifier"+str(i+1))
        plt.show()



        fig3 = plt.figure(figsize=(8,7))

        x_min, x_max = self.X.iloc[:, 0].min() - .5, self.X.iloc[:, 0].max() + .5
        y_min, y_max = self.X.iloc[:, 1].min() - .5, self.X.iloc[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        d = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=self.X.columns)
        Z = self.predict(d)
        Z = np.array(Z).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        plt.scatter(self.X.iloc[:,0], self.X.iloc[:,1], c=self.Y, cmap=cm_bright, edgecolors='k')
        plt.xlabel(str(self.X.columns[0]))
        plt.ylabel(str(self.X.columns[1]))
        plt.title("The combined decision surface")
        plt.tight_layout()
        plt.show()
        pass



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.trees = None
        self.features = None
        self.X = None
        self.data = [None for i in range(n_estimators)]
        self.Y = None

        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        pass

    def fit(self, X, y):
        assert(X.shape[0]==y.size)
        self.Y = y
        self.X = X
        m = min(2, X.shape[1])
        tree = DecisionTreeRegressor(max_depth = self.max_depth)
        trearr = []
        features_comp = []
        for i in range(self.n_estimators):
            temptree = copy.deepcopy(tree)
            features = random.sample(list(X.columns),k=(X.shape[1]-m))
            temptree.fit(X.drop(features, axis=1),y)
            self.data[i] =  pd.concat([X.drop(features, axis=1),y],axis = 1,ignore_index=True)
            trearr.append(temptree)
            features_comp.append(features)
        self.trees = trearr
        self.features = features_comp
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass

    def predict(self, X):
        trees = self.trees
        out = []
        for i in range(self.n_estimators):
            temptree =  trees[i]
            out.append(temptree.predict(X.drop(self.features[i], axis=1)))
        ans = [0 for i in range(X.shape[0])]
        for i in range(X.shape[0]):
            for j in range(len(out)):
               ans[i] = ans[i] +out[j][i]
        ans = np.array(ans)
        return(ans/self.n_estimators)

        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        
        fig1 = plt.figure(figsize=(self.n_estimators*7,5))
        for i in range(self.n_estimators):
            plt.subplot(1,self.n_estimators,i+1)
            temp = []
            for j in self.X.columns:
                temp.append(j)
            for j in temp:
                if(j in self.features[i]):
                    temp.remove(j)
            sktree.export_graphviz(self.trees[i], out_file='tree.dot', 
                class_names=[str(i) for i in self.Y.unique()],  
                filled=True, rounded=True, special_characters=True)
        plt.show() 

        h=0.02
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig2 = plt.figure(figsize=(self.n_estimators*5,2))

    
        fig2 = plt.figure(figsize=(6*self.n_estimators,7))
        for i in range(self.n_estimators):
            x_min, x_max = self.data[i].iloc[:, 0].min() - .5, self.data[i].iloc[:, 0].max() + .5
            y_min, y_max = self.data[i].iloc[:, 1].min() - .5, self.data[i].iloc[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
            if hasattr(self.trees[i], "decision_function"):
                Z = self.trees[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:   
                Z = self.trees[i].predict(np.c_[xx.ravel(), yy.ravel()])[:]
            Z = Z.reshape(xx.shape)
            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.data[i][0], self.data[i][1], c=self.data[i][2], cmap=cm_bright, edgecolors='k')
            plt.xlabel("Feature : " + str(self.data[i].columns[0]))
            plt.ylabel("Feature : " + str(self.data[i].columns[1]))
            plt.title("Dtree Classifier"+str(i+1))
        plt.show()



        fig3 = plt.figure(figsize=(8,7))

        x_min, x_max = self.X.iloc[:, 0].min() - .5, self.X.iloc[:, 0].max() + .5
        y_min, y_max = self.X.iloc[:, 1].min() - .5, self.X.iloc[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        d = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=self.X.columns)
        Z = self.predict(d)
        Z = np.array(Z).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        plt.scatter(self.X.iloc[:,0], self.X.iloc[:,1], c=self.Y, cmap=cm_bright, edgecolors='k')
        plt.xlabel(str(self.X.columns[0]))
        plt.ylabel(str(self.X.columns[1]))
        plt.title("The combined decision surface")
        plt.tight_layout()
        plt.show()
        pass

