import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        self.fit_intercept  = fit_intercept
        self.method = method
        self.theta = None
        self.y = None
        self.X = None
        pass

    def fit(self, X, y):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''
        if(self.method == 'normal'):
            if(self.fit_intercept):
                X = np.concatenate([np.ones((len(y), 1)), X], axis=1) 
            self.theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        else:
            self.theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        self.y = y
        self.X = X
        return

        pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if(self.fit_intercept):
            X  = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1) 
        return(X.dot(self.theta))
        pass

    def plot_residuals(self):
        pred = self.predict(self.X)
        fig = plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.scatter(x=[i for i in range(len(self.y))], y=self.y, label="Y_Given")
        plt.scatter(x=[i for i in range(len(self.y))], y=pred, label="Y_Predicted")
        plt.xlabel("Sample No.")
        plt.ylabel("Give Value vs Predicted Value")

        plt.subplot(1,3,2)
        sns.kdeplot(self.y - pred)
        plt.ylabel("KDE plot of the residuals")
        plt.title("Mean " + str(np.mean(self.y - pred)) + "Variance " +  str(np.var(self.y - pred)))
        self.residuals = self.y - pred

        plt.subplot(1,3,3)
        plt.bar([ i for i in range(len(self.theta))],self.theta)
        plt.yscale("log")
        plt.ylabel("Coefficients")
        plt.xlabel("Features")
        plt.title("Coefficients corresponding to each features")
        plt.show()
        return plt
        pass
        """
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.
        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(\hat{y})
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (\theta_i)
        """
        
