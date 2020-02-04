import math
import numpy as np 


def accuracy(y_hat, y):

    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    assert(y_hat.size == y.size)
    correct = 0
    y_hat = list(y_hat)
    y = list(y)
    for i in range(len(y_hat)):
        if(y_hat[i] == y[i]):
            correct+=1
    return(float(correct/len(y)*100))
    # TODO: Write here

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    assert(y_hat.size==y.size)
    TP = 0
    TP_FP = 0
    y_hat = np.array(y_hat)
    y = np.array(y)
    for i in range(len(y_hat)):
        if(y_hat[i]==cls):
            if(y_hat[i]==y[i]):
                TP+=1
            TP_FP +=1
    if(TP_FP == 0):
        return 0 
    return TP/TP_FP


    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    assert(y_hat.size==y.size)
    TP = 0
    TP_TN = 0
    y_hat = np.array(y_hat)
    y = np.array(y)
    for i in range(len(y_hat)):
        if(y[i]==cls):
            if(y[i]==y_hat[i]):
                TP+=1
            TP_TN +=1
    if(TP_TN == 0):
        return 0 
    return TP/TP_TN
    pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    ans = 0
    for i in range(len(y_hat)):
        ans = ans + (y_hat[i] - y[i])**2
    ans = ans/len(y_hat)
    return(math.sqrt(ans))
    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    ans = 0
    y_hat = np.array(y_hat)
    y = np.array(y)
    for i in range(len(y_hat)):
        ans = ans + abs(y_hat[i] - y[i])
    return((ans)/len(y))
    pass
