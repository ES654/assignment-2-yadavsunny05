
import math
import numpy as np
import pandas as pd

def entropy(Y):
    entro = 0.0
    sample_count = dict()
    for i in range(len(Y)):
        if(str(Y[i]) in sample_count):
            sample_count[str(Y[i])] +=Y[i][-1]
        else:
            sample_count[str(Y[i])] =Y[i][-1]
    for i in sample_count.keys():    
        prob = (sample_count[i]/np.sum(Y[:,-1]))
        entro -= (prob*math.log(prob,2))
    return entro        
    pass


def info_gain_real_discrete(left,right,current):
    return entropy(current) - entropy(left)*sum(left[:,-1])/(sum(left[:,-1])+sum(right[:,-1])) - entropy(right)*sum(right[:,-1])/(sum(left[:,-1])+sum(right[:,-1]))

def split(rows,threshold,col):
    left,right = [],[]
    for row in rows:
        if(row[col]>=threshold):
            right.append(row)
        else:
            left.append(row)
    if(len(left) == 0 or len(right) == 0):
        return None,None
    return left,right

def best_split(rows):
    best_gain=-99999
    threshold= -99999
    index = 0
    best_left = None
    best_right = None
    current=entropy(rows[:])
    for col in range(len(rows[0])-2):
        thresholds=set([row[col] for row in rows])
        for i in thresholds: 
            left,right=split(rows,i,col)
            if((left!=None)and(right!=None)):
                left = pd.DataFrame(left)
                right = pd.DataFrame(right)
                left = np.array(left)
                right = np.array(right)
                temp=info_gain_real_discrete(left,right,rows)
                if temp>=best_gain: 
                    best_left = np.array(left)
                    best_right  = np.array(right)
                    best_gain = temp
                    threshold=i
                    index = col
    return best_gain,threshold,index,best_left,best_right



def best_split_Real_Real(Y,attr):
    splitarr = []
    for i in range(len(attr)):
        splitarr.append([Y[i],attr[i]])
    splitarr = np.array(sorted(splitarr,key = lambda x:x[0]))
    min_info=999999
    split_value=0
    for i in range(1,len(Y)):
        temp = np.var(splitarr[:i,1]) + np.var(splitarr[i:,1])
        if(min_info>temp):
            min_info = temp
            split_value=(splitarr[i,1]+splitarr[i-1,1])/2 
    return((min_info,split_value))


def information_gain(Y, attr,weight):
    initial_gain = entropy(Y)
    y_val = dict()
    indexes=list(Y.index)
    for index in indexes:
            y_val[attr[index]]=[]
    for index in indexes:
        y_val[attr[index]].append(Y[index])
    for i in y_val:
        initial_gain-=entropy(y_val[i])*(len(y_val[i])/len(list(Y.index)))
    return(initial_gain)

