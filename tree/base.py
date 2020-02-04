"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain,best_split,split,info_gain_real_discrete,best_split_Real_Real

np.random.seed(42)
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.child = []

class DecisionTree():
    def __init__(self,sample_weight, criterion = "information_gain", max_depth = 1):
        self.criterion = criterion
        self.max_depth = 1
        self.root = None
        self.tree = None
        self.sample_weight = sample_weight
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        pass

    def fit(self, X, Y):
        self.tree = self.Real_Discrete(X,Y,self.max_depth,self.sample_weight) 
        # if(str(X.dtype) == "category" and str(Y.dtype)  == "category"):
        #     print(1)
        #     for crit in self.criterion:
        #         if(crit == "information_gain"):
        #             self.tree = self.ID3(X,Y,list(X.index),self.max_depth)
        #         elif(crit == "gini_index"):
        #             self.tree = self.ID3(X,Y,list(X.index),self.max_depth)
        # elif(str(X.dtype)  == "category" and str(Y.dtype)  != "category"):
        #     print(2)
        #     self.tree = self.ID3_discrete_real(X,Y,list(X.index),self.max_depth)  
        # elif(str(X.dtype) !="category" and str(Y.dtype)  == "category"):
        #     print(3)
            
        # elif(str(X.dtype)  != "category" and str(Y.dtype)  != "category"):
        #     print(4)
        #     self.tree = self.realreal(pd.core.frame.DataFrame(X),pd.core.series.Series(Y),list(X.index),self.max_depth)


        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """    
    def Real_Discrete(self,X,Y,depth,weights):
        weights = pd.DataFrame(weights)
        X[X.columns[-1] + 1] = Y
        X[[X.columns[-1] + 1]] = weights
        data = np.array(X)
        return self.build_real_Discrete(data,self.max_depth)


    def build_real_Discrete(self,data,depth):
        node = Node(None)
        if(len(data) == 0):
            return
        best_gain,threshold,index,left, right = best_split(data)
        node.index = index
        node.threshold = threshold
        if(depth == 0):
            nums = dict()
            for i in left:
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            for i in right:
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
            node.predicted_class = max_class
            return node
        
        if(best_gain == -99999):
            nums = dict()
            if(left!=None):
                for i in left:
                    if(i[-2] in nums):
                        nums[i[-2]]+=i[-1]
                    else:
                        nums[i[-2]] = i[-1]
            if(right!=None):
                for i in right:
                    if(i[-2] in nums):
                        nums[i[-2]]+=i[-1]
                    else:
                        nums[i[-2]] = i[-1]
            if(len(nums) == 0):
                return
            max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
            node.predicted_class = max_class
            return node
        if(best_gain == 0):
            nums = dict()
            for i in left:
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            for i in right:
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
            node.predicted_class = max_class
            return node
        if(len(left) ==0 or len(right) == 0):
            nums = dict()
            for i in left:
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            for i in right:
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
            node.predicted_class = max_class
            return node
        if(depth == 0):
            nums = dict()
            for i in left:
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            for i in right: 
                if(i[-2] in nums):
                    nums[i[-2]]+=i[-1]
                else:
                    nums[i[-2]] = i[-1]
            max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
            node.predicted_class = max_class
        node.left = self.build_real_Discrete(left,depth-1)
        node.right = self.build_real_Discrete(right,depth-1)
        return node
    # def ID3(self,x,y,row_index,depth):
    #     tree = dict()
    #     subset_x = x.iloc[row_index,:]
    #     subset_y = y.iloc[row_index]
    #     if(len(list(set(subset_y))) == 1):
    #         tree['terminal']=y[0]
    #         return(tree)
    #     if(depth == 0):
    #         nums = dict()
    #         for i in y:
    #             if(i in nums):
    #                 nums[i]+=1
    #             else:
    #                 nums[i] = 1
    #         max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
    #         tree['terminal']= max_class
    #         return(tree)
    #     max_info = -99999
    #     info_key = ""
    #     temparr1 = []
    #     temparr2 = []
    #     for i in subset_x:
    #         temparr1.append(information_gain(subset_y,subset_x[i]))
    #         temparr2.append(i)
    #     max_info = max(temparr1)
    #     info_key = temparr2[temparr1.index(max(temparr1))]
    #     if(info_key in row_index):
    #         if(len(row_index)>1):    
    #             row_index.remove(info_key)
    #     if(info_key!=""):
    #         tree[info_key] = dict()
    #         for i in set(subset_x[info_key]):
    #             tree[info_key][i]=self.ID3(x.drop(columns = info_key),y,list(subset_x[subset_x[info_key] == i].index),depth-1)
    #     return(tree)



    # def ID3_discrete_real(self,x,y,row_index,depth):
    #     tree = dict()
    #     subset_x = x.iloc[row_index,:]
    #     subset_y = y.iloc[row_index]
    #     if(len(list(set(subset_y))) == 1):
    #         tree['terminal']=list(set(subset_y))[0]
    #         return(tree)
    #     if(depth == 0):
    #         tree['terminal']=np.mean(subset_y)
    #         return(tree)
    #     max_info = -99999
    #     info_key = ""
    #     temparr1 = []
    #     temparr2 = []
    #     for i in subset_x:
    #         temparr1.append(STD(subset_y,subset_x[i]))
    #         temparr2.append(i)
    #     max_info = max(temparr1)
    #     info_key = temparr2[temparr1.index(max(temparr1))]
    #     if(info_key in row_index):
    #         if(len(row_index)>1):    
    #             row_index.remove(info_key)
    #     if(info_key!=""):
    #         tree[info_key] = dict()
    #         for i in set(subset_x[info_key]):
    #             tree[info_key][i]=self.ID3_discrete_real(x.drop(columns = info_key),y,list(subset_x[subset_x[info_key] == i].index),depth-1)
    #     return(tree)

    

    # def realreal(self,X,y,row_index,depth):
    #     subset_x = X.iloc[row_index,:]
    #     subset_y = y.iloc[row_index] 
    #     node = Node(None)
    #     if(len(subset_x) == 0 or len(subset_y) ==0):
    #         return
    #     if(depth == 0):
    #         node.predicted_class = np.mean(subset_y)
    #         return(node)
    #     if(len(list(set(subset_y))) == 1):
    #         node.predicted_class = np.array(subset_y)[0]
    #         return(node)
    #     max_info = -99999
    #     info_key = ""
    #     temparr1 = []
    #     temparr2 = []
    #     temparr3 = []
    #     for i in subset_x:
    #         p,q = best_split_Real_Real(np.array(subset_y),np.array(subset_x[i]))
    #         temparr1.append(p)
    #         temparr2.append(i)
    #         temparr3.append(q)
    #     if(len(temparr1)!=0):       
    #         max_info = max(temparr1)
    #         info_key = temparr2[temparr1.index(max(temparr1))]
    #         mean_opt = temparr3[temparr1.index(max(temparr1))]
    #         node.index = info_key  
    #         if(info_key in row_index):
    #             if(len(row_index)>1):    
    #                 row_index.remove(info_key)
    #         if(info_key!=""):
    #             max_info = max(temparr1)
    #             info_key = temparr2[temparr1.index(max(temparr1))]
    #             mean_opt = temparr3[temparr1.index(max(temparr1))]
    #             node.index = info_key  
    #             node.threshold = mean_opt
    #             node.right = self.realreal(X.drop(columns=info_key),y,list(subset_x[subset_x[info_key]>mean_opt].index),depth-1)
    #             node.left = self.realreal(X.drop(columns=info_key),y,list(subset_x[subset_x[info_key] <= mean_opt].index),depth-1)
    #             if(node.left == None):
    #                 node.predicted_class = np.mean(subset_y)
    #             if(node.right == None):
    #                 node.predicted_class = np.mean(subset_y)
    #             return(node)

    def predict(self, X):
        ans = []
        for i in range(len(X)):
            tree1 = self.tree
            while(tree1.predicted_class==None):
                if(X.iloc[i][tree1.index]> tree1.threshold):
                    tree1 = tree1.right
                else:
                    tree1 = tree1.left
            ans.append(tree1.predicted_class)
        return(ans)
        # if(str(X.dtype) != "category"):
        #     tree = self.tree
        #     if(type(tree) == type(dict())):
        #         return
        #     else:
                
        # else:
        #     tree = self.tree
        #     if(type(tree) == type(dict())):
        #         return
        #     else:
        #         ans = []
        #         for i in range(len(X)):
        #             tree1 = tree
        #             while(tree1.predicted_class==None):
        #                 if(X.iloc[i][tree1.index]>= tree1.threshold):
        #                     tree1 = tree1.right
        #                 else:
        #                     tree1 = tree1.left
        #             ans.append(tree1.predicted_class)
        #         return(ans)

        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

    def plot(self):
        if(type(self.tree) == type(dict())):
            print(self.tree)
        else:
            d = dict()
            d[0] = [self.tree.threshold,self.tree.index]
            self.recursive_plot(self.tree,1,d)
            for i in d.keys():
                print(d[i])
                print(" ")
            print(d.keys())
            

    def recursive_plot(self,tree,depth,dic):
        if(tree == None):
            return
        elif(tree.predicted_class != None):
            if(depth not in dic):
                dic[depth] = [[['leaf'],[tree.predicted_class]]]
            else:
                dic[depth].append([['leaf'],[tree.predicted_class]])
        else:
            if(depth not in dic):
                dic[depth] = [[[tree.left.threshold,tree.index],[tree.right.threshold,tree.index]]]
            else:
                dic[depth].append([[tree.left.threshold,tree.index],[tree.right.threshold,tree.index]])
        self.recursive_plot(tree.left,depth+1,dic)
        self.recursive_plot(tree.right,depth+1,dic)



        
        """
        Function to plot the tree
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass