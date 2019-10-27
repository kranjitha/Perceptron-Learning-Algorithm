import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math

class Perceptron:

    def __init__(self,weights=None,errors=None,step=None):
        self.weights=weights
        self.errors=errors
        self.step=step


    def create_data(self,m,k,epsilon):
        """
        Function to create data set
        :param m: The size of data set
        :param k: Number of features
        :param epsilon: The factor that can control the margin
        :return: Pandas Data Frame (m x k)

        """
        lst1=[]
        if self.step is None:
            self.step=0
        if self.weights is None:
            self.weights=np.zeros(k+1)
        if self.errors is None:
            self.errors=np.zeros(m)
            self.errors+=-1
        # for i in range(0,k+1):
        #     self.weights.append(0)
        data=[] # list of lists to hold data rows
        column_names=[] # column names
        for i in range(0,k+1):
            column_names.append("X"+str(i))
        column_names.append("Y")

        for i in range(m):
            lst=[]# list to hold row information
            lst.append(1)

            lst=lst+(list(np.random.normal(0,1,k-1)))
            d=np.random.exponential(1)
            lst.append(np.random.choice([epsilon+d,-epsilon-d],p=[0.5,0.5]))
            data.append(lst)
            #X=np.matrix(data)
            if lst[-1]>0:
                lst1.append(1)
            else:
                lst1.append(-1)

        return (data,np.array(lst1))

    def predict(self,X):

        predictions=[]
        for i in range(len(X)):
            activation = 0
            for j in range(1,len(X[0])):
                activation+=self.weights[j]*X[i][j]
            activation+=self.weights[0]
            if activation>=0:
                predictions.append(1)
            else:
                predictions.append(-1)
        return np.array(predictions)


    def accuracy(self,y,pred_y):


        if np.array_equal(y,pred_y):

            return 1
        else:
            return -1


    def fit(self, X, y):

        predictions=self.predict(X)# returns a numpy array with predictions
        self.errors=predictions-y
        if self.accuracy(y,predictions)==1: # checking if all the rows are properly classified
            return
        else:
            for i in range(len(X)):
                if predictions[i]!=y[i]:
                    self.step = self.step + 1
                    temp=y[i]
                    self.weights[0]=self.weights[0]+temp
                    for j in range(1,len(X[0])):
                        self.weights[j]=self.weights[j]+temp*X[i][j]
                    self.fit(X,y)
                    break





    #def fit(self,X,y):
p=np.zeros(5)
dict_steps={}
dict_100={}
dict_1000={}
for i in tqdm(range(2,40)):
    steps=0
    for j in range(100):
        model=Perceptron()
        data=model.create_data(100,i,1)
        model.fit(data[0],data[1])
        steps+=model.step
    dict_100[i]=steps/100
    print(dict_100)
for i in tqdm(range(2,40)):
    steps=0
    for j in range(100):
        model=Perceptron()
        data=model.create_data(1000,i,1)
        model.fit(data[0],data[1])
        steps+=model.step
    dict_1000[i]=steps/100
    print(dict_1000)
plt.ylim(0,140)
plt.plot(dict_100.keys(),dict_100.values(),label="m=100")
plt.plot(dict_1000.keys(),dict_1000.values(),label="m=1000")
plt.xlabel("The number of features")
plt.ylabel("Average number of steps")
plt.legend()
plt.show()






#model.fit(data[0],data[1])
