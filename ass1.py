import os
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras
#read the csv file from dataset
train= pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#describing the data 
train.describe()

#dispalying the data which is the head of dataset
train.head()
#splitting the data for cross validation
train,test=train_test_split(train,test_size=0.25)
#displaying the head of test dataset
test.head()

rn = range(1,26)
kf5 = KFold(n_splits=5, shuffle=False)
kf3 = KFold(n_splits=3, shuffle=False)



for train_index, test_index in kf3.split(rn):
    print(train_index, test_index)
test_x = test.values

#training examples here we have 
shape_x = train_x.shape
shape_y = train_y.shape

m = train_y.shape[0]

print ('The shape of X is: ' + str(shape_x))
print ('The shape of Y is: ' + str(shape_y))
print ('I have m = %d training examples!' % (m))


#applying naive bayes
X, y = train_x(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
...       % (X_test.shape[0], (y_test != y_pred).sum()))
