# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:26:27 2017

@author: Arshid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training=pd.read_csv('C:/Users/Arshid/Desktop/random-linear-regression/train.csv')
testing=pd.read_csv('C:/Users/Arshid/Desktop/random-linear-regression/test.csv')

training.drop(training.index[[213]], inplace=True)

#train=training.iloc[:,:].values
#test=testing.iloc[:,:].values

from sklearn.preprocessing import Imputer
imputer1=Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer1=imputer1.fit(training)
training=imputer1.transform(training)

x_train=training[:,:-1]
x_test=testing.iloc[:,:-1]
y_train=training[:,1]
y_test=testing.iloc[:,1]

"""
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.transform(x_test)



imputer2=Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer2=Imputer.fit(testing.values[:,:])
testing.values[:,:]=imputer2.transform(testing.values[:,:])
"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#y_pred=regressor.predict(x_test)


# PLotting the training set
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.xlabel('X_Plane')
plt.ylabel('Y_label')
plt.show()

# Plitting the test set

plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.xlabel('X_Plane')
plt.ylabel('Y_label')
plt.show()

