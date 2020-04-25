# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:33:50 2020

@author: satrio
"""

#Import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Create Polynomial Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6) #degree menunjukan pangkat, semakin besar semakin presisi
X_poly = poly_reg.fit_transform(X)#Fungsi Polimial
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) 

#Predicted New Salary
# Predicting a new result with Polynomial Regression
predic_result  =lin_reg_2.predict(poly_reg.fit_transform([[7.5]]))
print(predic_result.astype(int))

#Visualization with Graph
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.scatter(7.5, predic_result.astype(int), color='green')
plt.legend(['Polynomial Model','Real Value','Predicted Value'], loc='upper left')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary (USD)')
plt.show()