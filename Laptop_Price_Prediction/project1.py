# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 02:23:42 2018

@author: User
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn import metrics


data = pd.read_csv('lp_p.csv')
y=data['price'].values
data.drop('price',axis=1,inplace=True)
data.drop('audio_speakers',axis=1,inplace=True)
data.drop('optical_drive',axis=1,inplace=True)

x= data.values

lr = LinearRegression()
rfr= RandomForestRegressor()
abr=AdaBoostRegressor()
dtr=DecisionTreeRegressor()
sgd = linear_model.SGDRegressor()


score_lr = []
score_rfr = []
score_abr = []
score_dtr = []
score_sgd = []


kf = KFold(n_splits = 5,random_state = 0,shuffle = True)
for train_index,test_index in kf.split(x):
    x_train, x_test= x[train_index], x[test_index]
    y_train, y_test= y[train_index], y[test_index]
    
    scaler = preprocessing.StandardScaler().fit(x_train.astype(float))
    
    x_train = scaler.transform(x_train.astype(float))
    x_test = scaler.transform(x_test.astype(float))
    
    
    lr.fit(x_train, y_train)
    rfr.fit(x_train, y_train)
    abr.fit(x_train, y_train)
    dtr.fit(x_train, y_train)
    sgd.fit(x_train, y_train)
    4
    y_pred_lr = lr.predict(x_test)
    y_pred_rfr = rfr.predict(x_test)
    y_pred_abr = abr.predict(x_test)
    y_pred_dtr = dtr.predict(x_test)
    y_pred_sgd = sgd.predict(x_test)
    
    score_lr = round(r2_score(y_test, y_pred_lr),5)
    score_rfr = round(r2_score(y_test, y_pred_rfr),5)
    score_abr = round(r2_score(y_test, y_pred_abr),5)
    score_dtr = round(r2_score(y_test, y_pred_dtr),5)
    score_sgd = round(r2_score(y_test, y_pred_sgd),5)
    
    

print("\n\nLinear Regression  ")
    
print(" Avg R^2 Score :",np.mean(score_lr))
print('\n\n')

print("Random Forest  Regression")
    
print(" Avg R^2 Score :",np.mean(score_rfr))
print('\n\n')

print("AdaBoost Regressor  ")
    
print(" Avg R^2 Score :",np.mean(score_abr))
print('\n\n')

print("Decision Tree Regressor  ")
    
print(" Avg R^2 Score :",np.mean(score_dtr))
print('\n\n')

print("SGDRegressor  ")
    
print(" Avg R^2 Score :",np.mean(score_sgd))
print('\n\n')


print('\n\nPrice prediction Using Linear Regression\n',y_pred_lr)
print('\n\nPrice prediction Using Random Forest  Regression\n',y_pred_rfr)
print('\n\nPrice prediction Using AdaBoost Regressor\n',y_pred_abr)
print('\n\nPrice prediction Using Decision Tree Regressor\n',y_pred_dtr)
print('\n\nPrice prediction Using SGDRegressor\n',y_pred_sgd)
print('\n\n\n')





          