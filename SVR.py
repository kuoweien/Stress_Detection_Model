#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:20:07 2022

@author: weien
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split


# 讀取Data
df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/forSVM.xlsx')
# X = df.iloc[:, 3:15].values
df_train, df_validation = train_test_split(df, test_size=0.3, random_state=123)
X_train = df_train.iloc[:, 3:14].values
y_train = df_train.iloc[:, 16:17].values


# Support Vector Mechine


# clf=svm.SVC(kernel='rbf',C=1,gamma='auto')
# clf.fit(X_train,y_train)
# y_hat = clf.predict(X_test)
# clf.score(X_train,y_train))


# Support Vector Regression
regr=svm.SVR(C=1, kernel='linear')
# regr = svm.SVR(C=1, kernel='poly', degree=3, gamma='auto')
regr.fit(X_train, y_train)
y_pred = regr.predict(X_train)
accuracy = regr.score(X_train, y_train)

plt.figure()
plt.plot(y_train)
plt.plot(y_pred)