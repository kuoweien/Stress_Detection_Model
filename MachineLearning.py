#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:20:07 2022

@author: weien
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc # ACU
# from sklearn.metrics import confusion_matrix # Confusion Matrix
from sklearn.metrics import cohen_kappa_score
# 參考資料：https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
import seaborn as sns
from sklearn.decomposition import PCA
# 特徵選取
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
# Sequential Feature Selector (SFS特徵選取)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble, preprocessing, metrics


# 讀取Data
# df = pd.read_excel('Data/220714_HRV_DeleteMoreSpeech.xlsx')
df = pd.read_excel('Data/220714_HRV_DeleteMoreSpeech.xlsx')
# df = df[df['Situation'].isin(['Baseline','Speech'])]
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)
df_train = df_train.dropna()

# 選特徵
select_features = ['NN50', 'pNN50', 'EMG_RMS', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC']


# X_train = df_train.iloc[:, 3:15].values  # All features
# X_train = df_train.loc[:, ['Mean', 'SD', 'Kurtosis', 'EMG_RMS', 'EMG_MAV', 'EMG_VAR']].values
X_train = df_train.loc[:, select_features].values
y_train = df_train.loc[:, "Y_Binary"].values


# knn = KNeighborsClassifier(n_neighbors=6)
# sfs = SequentialFeatureSelector(knn, n_features_to_select=6)
# sfs.fit(X_train, y_train)
# features = df_train.iloc[:, 3:15].columns[sfs.get_support()]


df_test = df_test.dropna()
# X_test = df_test.iloc[:, 3:15].values  # All features
# X_test = df_test.loc[:, ['Mean', 'SD', 'Kurtosis', 'EMG_RMS', 'EMG_MAV', 'EMG_VAR']].values
X_test = df_test.loc[:, select_features].values
y_test = df_test.loc[:, "Y_Binary"].values




'''--------Support Vector Mechine (Binary)------'''
# 參數調整參考資料： https://blog.csdn.net/TeFuirnever/article/details/99646257
# clf = svm.SVC()
# clf = svm.SVC(kernel='linear') #kernal : linear, poly, rbf (預設) | c = 1(預設)
# clf.fit(X_train, y_train)
# Test Data準確度
# y_test_pred = clf.predict(X_test)
# accuracy_test = clf.score(X_test, y_test)

'''-------Random Forest---------'''
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(X_train, y_train)

y_test_pred = forest.predict(X_test)
accuracy_test = forest.score(X_test, y_test)




'''---------Support Vector Mechine (Multi-class classification)---------'''
# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(X_train, y_multi_train)
# y_pred_svm_multi = clf.predict(X_train)
# accuracy_svm_multi = clf.score(X_train, y_multi_train)


'''---------Support Vector Regression---------'''
# regr=svm.SVR(C=1, kernel='linear')
# # regr = svm.SVR(C=1, kernel='poly', degree=3, gamma='auto')
# regr.fit(X_train, y_train)
# y_pred_svr = regr.predict(X_train)
# accuracy_svr = regr.score(X_train, y_train)

# plt.figure()
# plt.plot(y_train)
# plt.plot(y_pred)

'''----------模型成果------------'''
# Confusion Matrix
y_test = y_test.astype(int)
y_test_pred = y_test_pred.astype(int)
nunique_labels = len(set(y_train))
conf_mat_shape = (nunique_labels, nunique_labels)
confusion_matrix = np.zeros(conf_mat_shape, dtype=int)
for actual, predict in zip(y_test, y_test_pred):
  confusion_matrix[actual, predict] += 1

plt.figure(figsize=(5,4))
ax = sns.heatmap(confusion_matrix, annot=True, fmt='.20g', cmap='Blues', annot_kws={'fontsize': 16}) # 個數
# ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
                    # fmt='.2%', cmap='Blues', annot_kws={'fontsize': 16}) # 比例
ax.set_title('All features', fontsize=16);
ax.set_xlabel('Predicted Values', fontsize=16)
ax.set_ylabel('\nActual Values', fontsize=16);
ax.xaxis.set_ticklabels(['Baseline','Stress'], fontsize=14)
ax.yaxis.set_ticklabels(['Baseline','Stress'], fontsize=14)
plt.tight_layout()


# 評估參數計算
total1=sum(sum(confusion_matrix))
sensitivity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
specificity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
F1score = 2 * (precision * sensitivity) / (precision + sensitivity)


#AUC and ROC curve
# y_score = clf.fit(X_test, y_test).decision_function(X_test)
# fpr,tpr,threshold = roc_curve(y_test, y_score)
# roc_auc = auc(fpr,tpr)
# kappa = cohen_kappa_score(y_test, y_test_pred)


#ROC curve
# lw = 2
# plt.figure()
# plt.plot(fpr, tpr, color='black',lw=lw) ###假正率為橫座標，真正率為縱座標做曲線
# plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.xlabel('False Positive Rate',fontsize=16)
# plt.ylabel('True Positive Rate',fontsize=16)
# plt.title('All features',fontsize=18)
# plt.show()


print()
print(confusion_matrix)
print('Accuracy = %0.2f' %accuracy_test)
print('Sensitivity = %0.2f' % sensitivity)
print('Specificity = %0.2f' % specificity)
print('Precision = %0.2f' % precision)
print('F1score = %0.2f' % F1score)
# print('AUC = %0.2f' % roc_auc)
# print('Kappa = %0.2f' % kappa)



# 選擇要保留的特徵數
# select_k = 3
# selection = SelectKBest(mutual_info_classif, k=select_k).fit(X_train, y_train)

# # 顯示保留的欄位
# features = df_train.iloc[:, 3:15].columns[selection.get_support()]

