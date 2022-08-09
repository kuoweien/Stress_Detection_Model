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
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble, preprocessing, metrics
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import xgboost as xgb

from sklearn import preprocessing # 特徵正規化

'''------Data Preprocessing------'''

def sfs_dimensionality(features, get_features_n, x_train, y_train):
    
    knn = KNeighborsClassifier(n_neighbors=get_features_n)
    sfs = SequentialFeatureSelector(knn, n_features_to_select=get_features_n)
    sfs.fit(x_train, y_train)
    features = df_train.loc[:, features].columns[sfs.get_support()]
    
    return features


def pca(X_data, components_n):
    # PAC降維
    pca = PCA(n_components = components_n)
    pca.fit(X_data)
    X_pca_data = pca.transform(X_data)
    
    return X_pca_data

def selectKBest(X_train, y_train, feature_n):
# 選擇要保留的特徵數
    select_k = feature_n
    selection = SelectKBest(mutual_info_classif, k=select_k).fit(X_train, y_train)
    # 顯示保留的欄位
    features = df_train.iloc[:, 3:25].columns[selection.get_support()]
    
    return features


def label_stress(df_all, label_style):
    
    if label_style == 'binary':
        df_all['Y_Label'][(df_all['Situation'] == 'Baseline')] = 0
        df_all['Y_Label'][(df_all['Situation'] == 'Stroop') | (df_all['Situation'] == 'Speech') | (df_all['Situation'] == 'Arithmetic')] = 1
        return df_all
        
    elif label_style == 'multiclass':
        df_all['Y_Label'][(df_all['Situation'] == 'Baseline')] = 0
        df_all['Y_Label'][(df_all['Situation'] == 'Stroop')] = 1
        df_all['Y_Label'][(df_all['Situation'] == 'Arithmetic')] = 2
        df_all['Y_Label'][(df_all['Situation'] == 'Speech')] = 3
        return df_all
        
    elif label_style == 'VAS_raw':
        df_vas = pd.read_excel('Data/Questionnaire.xlsx', sheet_name = 'forAnalyze')
        df_merge = pd.merge(df_all, df_vas, on=['N','Situation'])
        df_merge.rename(columns={'VAS': 'Y_Label'}, inplace=True)

        return df_merge
    
    elif label_style == 'VAS_int':
        df_vas = pd.read_excel('Data/Questionnaire.xlsx', sheet_name = 'forAnalyze')
        df_merge = pd.merge(df_all, df_vas, on=['N','Situation'])
        df_merge.rename(columns={'VAS': 'Y_Label'}, inplace=True)
        vas_num = np.round(df_merge['Y_Label'],0)
        df_merge['Y_Label'] = vas_num
        
    elif label_style == 'VAS_int_to_tens':
        df_vas = pd.read_excel('Data/Questionnaire.xlsx', sheet_name = 'forAnalyze')
        df_merge = pd.merge(df_all, df_vas, on=['N','Situation'])
        df_merge.rename(columns={'VAS': 'Y_Label'}, inplace=True)
        vas_num = np.round(df_merge['Y_Label'],-1)/10
        df_merge['Y_Label'] = vas_num
        
        return df_merge
    
def balance_baselineandstress_data(df): #Let Baseline and Stress data is balance
    df_stress = df[df['Situation'] != 'Baseline']
    df_baseline = df[df['Situation'] == 'Baseline']
    df_stress = df_stress.sample(n = df_baseline.shape[0])
    df_resample = df_stress.append(df_baseline)
    
    return df_resample
  
  
'''-----Model Building-------'''
 
# Support Vector Mechine (Binary)  
def svm_binary(X_train, y_train, X_valid, y_valid, kernel_style):
    #參數調整參考資料： https://blog.csdn.net/TeFuirnever/article/details/99646257
    #kernel_style : linear, poly, rbf (預設) | c = 1(預設)

    clf = svm.SVC()
    clf = svm.SVC(kernel=kernel_style) 
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    accuracy = clf.score(X_valid, y_valid)
    
    return accuracy, y_pred


# Support Vector Mechine (Multi-class classification)
def svm_multi_label(X_train, y_train, X_valid, y_valid):

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    accuracy = clf.score(X_valid, y_valid)
    
    return accuracy, y_pred

# Support Vector Regression
def svr(X_train, y_train, X_valid, y_valid, kernel_style):

    regr=svm.SVR(C=1, kernel=kernel_style)

    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_valid)
    accuracy = regr.score(X_valid, y_valid)
    
    return accuracy, y_pred


# Random Forest
def random_forest(X_train, y_train, X_valid, y_valid, n_estimators): 
    # # n_estimators 設定樹的深度 預設為100
    
    forest = ensemble.RandomForestClassifier(n_estimators = n_estimators)
    forest_fit = forest.fit(X_train, y_train)
    y_pred = forest.predict(X_valid)
    accuracy = forest.score(X_valid, y_valid)
    
    # X, y = make_classification(n_samples=1000, n_features=21, 
    #                            n_informative=2, n_redundant=0, 
    #                            random_state=0, shuffle=False)
    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    # clf.fit(X, y)
    # y_pred = clf.predict(X_valid)
    # accuracy = clf.score(X_valid, y_valid)
    
    return accuracy, y_pred


def regression(X_train, y_train, X_valid, y_valid):

    regr=linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_valid)
    accuracy = regr.score(X_valid, y_valid)
    
    return accuracy, y_pred


# Xgboost 迴歸
# 參考資料：https://ithelp.ithome.com.tw/articles/10273094

def xgb_regression(X_train, y_train, X_valid, y_valid):

    xgbrModel=xgb.XGBRegressor(max_depth = 6)
    xgbrModel.fit(X_train,y_train)
    y_pred =xgbrModel.predict(X_valid)
    accuracy = xgbrModel.score(X_valid, y_valid)
    
    return accuracy, y_pred



  

'''Label Data'''
df = pd.read_excel('Data/220801_Features.xlsx')
df = df.dropna()
df = df[df['Mean']!=0]

# 讀取Data
# classify_columns = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
classify_columns = ['Baseline',  'Speech']
label_column = 'Y_Label'  #Y_Binary, Y_Classification, VAS, VAS_binary


df = df[df['Situation'].isin(classify_columns)]
df = label_stress(df, label_style='VAS_int_to_tens')



# 選特徵

# 全部特徵
# select_features = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 
#             'EMG_RMS','EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC',
#             'TP', 'HF', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']

# 全部特徵刪除有線性關係
select_features = ['Mean','SD', 'RMSSD','NN50', 'Skewness','Kurtosis', 
            'EMG_RMS','EMG_RMS', 'EMG_ENERGY', 'EMG_VAR', 'EMG_ZC',
            'TP', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']

# 與VAS問卷有顯著相關的特徵
# select_features = ['Mean','SD','Kurtosis', 
            # 'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR',
            # 'TP', 'HF', 'LF', 'nLF', 'LF/HF', 'MNF', 'MDF']

# SFS選特徵 (特徵=6)
# select_features = ['NN50', 'pNN50', 'EMG_RMS', 'EMG_MAV', 'EMG_VAR', 'MNF'] #knn=6 select

filterfeatures = df.loc[:, select_features].values
normalize_features = preprocessing.scale(filterfeatures)


y = df['Y_Label']

# df_train, df_valid = train_test_split(normalize_features, y, test_size=0.3, random_state=123, stratify=df[label_column])  # stratify: 設定Valid data資料狀況平衡
X_train, X_valid, y_train, y_valid = train_test_split(normalize_features, y, test_size=0.3, random_state=123, stratify=df[label_column])  # stratify: 設定Valid data資料狀況平衡


# X_train = df_train.loc[:, select_features].values
# y_train = df_train.loc[:, label_column].values

# X_valid = df_valid.loc[:, select_features].values
# y_valid = df_valid.loc[:, label_column].values


# accuracy, y_pred = random_forest(X_train, y_train, X_valid, y_valid, 100)
# accuracy, y_pred = svr(X_train, y_train, X_valid, y_valid, kernel_style='linear')
accuracy, y_pred = xgb_regression(X_train, y_train, X_valid, y_valid)


plt.figure()
y_valid = np.array(y_valid)
plt.plot(y_valid)
plt.plot(y_pred)


'''Confusion Matrix'''
# y_valid = y_valid.astype(int)
# y_pred = y_pred.astype(int)
# nunique_labels = len(set(y_train))
# conf_mat_shape = (nunique_labels, nunique_labels)
# confusion_matrix = np.zeros(conf_mat_shape, dtype=int)
# for actual, predict in zip(y_valid, y_pred):
#   confusion_matrix[actual, predict] += 1

# plt.figure(figsize=(5,4))
# ax = sns.heatmap(confusion_matrix, annot=True, fmt='.20g', cmap='Blues', annot_kws={'fontsize': 16}) # 個數
# # ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
# #                     # fmt='.2%', cmap='Blues', annot_kws={'fontsize': 16}) # 比例
# ax.set_title('All features', fontsize=16);
# ax.set_xlabel('Predicted Values', fontsize=16)
# ax.set_ylabel('\nActual Values', fontsize=16);


# ax.xaxis.set_ticklabels(plt_columns, fontsize=14)
# ax.yaxis.set_ticklabels(plt_columns, fontsize=14)
# plt.tight_layout()


'''評估參數計算'''
# total1=sum(sum(confusion_matrix))
# sensitivity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
# specificity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
# precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
# F1score = 2 * (precision * sensitivity) / (precision + sensitivity)


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


# print()
# print(confusion_matrix)
# print('Accuracy = %0.2f' %accuracy)
# print('Sensitivity = %0.2f' % sensitivity)
# print('Specificity = %0.2f' % specificity)
# print('Precision = %0.2f' % precision)
# print('F1score = %0.2f' % F1score)
# # print('AUC = %0.2f' % roc_auc)
# # print('Kappa = %0.2f' % kappa)




