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
from sklearn.metrics import roc_curve, roc_auc_score, cohen_kappa_score, RocCurveDisplay ,precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# from sklearn.metrics import confusion_matrix # Confusion Matrix
# 參考資料：https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
# import seaborn as sns
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


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

import xgboost as xgb

from sklearn import preprocessing # 特徵正規化

from scipy.stats import norm

from sklearn.utils import resample

import statsmodels.api as sm #Bland altman



'''------Data Preprocessing------'''

def sfs_dimensionality(features, get_features_n, x_train, y_train, df_train):
    
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

def selectKBest(df_train, feature_n):
# 選擇要保留的特徵數
    select_k = feature_n
    selection = SelectKBest(mutual_info_classif, k=select_k).fit(X_train, y_train)
    # 顯示保留的欄位
    features = df_train.iloc[:, 3:25].columns[selection.get_support()]
    
    return features


def label_stress(df_all, label_style):
    
    if label_style == 'binary':
        
        df_all.loc[df_all['Situation'] == 'Baseline', 'Y_Label'] = 0
        df_all.loc[df_all['Situation'] != 'Baseline', 'Y_Label'] = 1
        
        return df_all
        
    elif label_style == 'multiclass':

        df_all.loc[df_all['Situation'] == 'Baseline', 'Y_Label'] = 0
        df_all.loc[df_all['Situation'] == 'Stroop', 'Y_Label'] = 1
        df_all.loc[df_all['Situation'] == 'Arithmetic', 'Y_Label'] = 2
        df_all.loc[df_all['Situation'] == 'Speech', 'Y_Label'] = 3
        
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
        
        return df_merge
        
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
    
    # forest = ensemble.RandomForestClassifier(n_estimators = n_estimators)
    # forest_fit = forest.fit(X_train, y_train)
    # y_train_pred = forest.predict(X_train)
    # y_pred = forest.predict(X_valid)
    # accuracy_valid = forest.score(X_valid, y_valid)
    clf=RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_valid)
    y_train_pred=clf.predict(X_train)
    accuracy_valid = clf.score(X_valid, y_valid)
    
    
    return clf, accuracy_valid, y_pred, y_train_pred


def regression(X_train, y_train, X_valid, y_valid):

    regr=linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_valid)
    accuracy = regr.score(X_valid, y_valid)
    
    return accuracy, y_pred

def decision_tree_regression(X_train, y_train, X_valid, y_valid, max_depth):
    
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_pred = tree.predict(X_valid)
    
    return y_train_pred, y_pred
    

def random_forest_regression(X_train, y_train, X_valid, y_valid):
    
    forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_valid_pred = forest.predict(X_valid)
    accuracy_valid = forest.score(X_valid, y_valid)
    accuracy_train = forest.score(X_train, y_train)
    
    return accuracy_train, accuracy_valid, y_train_pred, y_valid_pred, forest
    
    # !!! 參考資料！https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    


# Xgboost 迴歸
# 參考資料：https://ithelp.ithome.com.tw/articles/10273094

def xgb_regression(X_train, y_train, X_valid, y_valid):

    xgbrModel=xgb.XGBRegressor(max_depth = 6)
    xgbrModel.fit(X_train,y_train)
    y_pred =xgbrModel.predict(X_valid)
    accuracy = xgbrModel.score(X_valid, y_valid)
    
    return accuracy, y_pred



#%%

'''Label Data'''
df = pd.read_excel('Data/220810_Features.xlsx')
df = df.dropna()
df = df[df['Mean']!=0]

# 讀取Data
classify_columns = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
# classify_columns = ['Baseline',  'Speech']
label_column = 'Y_Label'  

df = df[df['Situation'].isin(classify_columns)]
df = label_stress(df, label_style='VAS_int') # binary, multiclass, VAS_int

plt_columns=['Baseline','Stroop', 'Arith', 'Speech']
# plt_columns=['No stress','Stress']



x = np.array([2,3,4,5])[:, np.newaxis]


# Resample

# df_nostress = df[df.Y_Label==0]
# df_stress = df[df.Y_Label==1]
# df_stress_upsampled = resample(df_stress, 
#                                  replace=True,     # sample with replacement
#                                  n_samples=len(df_nostress),    # to match majority class
#                                  random_state=42) # reproducible results
# df_upsampled = pd.concat([df_nostress, df_stress_upsampled])
# df = df_upsampled


#%%

# 選特徵


# 全部特徵

# select_features = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 
#             'EMG_RMS','EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC',
#             'TP', 'HF', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']

# 全部特徵刪除有線性關係
# Title = 'ECG and EMG features '
# select_features = ['Mean','SD', 'RMSSD','NN50', 'Skewness','Kurtosis', 
#             'EMG_RMS', 'EMG_ENERGY', 'EMG_VAR', 'EMG_ZC',
#             'TP', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']
# HRV參數
# Title = 'ECG features '
# select_features = ['Mean','SD', 'RMSSD','NN50', 'Skewness','Kurtosis', 
#             'TP', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF']

# EMG參數
# Title = 'EMG features '
# select_features = ['EMG_RMS','EMG_RMS', 'EMG_ENERGY', 'EMG_VAR', 'EMG_ZC', 'MNF', 'MDF']

# 與VAS問卷有顯著相關的特徵
Title = 'Significant correlation’s features'
select_features = ['Mean','SD','Kurtosis', 
            'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR',
            'TP', 'HF', 'LF', 'nLF', 'LF/HF', 'MNF', 'MDF']
            
# |Pearson| > 0.1            
# select_features = ['Mean', 'NN50', 'Skewness', 'EMG_RMS', 'EMG_ENERGY', 'EMG_ZC','LF', 'nLF', 'nHF', 'MNF', 'MDF']

# SFS選特徵 (特徵=6)
# select_features = ['NN50', 'pNN50', 'EMG_RMS', 'EMG_MAV', 'EMG_VAR', 'MNF'] #knn=6 select



y = df['Y_Label']

#%%
# 原始散佈圖
# sns.set(style='whitegrid', context='notebook')
# cols = ['Mean','SD', 'RMSSD','NN50', 'Skewness','Kurtosis', 'Y_Label']
# cols = ['EMG_RMS','EMG_ENERGY', 'EMG_VAR', 'EMG_ZC', 'Y_Label']
# cols = [ 'TP', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'Y_Label']
# cols = ['MNF', 'MDF', 'Y_Label']
# sns.pairplot(df[cols], size=2.5)
# plt.show()
#%%

# Pearson 相關性Heatmap圖
# plt.figure(figsize=(16, 16))
# heatmap = sns.heatmap(df[select_features].corr(), vmin=-1, vmax=1, annot=True)
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

# sns.reset_orig()

#%%
# # 模型訓練
filterfeatures = df.loc[:, select_features].values

X_train, X_valid, y_train, y_valid = train_test_split(filterfeatures, y, test_size=0.30, random_state=123, stratify=df[label_column])  # stratify: 設定Valid data資料狀況平衡


# model, accuracy, y_pred, y_train_pred = random_forest(X_train, y_train, X_valid, y_valid, 100)
# accuracy, y_pred = svr(X_train, y_train, X_valid, y_valid, kernel_style='linear')
# accuracy, y_pred = xgb_regression(X_train, y_train, X_valid, y_valid)
# y_train_pred, y_pred  = decision_tree_regression(X_train, y_train, X_valid, y_valid, 3)
accuracy_train, accuracy_valid, y_train_pred, y_pred, forest_model  = random_forest_regression(X_train, y_train, X_valid, y_valid)




#%%模型評估

'''學習曲線'''
# from sklearn.model_selection import learning_curve

# train_sizes, trains_scores, test_scores = learning_curve(forest_model, 
#                                                           X = X_train, y = y_train, train_sizes=np.linspace(0.1, 1.0, 10),
#                                                           cv=10, n_jobs=1)
# train_means = np.mean(trains_scores, axis=1)
# train_std = np.std(trains_scores, axis=1)
# test_means = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.figure()
# plt.plot(train_sizes, train_means, color='blue', marker='o', label='Training accuracy')
# plt.fill_between(train_sizes, train_means+train_std, train_means-train_std, alpha=0.15, color='blue')

# plt.plot(train_sizes, test_means, color='green', marker='o', label='Validation accuracy')
# plt.fill_between(train_sizes, test_means+test_std, test_means-test_std, alpha=0.15, color='green')
# plt.ylim(0,1)
# plt.ylabel('Accuracy')
# plt.xlabel('Number of training samples')
# plt.legend(loc='lower right')


'''迴歸模型'''

print('Training Accuracy: {}'.format(round(accuracy_train,2)))
print('Valid Accuracy: {}'.format(round(accuracy_valid,2)))

mse_train = mean_squared_error(y_train_pred, y_train)
mse_valid = mean_squared_error(y_pred, y_valid)
        

print('Training data MSE: {}'.format(round(mse_train,2)))
print('Validation data MSE: {}'.format(round(mse_valid,2)))

r2_train = r2_score(y_train_pred, y_train)
r2_valid = r2_score(y_valid, y_pred)
print('Training data R2: {}'.format(round(r2_train,2)))
print('Validation data R2: {}'.format(round(r2_valid,2)))

error_predandvalid = y_pred-y_valid
# 殘差圖
# plt.figure()
# plt.scatter(y_pred, error_predandvalid, c='steelblue', marker='o', label='Validation data')
# plt.scatter(y_train_pred, y_train_pred-y_train, c='black', marker='o', label=['Training data', 'Validation data'])
# plt.xlabel('Predictied values', fontsize=18)
# plt.ylabel('Residuals', fontsize=18)
# plt.title(Title, fontsize=20)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.ylim(-100,100)
# plt.xlim(0,100)
# plt.hlines(y=0, xmin=0, xmax=100, color='grey')


# plt.tight_layout()


# sns.displot(data=error_predandvalid,  kde=True, edgecolor="white", color='black')
# plt.xlabel('Error', fontsize=18)
# plt.ylabel('Count', fontsize=18)
# plt.title(Title, fontsize=20)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.ylim(0,80)
# plt.xlim(-100,100)

# plt.tight_layout()

# # Bland Altman
# f, ax = plt.subplots(1, figsize = (6,4))
# sm.graphics.mean_diff_plot(y_pred,y_valid, ax = ax, scatter_kwds={"color": "black"})
# ax.set_title(Title, fontsize=20)
# ax.set_ylim(-100, 100)
# ax.set_xlim(0, 100)
# ax.set_xlabel('Predictied values', fontsize=18)
# ax.set_ylabel('Difference', fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.tight_layout()
# plt.show()


f, ax = plt.subplots(1, figsize = (6,4))
sm.graphics.mean_diff_plot(y_train_pred,y_train, ax = ax, scatter_kwds={"color": "black"})
ax.set_title(Title, fontsize=20)
ax.set_ylim(-100, 100)
ax.set_xlim(0, 100)
ax.set_xlabel('Mean', fontsize=18)
ax.set_ylabel('Difference', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()





'''分類模型'''

'''Confusion Matrix'''
# y_valid = y_valid.astype(int)
# y_pred = y_pred.astype(int)
# nunique_labels = len(set(y_train))
# conf_mat_shape = (nunique_labels, nunique_labels)
# confusion_matrix = np.zeros(conf_mat_shape, dtype=int)
# for actual, predict in zip(y_valid, y_pred):
#   confusion_matrix[actual, predict] += 1

# plt.figure(figsize=(5,5))
# # ax = sns.heatmap(confusion_matrix, annot=True, fmt='.20g', cmap='Blues', annot_kws={'fontsize': 16}) # 個數
# ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
#                     fmt='.2%', cmap='Blues', annot_kws={'fontsize': 20}) # 比例
# ax.set_title(Title+'\n', fontsize=20);
# ax.set_xlabel('\nPredicted Values\n', fontsize=20)
# ax.set_ylabel('Actual Values\n', fontsize=20);
# ax.xaxis.set_ticklabels(plt_columns, fontsize=20)
# ax.yaxis.set_ticklabels(plt_columns, fontsize=20)
# plt.tight_layout()


# '''評估參數計算'''
# ####  Binary
# total1=sum(sum(confusion_matrix))
# sensitivity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
# specificity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
# precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
# F1score = 2 * (precision * sensitivity) / (precision + sensitivity)
# kappa = cohen_kappa_score(y_valid, y_pred)

# # AUC and ROC curve
# prob = model.predict_proba(X_valid)
# prob = prob[:, 1]
# fpr, tpr, thresholds = roc_curve(y_valid, prob)
# auc = metrics.auc(fpr, tpr)

# plt.figure(figsize=(6,6))
# plt.plot(fpr, tpr, color='steelblue')
# plt.plot([0, 1], [0, 1], color='black', linestyle='--')
# plt.xlabel('False Positive Rate', fontsize=18)
# plt.ylabel('True Positive Rate', fontsize=18)
# plt.title(Title+'\n', fontsize=22)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.show()

# print('')
# print(confusion_matrix)
# print('Accuracy = %0.2f' %accuracy)
# print('Sensitivity = %0.2f' % sensitivity)
# # print('Specificity = %0.2f' % specificity)
# print('Precision = %0.2f' % precision)
# print('F1score = %0.2f' % F1score)
# print('Kappa = %0.2f' % kappa)
# print('AUC = %0.2f' % auc)


### Muticlass

# F1score = f1_score(y_valid, y_pred, average=None)
# precision = precision_score(y_valid, y_pred, average=None)
# sensitivity = recall_score(y_valid, y_pred, average=None)
# kappa = cohen_kappa_score(y_valid, y_pred)
# confusion_matrix = confusion_matrix(y_valid, y_pred)

# df_result = pd.DataFrame({'Accuracy':round(accuracy,2) ,'kappa': round(kappa,2), 'sensitivity':np.around(sensitivity, decimals=2) ,'precision':np.around(precision, decimals=2), 'F1score': np.around(F1score, decimals=2)})


# # Confusion matrix
# ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
#                     fmt='.2%', cmap='Blues', annot_kws={'fontsize': 16}) # Rate
# ax.set_title(Title+'\n', fontsize=20);
# ax.set_xlabel('Predicted Values\n', fontsize=16)
# ax.set_ylabel('Actual Values\n', fontsize=16)
# ax.xaxis.set_ticklabels(plt_columns, fontsize=16)
# ax.yaxis.set_ticklabels(plt_columns, fontsize=16)
# plt.show()

# ## AUC and ROC curve
# n_classes = 4
# fpr = dict()
# tpr = dict()
# auc = dict()
# prob = model.predict_proba(X_valid)
# y_valid_dummies = pd.get_dummies(y_valid, drop_first=False).values
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_valid_dummies[:, i], prob[:, i])
#     auc[i] = metrics.auc(fpr[i], tpr[i])

# plt.figure()
# class_column = ['No stress', 'Stroop', 'Arithmetic', 'Speech']
# for i in range(n_classes):
    
#     plt.plot(fpr[i], tpr[i], label=class_column[i]+' (AUC = {})'.format(round(auc[i],2)))
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.xlabel('False Positive Rate', fontsize=18)
#     plt.ylabel('True Positive Rate', fontsize=18)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.title(Title+'\n', fontsize=20)
#     plt.legend(loc="lower right")
#     plt.show()
    

# df_result.to_excel('/Users/weien/Desktop/New ML/8.xlsx')


# print(confusion_matrix)


