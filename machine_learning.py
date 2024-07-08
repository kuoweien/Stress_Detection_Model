#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:20:07 2022

@author: weien
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

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
from sklearn import preprocessing  # 特徵正規化
from scipy.stats import norm
from sklearn.utils import resample
import statsmodels.api as sm  # Bland altman



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
    # # n_estimators: 樹的深度 預設為100

    clf=RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_valid)
    y_train_pred=clf.predict(X_train)
    accuracy_valid = clf.score(X_valid, y_valid)
    
    # accuracy_cross_valid = cross_val_score(clf,X_valid,y_valid,cv=5,scoring='accuracy')
    
    
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
    
    return forest, accuracy_train, accuracy_valid, y_train_pred, y_valid_pred
    
    # !!! 參考資料！https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    


# Xgboost 迴歸
# 參考資料：https://ithelp.ithome.com.tw/articles/10273094

def xgb_regression(X_train, y_train, X_valid, y_valid):

    xgbrModel=xgb.XGBRegressor(max_depth = 6)
    xgbrModel.fit(X_train,y_train)
    y_pred =xgbrModel.predict(X_valid)
    accuracy = xgbrModel.score(X_valid, y_valid)
    
    return accuracy, y_pred

# Pearson 有母
def pearson_corr(a_list, b_list):
    if len(a_list) <= 2 or len(a_list) <= 2:
        print('Calculate Pearson Corr:  X and Y must have length at least 2')
    r, p = stats.pearsonr(a_list, b_list)
    return r, p 

def spearmon_corr(a_list, b_list):
    if len(a_list) <= 2 or len(a_list) <= 2:
        print('Calculate Pearson Corr:  X and Y must have length at least 2')
    r, p = stats.spearmanr(a_list, b_list)
    return r, p 


def checkisNormal(value):
    u = value.mean()  
    std = value.std()  
    a=stats.kstest(value, 'norm', (u, std))
    
    p_value=a[1]  # corr=a[0]
    return p_value  # if p>0.05常態, p<0.05非常態


#%%
if __name__ == '__main__':
    ###2024/07/08 還未整理

    '''Label Data'''
    df = pd.read_excel('Data/Features/220912_Features.xlsx')
    df = df.dropna()
    df = df[df['Mean']!=0]

    # 讀取Data
    classify_columns = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
    # plt_columns=['No stress','Stress']
    plt_columns=['No stress','Stroop', 'Arith', 'Speech']


    # Select features
    select_features = 'Corr_VAS'  # Four types: ECG_EMG, ECG, EMG, Corr_VAS

    label_style = 'multiclass' # Three types: binary, multiclass, VAS_int



    label_column = 'Y_Label'
    df = df[df['Situation'].isin(classify_columns)] # Filter data by classify_columns
    df = label_stress(df, label_style=label_style) # Label answer by label_style


    y = df['Y_Label']

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
    #ECG+EMG
    if select_features == 'ECG_EMG':

        Title = 'ECG and EMG features '
        select_features = ['Mean','SD', 'RMSSD','NN50', 'Skewness','Kurtosis',
                    'EMG_RMS', 'EMG_ENERGY', 'EMG_VAR', 'EMG_ZC',
                    'TP', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']
    # ECG參數
    elif select_features == 'ECG':
        Title = 'ECG features '
        select_features = ['Mean','SD', 'RMSSD','NN50', 'Skewness','Kurtosis',
                    'TP', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF']

    # EMG參數
    elif select_features == 'EMG':
        Title = 'EMG features '
        select_features = ['EMG_RMS','EMG_RMS', 'EMG_ENERGY', 'EMG_VAR', 'EMG_ZC', 'MNF', 'MDF']

    # Corr with VAS
    elif select_features == 'Corr_VAS':
        # 與VAS問卷有顯著相關的特徵
        Title = 'Significant correlation’s features'
        select_features = ['Mean','SD','Skewness','Kurtosis',
                    'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR',
                    'TP', 'HF', 'LF', 'nLF','nHF', 'LF/HF', 'MNF', 'MDF']

    # |Pearson| > 0.1
    # select_features = ['Mean', 'NN50', 'Skewness', 'EMG_RMS', 'EMG_ENERGY', 'EMG_ZC','LF', 'nLF', 'nHF', 'MNF', 'MDF']

    # SFS選特徵 (特徵=6)
    # select_features = ['NN50', 'pNN50', 'EMG_RMS', 'EMG_MAV', 'EMG_VAR', 'MNF'] #knn=6 select



    #%% Check distributio of raw data

    # Scatter plot

    # sns.set(style='whitegrid', context='notebook')
    # cols = ['Mean','SD', 'RMSSD','NN50', 'Skewness','Kurtosis', 'Y_Label']
    # cols = ['EMG_RMS','EMG_ENERGY', 'EMG_VAR', 'EMG_ZC', 'Y_Label']
    # cols = [ 'TP', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'Y_Label']
    # cols = ['MNF', 'MDF', 'Y_Label']
    # sns.pairplot(df[cols], size=2.5)
    # plt.show()

    # Pearson correlation to Heatmap Plot

    # plt.figure(figsize=(16, 16))
    # heatmap = sns.heatmap(df[select_features].corr(), vmin=-1, vmax=1, annot=True)
    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
    # sns.reset_orig()

    #%%

    # Model training
    filterfeatures = df.loc[:, select_features].values

    X_train, X_valid, y_train, y_valid = train_test_split(filterfeatures, y, test_size=0.20, random_state=123, stratify=df[label_column])  # stratify: 設定Valid data資料狀況平衡

    model, accuracy, y_pred, y_train_pred = random_forest(X_train, y_train, X_valid, y_valid, 100)
    # forest_model, accuracy_train, accuracy_valid, y_train_pred, y_pred  = random_forest_regression(X_train, y_train, X_valid, y_valid)
    # accuracy, y_pred = svr(X_train, y_train, X_valid, y_valid, kernel_style='linear')
    # accuracy, y_pred = xgb_regression(X_train, y_train, X_valid, y_valid)
    # y_train_pred, y_pred  = decision_tree_regression(X_train, y_train, X_valid, y_valid, 3)



    #%%Model preformance
    ''' -------- Binary Label ------------'''

    '''Confusion Matrix'''
    # y_valid = y_valid.astype(int)
    # y_pred = y_pred.astype(int)
    # nunique_labels = len(set(y_train))
    # conf_mat_shape = (nunique_labels, nunique_labels)
    # confusion_matrix_binary = np.zeros(conf_mat_shape, dtype=int)
    # for actual, predict in zip(y_valid, y_pred):
    #   confusion_matrix_binary[actual, predict] += 1

    # plt.figure(figsize=(4,4))
    # plt.rcParams["font.family"] = "Arial"
    # # ax = sns.heatmap(confusion_matrix_binary, annot=True, fmt='.20g', cmap='Blues', annot_kws={'fontsize': 16}) # 個數
    # ax = sns.heatmap(confusion_matrix_binary/np.sum(confusion_matrix_binary), annot=True,
    #                     fmt='.2%', cmap='Greys', annot_kws={'fontsize': 20}, vmin=0, vmax=0.8) # 比例
    # ax.set_title(Title+'\n', fontsize=20);
    # ax.set_xlabel('\nPredicted Values\n', fontsize=20)
    # ax.set_ylabel('Actual Values\n', fontsize=20);
    # ax.xaxis.set_ticklabels(plt_columns, fontsize=16)
    # ax.yaxis.set_ticklabels(plt_columns, fontsize=16)
    # plt.tight_layout()
    # plt.savefig(Title+'_'+label_style+'_Confu.png', dpi=300)

    ''' Preformance: Sensitivity, specificity, Precision, F1, Kappa'''
    # total1=sum(sum(confusion_matrix_binary))
    # sensitivity = confusion_matrix_binary[1,1]/(confusion_matrix_binary[1,0]+confusion_matrix_binary[1,1])
    # specificity = confusion_matrix_binary[0,0]/(confusion_matrix_binary[0,0]+confusion_matrix_binary[0,1])
    # precision = confusion_matrix_binary[1,1]/(confusion_matrix_binary[1,1]+confusion_matrix_binary[0,1])
    # F1score = 2 * (precision * sensitivity) / (precision + sensitivity)
    # kappa = cohen_kappa_score(y_valid, y_pred)


    '''AUC and ROC curve'''
    # prob = model.predict_proba(X_valid)
    # prob = prob[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_valid, prob)
    # auc = metrics.auc(fpr, tpr)

    # plt.figure(figsize=(4,4))
    # plt.rcParams["font.family"] = "Arial"
    # plt.plot(fpr, tpr, color='black')
    # plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    # plt.xlabel('False Positive Rate', fontsize=20)
    # plt.ylabel('True Positive Rate', fontsize=20)
    # plt.title(Title+'\n', fontsize=20)
    # plt.xticks(np.linspace(0,1,3), fontsize=18)
    # plt.yticks(np.linspace(0,1,3), fontsize=18)
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(Title+'_'+label_style+'_ROC.png', dpi=300)

    # '''Print result'''
    # print('')
    # print(confusion_matrix)
    # print('Accuracy = %0.2f' %accuracy)
    # print('Sensitivity = %0.2f' % sensitivity)
    # # print('Specificity = %0.2f' % specificity)
    # print('Precision = %0.2f' % precision)
    # print('F1score = %0.2f' % F1score)
    # print('Kappa = %0.2f' % kappa)
    # print('AUC = %0.2f' % auc)



    ''' ------------- Muticlass Label -------------'''

    '''Preformance: Sensitivity, Precision, kappa, F1'''
    sensitivity = recall_score(y_valid, y_pred, average=None)
    precision = precision_score(y_valid, y_pred, average=None)
    F1score = f1_score(y_valid, y_pred, average=None)
    df_result = pd.DataFrame({'sensitivity':np.around(sensitivity, decimals=2) ,'precision':np.around(precision, decimals=2), 'F1score': np.around(F1score, decimals=2)})

    kappa = cohen_kappa_score(y_valid, y_pred)
    df_result = df_result.append({'accuracy':round(accuracy,2) ,'kappa': round(kappa,2), 'sensitivity':round(np.mean(sensitivity),2) ,'precision':round(np.mean(precision),2), 'F1score': round(np.mean(F1score),2)}, ignore_index=True)


    '''Confusion matrix'''
    confusion_matrix_multi = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(5,5))
    plt.rcParams["font.family"] = "Arial"
    ax = sns.heatmap(confusion_matrix_multi/np.sum(confusion_matrix_multi), annot=True,
                        fmt='.2%', cmap='Greys', annot_kws={'fontsize': 16}, vmin=0, vmax=0.16) # Rate
    ax.set_title(Title+'\n', fontsize=20);
    ax.set_xlabel('Predicted Values\n', fontsize=16)
    ax.set_ylabel('Actual Values\n', fontsize=16)
    ax.xaxis.set_ticklabels(plt_columns, fontsize=14)
    ax.yaxis.set_ticklabels(plt_columns, fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.savefig(Title+'_'+label_style+'_Confusion.png', dpi=300)

    '''AUC and ROC curve'''

    auc_list = []

    n_classes = 4
    fpr = dict()
    tpr = dict()
    auc = dict()
    prob = model.predict_proba(X_valid)
    y_valid_dummies = pd.get_dummies(y_valid, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_valid_dummies[:, i], prob[:, i])
        auc[i] = metrics.auc(fpr[i], tpr[i])

    color = ['black', 'dimgray', 'darkgrey', 'lightgrey']
    plt.figure(figsize=(4,4))
    for i in range(n_classes):

        auc_list.append(round(auc[i],2))

        plt.plot(fpr[i], tpr[i], label=plt_columns[i], color=color[i])
        plt.rcParams["font.family"] = "Arial"
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.xticks(np.linspace(0,1,3), fontsize=18)
        plt.yticks(np.linspace(0,1,3), fontsize=18)
        plt.title(Title+'\n', fontsize=20)
        plt.legend(loc="lower right")

    plt.show()
    plt.tight_layout()
    plt.savefig(Title+'_'+label_style+'_ROC.png', dpi=300)

    auc_list.append(np.mean(auc_list))
    df_result['AUC']=auc_list

    # Output preformance
    df_result = df_result[['accuracy', 'kappa', 'sensitivity', 'precision', 'F1score', 'AUC']]
    # df_result.to_excel('Data/{}.xlsx'.format(Title))



    '''--------------Regression Model------------'''

    '''Correlation between predict and true data'''
    # train_r, train_p = spearmon_corr(y_train_pred, y_train)
    # test_r, test_p = spearmon_corr(y_pred, y_valid)

    # print("Trainning data: r {}, p {}".format(round(train_r,2), round(train_p,2)))
    # print("Test data: r {}, p {}".format(round(test_r,2), round(test_p,2)))

    # '''Preformance: AC, MSE, R2 '''
    # print('Training Accuracy: {}'.format(round(accuracy_train,2)))
    # print('Valid Accuracy: {}'.format(round(accuracy_valid,2)))

    # mse_train = mean_squared_error(y_train_pred, y_train)
    # mse_valid = mean_squared_error(y_pred, y_valid)


    # print('Training data MSE: {}'.format(round(mse_train,2)))
    # print('Validation data MSE: {}'.format(round(mse_valid,2)))

    # r2_train = r2_score(y_train_pred, y_train)
    # r2_valid = r2_score(y_valid, y_pred)
    # print('Training data R2: {}'.format(round(r2_train,2)))
    # print('Validation data R2: {}'.format(round(r2_valid,2)))


    '''Redius plot'''
    # error_predandvalid = y_pred-np.array(y_valid)
    # error_train = y_train_pred-np.array(y_train)

    # Training data
    # plt.figure(figsize=(4,3))
    # plt.scatter(y_train_pred, error_train, c='black', marker='o')
    # plt.rcParams["font.family"] = "Arial"
    # plt.xlabel('Predictied values', fontsize=18)
    # plt.ylabel('Residuals', fontsize=18)
    # plt.title(Title+'\n', fontsize=18)
    # plt.xticks(range(0, 125, 25), fontsize=18)
    # plt.yticks(range(-100, 150, 50), fontsize=18)
    # plt.ylim(-100,100)
    # plt.xlim(0,100)
    # plt.hlines(y=0, xmin=0, xmax=100, color='grey')
    # plt.tight_layout()
    # plt.savefig(Title+'_'+label_style+'_Train_error.png', dpi=200)

    # sns.displot(data=error_train,  kde=True, edgecolor="white", color='black', height=3, aspect=1.2)
    # plt.rcParams["font.family"] = "Arial"
    # plt.xlabel('Error', fontsize=18)
    # plt.ylabel('Count', fontsize=18)
    # plt.title(Title+'\n', fontsize=18)
    # plt.yticks(range(0, 250, 50), fontsize=18)
    # plt.xticks(range(-100, 150, 50), fontsize=18)
    # plt.ylim(0,200)
    # plt.xlim(-100,100)
    # plt.tight_layout()
    # plt.savefig(Title+'_'+label_style+'_Train.png', dpi=200)

    # # Testing data
    # plt.figure(figsize=(4,3))
    # plt.rcParams["font.family"] = "Arial"
    # plt.scatter(y_pred, error_predandvalid, c='black', marker='o', label='Validation data')
    # plt.xlabel('Predictied values', fontsize=18)
    # plt.ylabel('Residuals', fontsize=18)
    # plt.title(Title+'\n', fontsize=20)
    # plt.xticks(range(0, 125, 25), fontsize=18)
    # plt.yticks(range(-100, 150, 50), fontsize=18)
    # plt.ylim(-100,100)
    # plt.xlim(0,100)
    # plt.hlines(y=0, xmin=0, xmax=100, color='grey')
    # plt.tight_layout()
    # plt.savefig(Title+'_'+label_style+'_Test_error.png', dpi=200)

    # sns.displot(data=error_predandvalid,  kde=True, edgecolor="white", color='black', height=3, aspect=1.2)
    # plt.rcParams["font.family"] = "Arial"
    # plt.xlabel('Error', fontsize=18)
    # plt.ylabel('Count', fontsize=18)
    # plt.title(Title+'\n', fontsize=18)
    # plt.yticks(range(0, 100, 20), fontsize=18)
    # plt.xticks(range(-100, 150, 50), fontsize=18)
    # plt.ylim(0,80)
    # plt.xlim(-100,100)
    # plt.tight_layout()
    # plt.savefig(Title+'_'+label_style+'_Test.png', dpi=200)


    '''Bland Altman'''
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
    # plt.rcParams["font.family"] = "Arial"
    # plt.show()
    # plt.savefig(Title+'_'+label_style+'_Train_BA.png', dpi=200)


    # f, ax = plt.subplots(1, figsize = (6,4))
    # sm.graphics.mean_diff_plot(y_train_pred,y_train, ax = ax, scatter_kwds={"color": "black"})
    # ax.set_title(Title, fontsize=20)
    # ax.set_ylim(-100, 100)
    # ax.set_xlim(0, 100)
    # ax.set_xlabel('Mean', fontsize=18)
    # ax.set_ylabel('Difference', fontsize=18)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.tight_layout()
    # plt.rcParams["font.family"] = "Arial"
    # plt.show()
    # plt.savefig(Title+'_'+label_style+'_Test_BA.png', dpi=200)



