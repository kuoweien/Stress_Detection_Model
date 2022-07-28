#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:54:48 2022

@author: weien
"""
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import scipy.stats as stats

#看是否常態
def checkisNormal(value):
    u = value.mean()  
    std = value.std()  
    a=stats.kstest(value, 'norm', (u, std))
    
    p_value=a[1]  # corr=a[0]
    return p_value  # if p>0.05常態, p<0.05非常態

# T Test (控制組與對照組相比)(資料為常態時使用)
def t_test(A_list, B_list):
    A_mean = np.mean(A_list)
    B_mean = np.mean(B_list)
    A_sd = np.std(A_list)
    B_sd = np.std(B_list)
    A_nobs = len(A_list)
    B_nobs = len(B_list)
    
    modified_sd_A = np.sqrt(np.float32(A_nobs)/
                    np.float32(A_nobs-1)) * A_sd
    modified_sd_B = np.sqrt(np.float32(B_nobs)/
                    np.float32(B_nobs-1)) * B_sd
    t, p = stats.ttest_ind_from_stats(mean1=A_mean, std1=modified_sd_A, nobs1=A_nobs,mean2=B_mean, std2=modified_sd_B, nobs2=B_nobs)
    
    return t, p


# Mann Whitney U test (控制組與對照組相比)(資料為非常態時使用)
def mannwhitneyu_test(A_list, B_list):
    u, p = mannwhitneyu(A_list, B_list)
    
    return u, p 

# Paired T Test
def paired_ttest(pre, post): # (同組人介入前後相比)(資料為常態時使用)
    t, p = stats.ttest_rel(pre, post)
    
    return t, p

# Wilcoxon Signed-Rank Test
def wilcoxon_signed_rank_test(pre, post): # (同組人介入前後相比)(資料為常態時使用)
    w, p = stats.wilcoxon(pre, post)
    
    return w, p
    
    
# 篩除極端值
def deleteOutlier(data):
    data_lower = np.quantile(data, 0.25, interpolation='lower')
    data_higher = np.quantile(data, 0.75, interpolation='higher')
    iqr = data_higher-data_lower
    high_limit = data_higher + 1.5*iqr
    low_limit = data_lower - 1.5*iqr
    
    data_filter = []
    for i in data:
        if i<=high_limit and i>=low_limit:
            data_filter.append(i)
    # data_filter = data[np.where(data <= high_limit)]
    # data_filter = data_filter[np.where(data_filter >= low_limit)]
    
    return data_filter #刪除oitlier之data陣列

# Pearson 有母
def pearson_corr(a_list, b_list):
    if len(a_list) <= 2 or len(a_list) <= 2:
        print('Calculate Pearson Corr:  X and Y must have length at least 2')
    r, p = stats.pearsonr(a_list, b_list)
    return r, p 

# Spearman

#%% 
'''-----------與Baseline相比 壓力情境是否有統計差異-------------(HRV EMG相關參數統計計算)'''

# url = 'Data/220714_HRV.xlsx'
# df = pd.read_excel(url)

# df_baseline = df[df['Situation'] == 'Baseline']


# parameter = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 'EMG_RMS', 'EMG_VAR', 'EMG_ZC', 'EMG_ENERGY', 'EMG_MAV'] # 計算的參數
# for para in parameter:
#     print('[{}]'.format(para))
       
#     columns = ['Stroop', 'Arithmetic', 'Speech'] # 人壓力情境
#     # columns = ['Touch', 'Scared'] # 狗情緒情境
    
#     #壓力情境與Baseline比，是否有達到顯著差異
#     for j in columns:
#         df_stress = df[df['Situation'] == j]
#         print(j)
    
#         baseline_mean = df_baseline[para]
#         stess_mean = df_stress[para]
        
#         baseline_mean_pvalue = checkisNormal(baseline_mean)
#         stress_mean_pvalue = checkisNormal(stess_mean)
        
#         if  baseline_mean_pvalue>0.05 and stress_mean_pvalue>0.05:
#             r, pvalue = paired_ttest(baseline_mean, stess_mean)
#             print('Paired T Test')
            
#         elif  baseline_mean_pvalue>0.05 and stress_mean_pvalue<=0.05:
#             r, pvalue = wilcoxon_signed_rank_test(baseline_mean, stess_mean)
#             print('Wilcoxon Signed-Rank Test')
            
#         elif  baseline_mean_pvalue<=0.05 and stress_mean_pvalue>0.05:
#             r, pvalue = wilcoxon_signed_rank_test(baseline_mean, stess_mean)
#             print('Wilcoxon Signed-Rank Test')
            
#         elif  baseline_mean_pvalue<=0.05 and stress_mean_pvalue<=0.05:
#             r, pvalue = wilcoxon_signed_rank_test(baseline_mean, stess_mean)
#             print('Wilcoxon Signed-Rank Test')
            
        
#         print('p={}'.format(round(pvalue,3)))
#         print('-----')
      
   
'''-----------VAS內容統計差異-------------(問卷內容統計計算)'''

# url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/問卷&流程資料/Questionnaire.xlsx'
# df_vas = pd.read_excel(url)
# baseline_vas = df_vas['Baseline_VAS (mm)']
# stroop_vas = df_vas['Stroop test_VAS (mm)']
# arithmetic_vas = df_vas['Arithmetic_VAS (mm)']
# speech_vas = df_vas['Speech_VAS (mm)']

# columns = [stroop_vas, arithmetic_vas, speech_vas]
# baseline_vas_pvalue = checkisNormal(baseline_vas)

# # columns = [arithmetic_vas, speech_vas]
# # stroop_vas_pvalue = checkisNormal(stroop_vas)
# # baseline_vas_pvalue = stroop_vas_pvalue

# # columns = [speech_vas]
# # stroop_vas_pvalue = checkisNormal(arithmetic_vas)
# # baseline_vas_pvalue = stroop_vas_pvalue

# for i in columns:

#     stress_vas_pvalue = checkisNormal(i)
    
#     if  baseline_vas_pvalue>0.05 and stress_vas_pvalue>0.05:
#         r, pvalue = paired_ttest(baseline_vas, i)
#         print('Paired T Test')
            
#     elif  baseline_vas_pvalue>0.05 and stress_vas_pvalue<=0.05:
#         r, pvalue = wilcoxon_signed_rank_test(baseline_vas, i)
#         print('Wilcoxon Signed-Rank Test')
            
#     elif  baseline_vas_pvalue<=0.05 and stress_vas_pvalue>0.05:
#         r, pvalue = wilcoxon_signed_rank_test(baseline_vas, i)
#         print('Wilcoxon Signed-Rank Test')
            
#     elif  baseline_vas_pvalue<=0.05 and stress_vas_pvalue<=0.05:
#         r, pvalue = wilcoxon_signed_rank_test(baseline_vas, i)
#         print('Wilcoxon Signed-Rank Test')
    
#     print('P value={}'.format(pvalue))

'''------------基本資料計算------------------'''
# df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/問卷&流程資料/基本資料&設備編號檔名.xlsx')
# participants_age = df['Age']
# age_mean = round(np.mean(participants_age),1)
# age_std = round(np.std(participants_age),1)

# df_quetionnaire = pd.read_excel('/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/問卷&流程資料/Questionnaire.xlsx')
# baseline_VAS = df_quetionnaire['Baseline_VAS (mm)']
# baseline_VAS_mean = round(np.mean(baseline_VAS),1)
# baseline_VAS_std = round(np.std(baseline_VAS),1)

# stroop_VAS = df_quetionnaire['Stroop test_VAS (mm)']
# stroop_VAS_mean = round(np.mean(stroop_VAS),1)
# stroop_VAS_std = round(np.std(stroop_VAS),1)

# arithmetic_VAS = df_quetionnaire['Arithmetic_VAS (mm)']
# arithmetic_VAS_mean = round(np.mean(arithmetic_VAS),1)
# arithmetic_VAS_std = round(np.std(arithmetic_VAS),1)

# speech_VAS = df_quetionnaire['Speech_VAS (mm)']
# speech_VAS_mean = round(np.mean(speech_VAS),1)
# speech_VAS_std = round(np.std(speech_VAS),1)

'''---------每人的HRV參數平均 -------'''
#要先建立空的excel檔，先定好欄位名

# df = pd.read_excel('Data/220714_HRV.xlsx')
# total_N = 24

# columns = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 'EMG_RMS','EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC']
# stress = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']


# df_person_siation_mean = pd.DataFrame()

# df_siation_mean = pd.read_excel('/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRVMean_VAS/220716_HRVMean.xlsx')
# # df_siation_mean = pd.DataFrame()
# n_list = []

# for n in range(2,(total_N+1)):
#     df_a_person = df[df['N']==n]
#     for i in stress:
#         df_situation = df_a_person[df_a_person['Situation']==i]
#         df_person_siation_mean = df_person_siation_mean.append(df_situation.mean(), ignore_index=True)
#     n_list = n_list+[n]*4

# df_siation_mean = df_siation_mean.append(df_person_siation_mean, ignore_index=True)
# df_siation_mean['Situation'] = stress*(total_N-1)
# df_siation_mean['N'] = n_list


## df_siation_mean.to_excel('/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRVMean_VAS/220716_HRVMean.xlsx')

'''------------計算相關：HRV與VAS相關-------------'''

df = pd.read_excel('Data/220719_HRVMean.xlsx')
df_output = pd.DataFrame()
allN = 24

for j in range(2, (allN+1)):
    if j == 7:
        continue
    
    df_N = df[df['N']==j]
    if len(df_N) == 0:
        continue
        
    vas_N = df_N['VAS']
    
    columns = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 'EMG_RMS','EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC']

    
    for i in columns:
        hrv_N = df_N[i]
        r, p = pearson_corr(vas_N, hrv_N)
        df_output = df_output.append({'N': j, 'HRVPara': i, 'Corr_HRVandVAS': r }, ignore_index=True)

df_output.to_excel('Data/HRVandVAS_Corr.xlsx')




