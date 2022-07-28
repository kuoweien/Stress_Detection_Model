#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:16:34 2022

@author: weien
"""

import pandas as pd


'''---------------原始HRV參數（Raw檔）整理成SigmaPlot要的樣子---------------------'''

# url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRV原始Data/220714_HRV.xlsx'
# df = pd.read_excel(url, sheet_name='Sheet1')
# df_output = pd.DataFrame()

# Stress = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
# Parameters = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC'] # 計算的參數


# i = 0
# for para in Parameters:
    
#     df_split = df.loc[:,['Situation',para]]
    
#     df_output[i] = para
#     i+=1
    
#     for stress in Stress:
        
#         df_stress = df_split[df_split['Situation'] == stress]
        
#         stress_list = df_stress[para] 
#         stress_list = stress_list.reset_index(drop=True)
        
#         df_output[i] = stress_list
#         i+=1

# output_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRV原始Data/220714_HRV.xlsx'
# # df_output.to_excel(output_url, sheet_name='forsigmaPlot')
    


'''---------------平均HRV參數整理成SigmaPlot要的樣子 -散步圖畫每個人HRV參數與VAS問卷的相關性---------------------'''
# url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRVMean_VAS/220719_HRVMean.xlsx'
# allN = 24

# df = pd.read_excel(url)

# df_output = pd.DataFrame()

# Stress = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
# Parameters = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC'] # 計算的參數


# i = 0
# for para in Parameters:
    
#     df_output[i] = [para]*4 #標示參數
#     i+=1 #進到下一欄位
    
#     df_split = df.loc[:,['N', 'Situation','VAS',para]]
    
    
    
#     for n in range(2, allN+1):
#         if n ==7:
#             continue
        
#         df_oneN = df_split[df_split['N'] == n]
#         df_oneN = df_oneN.reset_index(drop=True)  #Series index要改為0為起始
        
#         oneN_vas = df_oneN['VAS']
#         df_output[i] = oneN_vas
#         i+=1 #進到下一欄位
        
#         oneN_para = df_oneN[para]
#         df_output[i] = oneN_para
#         i+=1 #進到下一欄位
 
# output_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRVMean_VAS/'
# df_output.to_excel(output_url+'220719_HRVMean_Temp.xlsx')


'''----------相關性整理-------'''
url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRVMean_VAS/HRVandVAS_Corr.xlsx'
allN_noerrordata = 22

df = pd.read_excel(url)
df_output = pd.DataFrame()

Parameters = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC'] # 計算的參數

i=0
for para in Parameters:
    
    df_output[i] = [para]*allN_noerrordata
    i+=1
    
    df_para = df[df['HRVPara'] == para]
    corr = df_para['Corr_HRVandVAS']
    corr = corr.reset_index(drop=True)
    
    df_output[i] = corr
    i+=1
    
df_output.to_excel('/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/HRVMean_VAS/corrforsigma.xlsx')



