#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:16:34 2022

@author: weien
"""

import pandas as pd


'''---------------Raw ECG EMG Features Mean tidy for SigmaPlot---------------------'''

# url = 'Data/220912_FeaturesMean.xlsx'
# df = pd.read_excel(url, sheet_name='Sheet1')
# df_output = pd.DataFrame()

# Stress = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
# Parameters = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 
#               'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC',
#               'TP', 'HF', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']


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

# output_url = 'Data/forsigmaPlot.xlsx'
# df_output.to_excel(output_url, sheet_name='forsigmaPlot')
    


'''---------------Scatter plot to show the correlation between ECG/EMG features and VAS of Stress---------------------'''
# url = 'Data/220912_FeaturesMean.xlsx'
# last_N = 51

# df = pd.read_excel(url)

# df_output = pd.DataFrame()


# vas_url = 'Data/Questionnaire.xlsx'
# df_vas = pd.read_excel(vas_url, sheet_name = 'forAnalyze')
# vas = df_vas['VAS']

# df['VAS'] = vas

# Stress = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
# Parameters = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 
#         'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC',
#         'TP', 'HF', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']

# i = 0
# for para in Parameters:
    
#     df_output[i] = [para]*4 #標示參數
#     i+=1 #進到下一欄位
    
#     df_split = df.loc[:,['N', 'Situation','VAS',para]]
    
    
    
#     for n in range(1, last_N+1):
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
 
# # df_output.to_excel('Data/220912_forsigma.xlsx')


'''----------相關性整理 成SigmaPlot要的-------'''

# url = 'Data/220912_FeaturesandVAS_Corr.xlsx'

# allN_noerrordata = 51

# df = pd.read_excel(url)
# df_output = pd.DataFrame()


# Parameters = ['Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 
#         'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC',
#         'TP', 'HF', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']
            
# i=0
# for para in Parameters:
    
#     df_output[i] = [para]*allN_noerrordata
#     i+=1
    
#     df_para = df[df['HRVPara'] == para]
#     corr = df_para['Corr_HRVandVAS']
#     corr = corr.reset_index(drop=True)
    
#     df_output[i] = corr
#     i+=1
    
# df_output.to_excel('Data/corrforsigma.xlsx')



