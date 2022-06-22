#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:46:14 2022

@author: weien
"""
import def_readandget_Rawdata as getRawdata
import matplotlib.pyplot as plt
import pandas as pd
import def_readandget_Rawdata


#%% 解碼Raw檔

raw_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/Dataset/Rawdata/220517-4紹芬/LTA3/'
filename = '220517e.241'
n = 4 #受試者代碼

rawfile_url = raw_url+filename #壓縮檔的url
time_url = 'Data/Protocal_Time.xlsx'

ecg_raw, fs, updatetime = def_readandget_Rawdata.openRawFile(rawfile_url) #解Raw檔
# ecg_raw = def_readandget_Rawdata.get_data_complement(ecg_raw) #取補數
# ecg_raw = ecg_raw*-1

# plt.figure(figsize=(14,2))
# plt.plot(ecg_raw,'black')

#讀protocl時間
df_time = pd.read_excel(time_url)
df_onedata = df_time[df_time['N']==n ]

#將data分成不同csv檔儲存
columns = ['Baseline', 'Stroop', 'Baseline_after_stroop', 'Arithmetic', 'Baseline_after_Arithmetic', 'Speech', 'Baseline_after_speech']
# columns = ['Stroop']

for i in columns:
    time = ((df_onedata[i]).tolist())[0]
    start_time = time.split('-')[0]
    end_time = time.split('-')[1]
    ecg_condition = def_readandget_Rawdata.inputtimetoClipRawdata(ecg_raw, fs, updatetime, start_time ,end_time) #取時間
    
    df = pd.DataFrame({'ECG':ecg_condition})
    df.to_csv('Data/N{}/{}.csv'.format(str(n), i))
    
    plt.figure(figsize=(12,2))
    plt.plot(ecg_condition ,'black')



