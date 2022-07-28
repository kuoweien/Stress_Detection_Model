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

import datetime



'''------------------------解碼Raw檔---------------------------'''

raw_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/Rawdata/LTA3_Rawdata/'
# filename = '12-220624A.RAW'
filename = '28-220725.241'
n = 28 #受試者代碼

# 調整儀器與標準時間的差異 正值表示儀器較標準時間快
adjusttime_hr = 0
adjusttime_min = 0
adjusttime_sec = 0

rawfile_url = raw_url+filename #壓縮檔的url
ecg_raw, fs, updatetime = def_readandget_Rawdata.openRawFile(rawfile_url) #解Raw檔
# ecg_raw = def_readandget_Rawdata.get_data_complement(ecg_raw) #取補數

# 印原始圖
# plt.figure(figsize=(12,4))
# plt.plot(ecg_raw ,'black')
# plt.title('RawECG')
# ecg_raw = ecg_raw*-1


'''------------------------將data分成不同csv檔儲存---------------------------'''
#讀protocl時間
time_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/問卷&流程資料/Protocal_Time.xlsx'
df_time = pd.read_excel(time_url)
df_onedata = df_time[df_time['N']==n ]



# 人的
columns = ['Baseline', 'Stroop', 'Baseline_after_stroop', 'Arithmetic', 'Baseline_after_Arithmetic', 'Speech', 'Speech_3m', 'Baseline_after_speech']
# 狗的
# columns = ['Baseline', 'Touch', 'Baseline_after_touch', 'Scared', 'Baseline_after_Scared', 'Play', 'Baseline_after_Play', 'Seperate', 'Baseline_after_Seperate', 'Eat', 'Baseline_after_Eat']

for i in columns:

    time = ((df_onedata[i]).tolist())[0]
    if str(time) == 'nan':
        continue
    
    adjust_totalseconds = adjusttime_hr*3600 + adjusttime_min*60 + adjusttime_sec

    start_time = time.split('-')[0]
    start_time_datetime = datetime.datetime.strptime(start_time, '%H:%M:%S')
    start_time = (start_time_datetime+datetime.timedelta(seconds = adjust_totalseconds)).strftime('%H:%M:%S')
    

    
    end_time = time.split('-')[1]
    end_time_datetime = datetime.datetime.strptime(end_time, '%H:%M:%S')
    end_time = (end_time_datetime+datetime.timedelta(seconds = adjust_totalseconds)).strftime('%H:%M:%S')
    
    
    ecg_condition = def_readandget_Rawdata.inputtimetoClipRawdata(ecg_raw, fs, updatetime, start_time ,end_time) #取時間
    
    df = pd.DataFrame({'ECG':ecg_condition})
    outputurl = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/ClipSituation_eachN/N{}/{}.csv'.format(n, i)
    df.to_csv(outputurl)
    
    plt.figure(figsize=(12,2))
    plt.plot(ecg_condition ,'black')
    plt.title(i)






