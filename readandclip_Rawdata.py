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
import numpy as np
import datetime


'''------------------------Unzip Raw File---------------------------'''

raw_url = 'Data/Rawdata/LTA3_Rawdata/'
filename = '49-220830a.241'
n = 49 # Participant's number


# Adjust the different between devices time and standard time (>0 means devices time is faster)
adjusttime_hr = 0
adjusttime_min = 0
adjusttime_sec = 0

rawfile_url = raw_url+filename # unzip url
ecg_raw, fs, updatetime = getRawdata.openRawFile(rawfile_url) # unzip raw file
# ecg_raw = getRawdata.get_data_complement(ecg_raw) # get complement data


# print raw data plot
# plt.figure(figsize=(12,6))
# plt.plot(np.linspace(0, 200, len(ecg_mV)), ecg_mV ,'black')
# plt.xlim(0,200)
# plt.ylim(-1,1.5)
# plt.ylabel('ECG (mV)', fontsize=14)
# plt.xlabel('Time (s)', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# ecg_raw = ecg_raw*-1


'''------------------------save csv file of each situation data---------------------------'''

# read protocl time
time_url = 'Data/BasicData/Protocal_Time.xlsx'
df_time = pd.read_excel(time_url)
df_onedata = df_time[df_time['N']==n ]



# Human
columns = ['Baseline', 'Stroop', 'Baseline_after_stroop', 'Arithmetic', 'Baseline_after_Arithmetic', 'Speech', 'Speech_3m', 'Baseline_after_speech']
# Dog
# columns = ['Baseline', 'Touch', 'Baseline_after_touch', 'Scared', 'Baseline_after_Scared', 'Play', 'Baseline_after_Play', 'Seperate', 'Baseline_after_Seperate', 'Eat', 'Baseline_after_Eat']


for i in columns:

    time = ((df_onedata[i]).tolist())[0]
    if str(time) == 'nan':
        print('This parrticipant (N{}) has no time data'.format(n))
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
    outputurl = 'Data/ClipSituation_CSVfile/N{}/{}.csv'.format(n, i)
    df.to_csv(outputurl)
    
    plt.figure(figsize=(12,2))
    plt.plot(ecg_condition ,'black')
    plt.title(i)


