#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:46:14 2022

@author: weien
"""
# import def_readandget_Rawdata as getRawdata
import matplotlib.pyplot as plt
import pandas as pd
# import def_readandget_Rawdata
import numpy as np
import datetime
import Library.def_dataDecode as dataDecode

# open_RawFile
def open_rawfile(filename):
    with open(filename, 'rb') as f:
        rawtxt = f.read()
    rawlist = dataDecode.dataDecode.rawdataDecode(rawtxt)
    rawdata = rawlist[0]
    ecg_rawdata = rawdata[0]  # 取原始資料channel1 (channel 234是加速度)
    frquency = rawlist[1]
    ecg_fq = frquency[0]  # 取頻率 狗狗的是250Hz
    updatetime_str = rawlist[2].split(' ')[1]  # 抓上傳時間
    update_datetime = datetime.datetime.strptime(updatetime_str, '%H:%M:%S')  # 上傳時間字串轉datetime type

    return ecg_rawdata, ecg_fq, update_datetime


# 輸入時間，取出時間斷之rawdata
# inputtime_toClipRawdata
def clip_rawdata_byinputtime(ecgrawlist, frequency, updatetime, clip_start_time, clip_end_time):
    start_datetime = datetime.datetime.strptime(clip_start_time, '%H:%M:%S')
    end_datetime = datetime.datetime.strptime(clip_end_time, '%H:%M:%S')

    start_time_index = start_datetime - updatetime
    end_time_index = end_datetime - updatetime

    start_index = start_time_index.total_seconds()
    end_index = end_time_index.total_seconds()

    rawdata_list = ecgrawlist[int(start_index * frequency):int(end_index * frequency)]

    return rawdata_list

# 計算補數
def get_data_complement(signal):  # 2的補數 有的設備因為有存正負號
    np_signal = np.array(signal)
    for i in range(len(np_signal)):
        if np_signal[i] < 32768:
            np_signal[i] += 65535
    np_signal -= 65535
    return np_signal

def export_tomv_lta3(ecg):
    ecg_mv = ((ecg * (1.8 / 65535) - 0.9) / 250) * 1000
    return ecg_mv

def export_tomv_patch(ecg):
    ecg_mv = ((ecg * (1.8 / 65535) - 0.0) / 120) * 1000
    return ecg_mv


if __name__ == '__main__':

    # -------------------

    raw_url = 'Data/Rawdata/LTA3_Rawdata/'  # 調整要用LTA3 or Patch設備
    output_url = 'Data/ClipSituation_CSVfile/'

    n = 49
    filename = '49-220830a.241'

    is_ecg_upsidedown = False  # 是否顛倒ECG
    is_ecg_complement = False  # 是否補數

    # ----------------

    '''------Unzip Raw File-----------'''
    # Adjust the different between devices time and standard time (>0 means devices time is faster)
    adjusttime_hr = 0
    adjusttime_min = 0
    adjusttime_sec = 0

    rawfile_url = raw_url+filename  # unzip url
    ecg_raw, fs, updatetime = open_rawfile(rawfile_url)  # unzip raw file
    if is_ecg_complement:
        ecg_raw = get_data_complement(ecg_raw)
    if is_ecg_upsidedown:
        ecg_raw = ecg_raw * -1

    # Plot raw data
    # plt.figure(figsize=(12,6))
    # plt.plot(np.linspace(0, 200, len(ecg_mV)), ecg_mV ,'black')
    # plt.xlim(0,200)
    # plt.ylim(-1,1.5)
    # plt.ylabel('ECG (mV)', fontsize=14)
    # plt.xlabel('Time (s)', fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)


    '''----------Save csv file of each situation data------------'''
    # Read protocl time
    time_url = 'Data/BasicData/Protocal_Time.xlsx'
    df_time = pd.read_excel(time_url)
    df_onedata = df_time[df_time['N'] == n]

    # Human
    columns = ['Baseline', 'Stroop', 'Baseline_after_stroop', 'Arithmetic', 'Baseline_after_Arithmetic', 'Speech', 'Speech_3m', 'Baseline_after_speech']
    # Dog
    # columns = ['Baseline', 'Touch', 'Baseline_after_touch', 'Scared', 'Baseline_after_Scared', 'Play', 'Baseline_after_Play', 'Seperate', 'Baseline_after_Seperate', 'Eat', 'Baseline_after_Eat']

    #  Read situations
    for i in columns:
        time = ((df_onedata[i]).tolist())[0]
        if str(time) == 'nan':
            print('This parrticipant (N{}) has no time data'.format(n))
            continue

        adjust_totalseconds = adjusttime_hr*3600 + adjusttime_min*60 + adjusttime_sec

        start_time = time.split('-')[0]
        start_time_datetime = datetime.datetime.strptime(start_time, '%H:%M:%S')
        start_time = (start_time_datetime+datetime.timedelta(seconds=adjust_totalseconds)).strftime('%H:%M:%S')
        end_time = time.split('-')[1]
        end_time_datetime = datetime.datetime.strptime(end_time, '%H:%M:%S')
        end_time = (end_time_datetime+datetime.timedelta(seconds=adjust_totalseconds)).strftime('%H:%M:%S')

        ecg_condition = clip_rawdata_byinputtime(ecg_raw, fs, updatetime, start_time, end_time)

        df = pd.DataFrame({'ECG': ecg_condition})
        outputfile_url = output_url+'N{}/{}.csv'.format(n, i)
        df.to_csv(outputfile_url)

        # plt.figure(figsize=(12, 2))
        # plt.plot(ecg_condition, 'black')
        # plt.title(i)
        # plt.show()