#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:21:18 2021

@author: weien
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import signal
import numpy as np
import def_dataDecode as dataDecode

# 讀Raw檔
def openRawFile(filename):
    
    with open(filename,'rb') as f:
        rawtxt=f.read()
    rawlist=dataDecode.dataDecode.rawdataDecode(rawtxt)
    
    rawdata=rawlist[0]
    ecg_rawdata=rawdata[0]#取原始資料channel1 (234是加速度)
    
    frquency=rawlist[1]
    ecg_fq=frquency[0]#頻率 狗狗的是250Hz
    
    updatetime_str=rawlist[2].split(' ')[1]#抓上傳時間
    update_datetime = datetime.datetime.strptime(updatetime_str, '%H:%M:%S')#上傳時間字串轉datetime type

    return ecg_rawdata,ecg_fq,update_datetime


#輸入時間，取出時間斷之rawdata
def inputtimetoClipRawdata(ecgrawlist,frequency, updatetime, clip_start_time, clip_end_time, ):
    
    start_datetime = datetime.datetime.strptime(clip_start_time, '%H:%M:%S')
    end_datetime = datetime.datetime.strptime(clip_end_time, '%H:%M:%S')
    

    start_time_index = start_datetime-updatetime
    end_time_index = end_datetime-updatetime
    
    start_index = start_time_index.total_seconds()
    end_index = end_time_index.total_seconds()
        
    rawdata_list = ecgrawlist[int(start_index*frequency):int(end_index*frequency)]
    
    return rawdata_list


# 計算補數
def get_data_complement(signal): # 2的補數 有的設備因為有存正負號
    np_signal=np.array(signal)
    for i in range(len(np_signal)):
        if np_signal[i]<32768:
            np_signal[i]+=65535
    np_signal-=65535      
    return np_signal 

'''
# 讀Rawdata並取片段
def readRawdata(url, filename, cliptime_start_str, cliptime_end_str): #不需要補數 #若不需要切時間點，則cliptime_start_str='', cliptime_end_str=''
    
    ecg_rawdata, ecg_fq, updatetime = openRawFile(url+'/'+filename) #minus_V = -0.9
    if len(cliptime_start_str) != 0 and len(cliptime_end_str) != 0:
        ecg_rawdata = getConditionRawdata(cliptime_start_str, cliptime_end_str, ecg_rawdata, ecg_fq, updatetime)
    
    return ecg_rawdata, ecg_fq, updatetime

# 讀需要補數的Data
def readComplementRawdata(url, filename, cliptime_start_str, cliptime_end_str): #需要對Rawdata做補數 (硬體的關係) #若沒有要切時間點，則cliptime_start_str='', cliptime_end_str=''
    
    ecg_rawdata, ecg_fq, updatetime = openRawFile(url+'/'+filename) #minus_V = -0.9
    if len(cliptime_start_str) != 0 and len(cliptime_end_str) != 0:
        ecg_rawdata = getConditionRawdata(cliptime_start_str, cliptime_end_str, ecg_rawdata, ecg_fq, updatetime)
    # ecg_rawdata = get_data_complement(ecg_rawdata) 
    np_signal=np.array(ecg_rawdata)
    for i in range(len(np_signal)):
        if np_signal[i]<32768:
            np_signal[i]+=65535
    np_signal-=65535  
    # ecg_rawdata = ((np.array(ecg_rawdata)*(1.8/65535)-0.0)/120)*1000  #轉換為電壓

    return np_signal, ecg_fq, updatetime
'''

def outputDatatoCSV(column_name, data, outputurl):
    df = pd.DataFrame({column_name: data})
    df.to_csv(outputurl)
    
def ecgEpochScore(ecg):
    rms = np.sqrt(np.mean(np.array(ecg)**2))
    noise_score = ((rms-753.632)/(32767-753.632))*100
    
    return noise_score

def exporttomV_lta3(ecg):
    ecg_mV = ((ecg*(1.8/65535)-0.9)/250)*1000
    return ecg_mV
    
def exporttomV_patch(ecg):
    ecg_mV = ((ecg*(1.8/65535)-0.0)/120)*1000
    return ecg_mV
    




    
    

    



