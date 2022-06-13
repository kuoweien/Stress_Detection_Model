#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:28:55 2022

@author: weien
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import def_Shannon as shannon
import def_passFilter as passFilter
import seaborn as sns
import def_passFilter as bandfilter

#計算skewness kurtosis等SQI

url = '/Users/weien/Desktop/ECG穿戴/HRV實驗/狗/Dataset/2110Nimo/scared.csv'
# url = '/Users/weien/Desktop/人體壓力測試/220426陳云/Stroop+Radio.csv'
situation = 'scared'

length_s = 10
index = 6
magnification = 500

plt.figure(figsize=(16,20))

for index in range(0,12):

    clip_start = index*(length_s*250) #scared 2500 #petted 10000
    clip_end = (index+1)*(length_s*250)  #scared 5000 #petted 12500
    
    df=pd.read_csv(url).iloc[clip_start:clip_end]
    rawdata = df[situation].reset_index(drop = True)
    rawdata_mV = ((rawdata*(1.8/65535)-0.9)/magnification)*1000  #轉換為電壓
    median_filter_data = shannon.medfilt(np.array(rawdata_mV), 61)
    rawdata_mV_medianfilter = rawdata_mV - median_filter_data
    lowpass_data = bandfilter.lowPassFilter(30, rawdata_mV_medianfilter)  #低通
    bandfilter_data = bandfilter.highPassFilter(5, lowpass_data)    #高通
    
    signal = bandfilter_data
    noise = rawdata_mV-bandfilter_data
    
    signal_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    rmsSQI = signal_rms/noise_rms
    print('RMS')
    print('index='+str(index))
    print(round(signal_rms,3))
    print(round(noise_rms,3))
    print(round(rmsSQI,3))
    print('')
       
    #Signal Quality Indices
    y = rawdata_mV  
    mean = np.mean(y)
    std = np.std(y)
    n = len(y)    
    #SQI
    sqi_snr =  (std**2) / (np.std(np.abs(y))**2)
    sqi_skew = np.sum(((y-mean)/std)**3)/n
    sqi_snr = 10*math.log10(sqi_snr)
    sqi_kur = np.sum(((y-mean)/std)**4)/n
    
    plt.subplot(6,2,index+1)
    plt.plot(rawdata_mV, color='black')
    plt.ylim(-1,1)
    plt.title('rmsSQI='+str(round(rmsSQI,3)))
    plt.tight_layout()

    

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(rawdata_mV, color='black')
plt.ylim(-1,1)

plt.subplot(3,1,2)
plt.plot(signal, color='black')
plt.ylim(-1,1)

plt.subplot(3,1,3)
plt.plot(noise, color='black')
plt.ylim(-1,1)

plt.tight_layout()

plt.title('Measure Noise')


#Signal Quality Indices
y = rawdata_mV

mean = np.mean(y)
std = np.std(y)
n = len(y)


#SQI
sqi_snr =  (std**2) / (np.std(np.abs(y))**2)
sqi_skew = np.sum(((y-mean)/std)**3)/n
sqi_snr = 10*math.log10(sqi_snr)
sqi_kur = np.sum(((y-mean)/std)**4)/n
# sqi_hos = np.abs(sqi_skew) * (sqi_kur/5)



print('sqi_skew: {}'.format(round(sqi_skew,3)))
print('sqi_snr: {}'.format(round(sqi_snr,3)))
print('sqi_kur: {}'.format(round(sqi_kur,3)))


# #畫直方圖
# sns.set_style("white")
# plt.figure(figsize=(10,7), dpi= 80)
# hist_binvalue = []
# sns.histplot(y, alpha=0.6, linewidth=2, color='black', kde=True, label='RR Interval') #kde畫常態線
# plt.grid(True)
# plt.xlim(-1,1)
# plt.ylim(0,200)
# plt.legend()

#%%noise-measure SQI

#方法二

'''
temp_data = bandfilter.lowPassFilter(30, rawdata_mV)  #低通
signal = bandfilter.highPassFilter(5, temp_data)    #高通
# signal = normalize_data
noise = (rawdata_mV - signal)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(signal, color='black')

plt.subplot(2,1,2)
plt.plot(noise, color='black')

plt.title('Measure Noise')
'''  


