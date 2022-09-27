#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:42:46 2022

@author: weien
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import def_getRpeak_main as getRpeak

fs=250
lta3_baseline = 0.9
lta3_magnification = 250

# get R peak parameters
medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.32*fs)*2  #草哥使用(0.3*fs)*2 #Patch=0.4*fs*2 LTA3=0.35*fs*2
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 20
highpass_fq = 10

# devide EMG parameters
qrs_range = int(0.32*fs)    
tpeak_range = int(0.2*fs) 

# Negative control (Baseline)

url_data = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/ClipSituation_eachN/N1/Baseline.csv'
df_data = pd.read_csv(url_data)
ecg_baseline=df_data['ECG']
ecg_baseline = ecg_baseline[3*30*fs:4*30*fs]
ecg_baseline_mV = (((np.array(ecg_baseline))*1.8/65535-lta3_baseline)/lta3_magnification)*1000

ecg_baseline_median, baseline_rpeak_index = getRpeak.getRpeak_shannon(ecg_baseline_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)


# Positive control (Shake shoulder)

url_data = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/ClipSituation_eachN/驗證資料/Shake_shoulder.csv'
df_data = pd.read_csv(url_data)
ecg_shake=df_data['ECG']
ecg_shake = ecg_shake[15*fs:45*fs]
ecg_shake_mV = (((np.array(ecg_shake))*1.8/65535-lta3_baseline)/lta3_magnification)*1000
ecg_shake_median, shake_rpeak_index  = getRpeak.getRpeak_shannon(ecg_shake_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)


x_time = np.linspace(0,30,30*fs)

plt.figure(figsize=(8, 12))

# Negative
plt.subplot(6,1,1)
plt.plot(x_time, ecg_baseline_mV, c='black')
plt.ylim(-2,2)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.xlim(0,30)
plt.title('Negative control', fontsize=16)
plt.yticks(fontsize=12)

plt.subplot(6,1,2)
plt.plot(x_time, ecg_baseline_median, c='black')
plt.scatter(np.array(baseline_rpeak_index)/fs, ecg_baseline_median[baseline_rpeak_index], c='steelblue', alpha=0.8)
plt.ylim(-2,2)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.xlim(0,30)
plt.yticks(fontsize=12)

# Positive
plt.subplot(6,1,4)
plt.plot(x_time, ecg_shake_median, c='black')
plt.ylim(-2,2)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.xlim(0,30)
plt.title('Positive control', fontsize=16)
plt.yticks(fontsize=12)


plt.subplot(6,1,5)
plt.plot(x_time, ecg_shake_mV, c='black')
plt.scatter(np.array(shake_rpeak_index)/fs, ecg_shake_median[shake_rpeak_index], c='steelblue', alpha=0.8)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.ylim(-2,2)
plt.xlim(0,30)
plt.yticks(fontsize=12)

_, emg_basline_list = getRpeak.deleteRTpeak(pd.Series(ecg_baseline_median),baseline_rpeak_index, qrs_range, tpeak_range) #刪除rtpeak並補0

_, emg_shake_list = getRpeak.deleteRTpeak(pd.Series(ecg_shake_median),shake_rpeak_index, qrs_range, tpeak_range) #刪除rtpeak並補0
# emg_shake_withoutZero = getRpeak.deleteZero(emg_shake_linearwithzero) 
# emg_shake_index = emg_shake_withoutZero.index.tolist()       


plt.subplot(6,1,3)
for i in range(len(emg_basline_list)):
    plt.plot(np.array(emg_basline_list[i].index.tolist())/fs, emg_basline_list[i], c='black')
plt.ylim(-2,2)
plt.xlim(0,30)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.subplot(6,1,6)
for i in range(len(emg_shake_list)):
    plt.plot(np.array(emg_shake_list[i].index.tolist())/fs, emg_shake_list[i], c='black')
plt.ylim(-2,2)
plt.xlim(0,30)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.tight_layout()

# plt.savefig('/Users/weien/Desktop/論文圖/Devide_ECMEMG.png', dpi=800)


