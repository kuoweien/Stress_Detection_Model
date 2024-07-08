#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:56:29 2022

@author: weien
"""
import numpy as np


#%%
'''----------2秒為一個Epoch，確認是否為雜訊-----------'''

# 標準化
def normalize(data, lowest_rms, highest_rms):
    point = ((data-lowest_rms)/(highest_rms-lowest_rms))*100
    return point

# 計算雜訊分數
def cal_noise_score(ecg):
    rms = np.sqrt(np.mean(np.array(ecg)**2))
    # Define lowest_rms and highest_rms by devices having cleas ECG and full noisy
    noise_score = normalize(rms, 1186.45, 32767)
    return noise_score

def replace_noisy_ecg_tozero(ecg_raw, fs, sqi_threshold):
    ecg_raw = ecg_raw-np.median(ecg_raw)
    for i in range(0, len(ecg_raw)//fs-2, 2):  #220728修改
        clip_ecg = ecg_raw[i*fs:(i+2)*fs]
        if cal_noise_score(clip_ecg) > sqi_threshold:
            ecg_raw[i*fs: (i+1)*fs] = 0
    return ecg_raw


#%%
'''--------------給予每個Rpeak分數-------------'''

#  計算每個Rpeak到下一個Rpeak中的Root Mean Square
def measure_rms_by_peaks(rr_index, ecg_raw):

    rms_rpeaks = []
    for i in range(len(rr_index)):
        if i == len(rr_index)-1:
            y = ecg_raw[rr_index[i]:]
        else:
            y = ecg_raw[rr_index[i]:rr_index[i+1]]
            
        rms = np.sqrt(np.mean(np.array(y)**2))
        rms_rpeaks.append(round(rms, 3))  # 新增R點的index值 以及計算到下一個R點前的RMS
          
    return rms_rpeaks  # 印出每個R波的RMS
    
# rr_index = [2,5,8,10]
# ecg_raw = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# rms = measureRPoint(rr_index, ecg_raw)