#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:56:29 2022

@author: weien
"""
import numpy as np




#%%
'''----------2秒為一個Epoch，確認是否為雜訊-----------'''

#標準化
def normalize(data, lowest_rms, highest_rms):
    
    point = ((data-lowest_rms)/(highest_rms-lowest_rms))*100
    
    return point

#計算雜訊分數
def ecgEpochScore(ecg):
    rms = np.sqrt(np.mean(np.array(ecg)**2))
    noise_score = normalize(rms, 1186.45, 32767)
    return noise_score


def splitEpochandisCleanSignal(ecg_raw, fs, sqi_threshold):
    ecg_raw = ecg_raw-np.median(ecg_raw)
    for i in range(0, len(ecg_raw)):
        clip_ecg = ecg_raw[i*fs : (i+1)*fs]
        if ecgEpochScore(clip_ecg) > sqi_threshold:
            ecg_raw[i*fs : (i+1)*fs] = 0
    return ecg_raw


#%%
'''--------------給予每個Rpeak分數-------------'''

# 計算每個Rpeak到下一個Rpeak中的Root Mean Square
def measureRPoint(rr_index,ecg_raw):
    output = []
    for i in range(len(rr_index)):        
        
        if i == len(rr_index)-1:  
            
            y = ecg_raw[rr_index[i]:]
            
        else:
            y = ecg_raw[rr_index[i]:rr_index[i+1]]
            
        rms = np.sqrt(np.mean(np.array(y)**2))
        output.append(round(rms,3)) #新增R點的index值 以及計算到下一個R點前的RMS
          
    return output #印出每個點的RMS
    
# rr_index = [2,5,8,10]
# ecg_raw = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# rms = measureRPoint(rr_index, ecg_raw)


