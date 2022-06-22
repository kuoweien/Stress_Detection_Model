#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:37:48 2022

@author: weien
"""

import def_getRpeak_main as getRpeak
import matplotlib.pyplot as plt
import numpy as np
import def_readandget_Rawdata as getRawdata
import pandas as pd

fs = 500

# ecg, fs, time = getRawdata.openRawFile('/Users/weien/Desktop/kylab專案/勝杰的心肌梗塞大鼠分析/MIraw檔/MI前/wc1.RAW')

df = pd.read_csv('/Users/weien/Desktop/kylab專案/勝杰的心肌梗塞大鼠分析/MIraw檔/MI前/ECG_data.csv')
ecg = df['ECG']

rawdata_clip = ecg[int(0*fs) : int(30*fs)]

# patch_baseline = 0
# lta3_baseline = 0.9
# patch_magnification = 120
# lta3_magnification = 250

medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 1.25 
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.2*fs)*2  #草哥使用(0.3*fs)*2
rpeak_close_range = 0.0*fs #0.1*fs
lowpass_fq = 300
highpass_fq = 2


median_ecg, rpeakindex = getRpeak.getRpeak_shannon(rawdata_clip, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
# median_ecg, rpeakindex = getRpeak.getRpeak_pantompskin(rawdata_clip, fs, medianfilter_size, lowpass_fq, highpass_fq)


plt.figure(figsize=(14,2))
plt.plot(median_ecg, 'black')
# plt.ylim(-1.5, 1.8)
plt.scatter(np.array(rpeakindex), median_ecg[rpeakindex], alpha=0.5, c='r')