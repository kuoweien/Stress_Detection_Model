#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:43:30 2022

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

url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/Dataset/Rawdata/220510-1郭葦珊/LTA3/Stroop.csv'
df = pd.read_csv(url)

ecg = df['Stroop']
ecg = ecg[100:5100]

ecg_mV = (((np.array(ecg))*1.8/65535-0.9)/250)*1000

fs = 250
medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5 
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.3*fs)*2  #草哥使用(0.3*fs)*2
detectR_minvalue_range = (0.5*fs)*2 
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 15
highpass_fq = 5

# median_ecg, rpeakindex = shannon.getRpeak_shannon(ecg_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
median_ecg, rpeakindex = shannon.getRpeak_pantompskin(ecg_mV, fs, medianfilter_size, lowpass_fq, highpass_fq)

plt.plot(median_ecg, 'black')
plt.scatter(np.array(rpeakindex), median_ecg[rpeakindex], alpha=0.5, c='r')
