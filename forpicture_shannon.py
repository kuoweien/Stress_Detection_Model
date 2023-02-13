#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 19:51:44 2022

@author: weien
"""

import pandas as pd
import matplotlib.pyplot as plt
import def_getRpeak_main as def_getRpeak
import numpy as np

fs = 250
epoch = 30
ecg_clip_start = 8000
ecg_clip_end = ecg_clip_start+epoch*fs


# Noisy 
checknoiseThreshold = 20 # Delete noisy threshold from 2 seconds epoch (Patch=0.419 LTA3=2.210)

# Get Rpeak parameters
medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.32*fs)*2  #草哥使用(0.3*fs)*2 #Patch=0.4*fs*2 LTA3=0.35*fs*2
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 20
highpass_fq = 10



ecg_url = '/Users/weien/Documents/GitHub/Stress_Detection_python/Data/ClipSituation_CSVfile/N37/Baseline.csv'
df = pd.read_csv(ecg_url)
ecg = df['ECG'][ecg_clip_start:ecg_clip_end]

# R peak algorithm
median_filter_data = def_getRpeak.medfilt(np.array(ecg), medianfilter_size)
median_ecg = ecg-median_filter_data
lowpass_data = def_getRpeak.lowPassFilter(20,fs,median_ecg)  # Lowpass filter
bandfilter_data = def_getRpeak.highPassFilter(10,fs,lowpass_data)    # Highrpass filter
dy_data = def_getRpeak.defivative(bandfilter_data) # Derivative
normalize_data = dy_data/np.max(dy_data) # Normalize
see_data = (-1)*(normalize_data**2)*np.log((normalize_data**2)) # Shannon envelop
gaussian_data = def_getRpeak.gaussian_filter(see_data, sigma=gaussian_filter_sigma)
hibert_data = np.imag(def_getRpeak.hilbert(gaussian_data))  # Hilbert tranform
movingaverage_data = def_getRpeak.movingaverage(hibert_data, moving_average_ms, fs) # Moving average
hibertmoving_data = hibert_data-movingaverage_data
zero_index = def_getRpeak.findZeroCross(hibertmoving_data)  # Positive zero crossing point
zero_shift_index = def_getRpeak.shiftArray(zero_index, final_shift) # Shift the results

#Decision Rules
detect_Rpeak_index, _   = def_getRpeak.findMaxvalue(median_ecg, zero_shift_index, detectR_maxvalue_range)  
re_detect_Rpeak_index = def_getRpeak.deleteCloseRpeak(detect_Rpeak_index, rpeak_close_range) 


x_time = np.linspace(0,epoch,epoch*fs)

plt_len = 8
plt.figure(figsize=(8,10))
plt.subplot(plt_len, 1, 1)
plt.plot(x_time, np.array(ecg)-np.mean(ecg), color='black')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xlim(0,epoch)
plt.ylim(-10000, 10000)
plt.yticks(fontsize=12)
plt.title('ECG', fontsize=16)

plt.subplot(plt_len, 1, 2)
plt.plot(x_time, median_ecg, color='black')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xlim(0,epoch)
plt.ylim(-10000, 10000)
plt.yticks(fontsize=12)
plt.title('Median filter', fontsize=16)

plt.subplot(plt_len, 1, 3)
plt.plot(x_time, bandfilter_data, color='black')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xlim(0,epoch)
plt.ylim(-2000,2000)
plt.yticks(fontsize=12)
plt.title('Bandpass filter (10-20Hz)', fontsize=16)

plt.subplot(plt_len, 1, 4)
plt.plot(x_time, dy_data, color='black')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xlim(0,epoch)
plt.ylim(-1000, 1000)
plt.yticks(fontsize=12)
plt.title('Differentiator', fontsize=16)

plt.subplot(plt_len, 1, 5)
plt.plot(x_time, see_data, color='black')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.xlim(0,epoch)
plt.ylim(-0.0,0.5)
plt.yticks(fontsize=12)
plt.title('SEE estimator', fontsize=16)

plt.subplot(plt_len, 1, 6)
plt.plot(x_time, gaussian_data, color='black')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.xlim(0,epoch)
plt.ylim(-0.0,0.3)
plt.yticks(fontsize=12)
plt.title('Gaussian smoothing', fontsize=16)

plt.subplot(plt_len, 1, 7)
plt.plot(x_time, hibertmoving_data, color='black')
plt.scatter(np.array(zero_index)/fs, hibertmoving_data[zero_index], alpha=0.5, color='black')
plt.axhline(y=0, xmin=0, xmax=30, color='grey')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.xlim(0,epoch)
plt.ylim(-0.2, 0.2)
plt.yticks(fontsize=12)
plt.title('Hilbert transformer', fontsize=16)

plt.subplot(plt_len, 1, 8)
plt.plot(x_time, median_ecg, color='black')
plt.scatter((np.array(re_detect_Rpeak_index)-8000)/fs, median_ecg[re_detect_Rpeak_index], alpha=0.8, color='steelblue')
plt.xlim(0,epoch)
plt.ylim(-10000, 10000)
plt.title('R-peaks', fontsize=16)
plt.xlabel('Time (s)', fontsize=16)
ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.tight_layout()

plt.savefig('SEE.png', dpi=800)






