#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:03:30 2022

@author: weien
"""
import biosppy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyhrv
import pyhrv.frequency_domain as fd
import def_getRpeak_main as getRpeak

t = np.linspace( 0, 1, 1000, endpoint = False ) # 定義時間陣列
x = np.sin( 2 * np.pi * 5 * t )                 # 產生弦波 #5是HZ

plt.figure()
plt.plot( t, x )                                # 繪圖
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -1.2, 1.2 ] )



ecg_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/ClipSituation_eachN/N27/'
filename_baseline = 'Speech.csv'
df_baseline1 = pd.read_csv(ecg_url+filename_baseline)
signal = df_baseline1['ECG']
# signal = signal[0:7500]
# signal, rpeaks = biosppy.signals.ecg.ecg(signal, show=False)[1:3]


fs = 250
# 抓Rpeak的參數
medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.32*fs)*2  #草哥使用(0.3*fs)*2 #Patch=0.4*fs*2 LTA3=0.35*fs*2
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 20
highpass_fq = 10



median_ecg, rpeakindex = getRpeak.getRpeak_shannon(signal, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
rrinterval = np.diff(rpeakindex)
rri = rrinterval/(fs/1000) #RRI index點數要換算回ms (%fs，1000是因為要換算成毫秒)



# Load NNI sample series
pyhrv.utils.load_sample_nni()
fbands = {'ulf': (0.0, 0.1), 'vlf': (0.1, 0.2), 'lf': (0.2, 0.3), 'hf': (0.3, 0.4)}
# Compute the PSD and frequency domain parameters using the NNI series
result = fd.welch_psd(rri, fbands=fbands)

# Access peak frequencies using the key 'fft_peak'
print(result['fft_peak'])




