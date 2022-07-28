#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:01:57 2022

@author: weien
"""

import pandas as pd
import numpy as np
import def_getRpeak_main as getRpeak
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import scipy.fft
import def_readandget_Rawdata

def interpolate(raw_signal,n):   #signal為原始訊號 n為要插入產生為多少長度之訊號

    x = np.linspace(0, len(raw_signal)-1, num=len(raw_signal), endpoint=True)
    f = interp1d(x, raw_signal, kind='cubic')
    xnew = np.linspace(0, len(raw_signal)-1, num=n, endpoint=True)  
    
    return f(xnew)

def window_function(window_len,window_type='hanning'):
    if window_type=='hanning':
        return np.hanning(window_len)
    elif window_type=='hamming':
        return np.hamming(window_len)

def fft_power(input_signal,sampling_rate,window_type):
    w=window_function(len(input_signal))
    window_coherent_amplification=sum(w)/len(w)
    y_f = np.fft.fft(input_signal*w)
    y_f_Real= 2.0/len(input_signal) * np.abs(y_f[:len(input_signal)//2])/window_coherent_amplification
    x_f = np.linspace(0.0, 1.0/(2.0*1/sampling_rate), len(input_signal)//2)        
    return y_f_Real,x_f

def medfilt (x, k): #基線飄移 x是訊號 k是摺照大小
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)  #做完之後還要再用原始訊號減此值
'''
t = np.linspace( 0, 10, 1000, endpoint = False ) # 定義時間陣列
x = np.sin( 2 * np.pi * 0.2 * t ) 
y_f_Real, x_f = fft_power(x, 100, 'hamming')

plt.figure()
plt.subplot(211)
plt.plot(t,x)
plt.subplot(212)
plt.plot(x_f,y_f_Real)
'''


lta3_baseline = 0.9
lta3_magnification = 250
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


epoch_len = 150 # seconds
rr_resample_rate = 1
slidingwidow_len = 30 #seconds

minute_to_second = 60


ecg_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/ClipSituation_eachN/N27/'
filename_baseline = 'Baseline.csv'
filename_stroop = 'Stroop.csv'
filename_b2 = 'Baseline_after_stroop.csv'
filename_arithmetic = 'Arithmetic.csv'
filename_b3 =  'Baseline_after_Arithmetic.csv'
filename_speech = 'Speech.csv'
filename_b4 = 'Baseline_after_speech.csv'


df_baseline1 = pd.read_csv(ecg_url+filename_baseline)
df_stroop = pd.read_csv(ecg_url+filename_stroop)
df_baseline2 = pd.read_csv(ecg_url+filename_b2)
df_arithmetic = pd.read_csv(ecg_url+filename_arithmetic)
df_baseline3= pd.read_csv(ecg_url+filename_b3)
df_speech = pd.read_csv(ecg_url+filename_speech)
df_baseline4 = pd.read_csv(ecg_url+filename_b4)


ecg_baseline1 = df_baseline1['ECG']
ecg_stroop = df_stroop['ECG']
ecg_baseline2 = df_baseline2['ECG']
ecg_arithmetic = df_arithmetic['ECG']
ecg_baseline3 = df_baseline3['ECG']
ecg_speech = df_speech['ECG']
ecg_baseline4 = df_baseline4['ECG']

ecg_raw = (((((ecg_baseline1.append(ecg_stroop))
            .append(ecg_baseline2)).append(ecg_arithmetic))
            .append(ecg_baseline3)).append(ecg_speech)).append(ecg_baseline4)


ecg_mV = (((np.array(ecg_raw))*1.8/65535-lta3_baseline)/lta3_magnification)*1000


tp_HRV = []
hf_HRV = []
lf_HRV = []
vlf_HRV = []
ulf_HRV = []
nLF_HRV = []
nHF_HRV = []
lfhf_ratio_hrv =[]

columns = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
baseline_index = [0*minute_to_second*fs, 7.5*minute_to_second*fs]
stroop_index = [5*minute_to_second*fs, 12.5*minute_to_second*fs]
arithmetic_index = [15*minute_to_second*fs, 22.5*minute_to_second*fs]
speech_index = [25*minute_to_second*fs, 32.5*minute_to_second*fs]


# Baseline 頻域
for i in range(10):
    
    input_ecg = ecg_mV[baseline_index[0]+(i*slidingwidow_len*fs) : int((baseline_index[0]+(2.5*minute_to_second*fs))+(i*slidingwidow_len*fs))]
        
    #抓Rpeak
    median_ecg, rpeakindex = getRpeak.getRpeak_shannon(input_ecg, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
    
    # RRI計算
    rrinterval = np.diff(rpeakindex)
    rrinterval = rrinterval/(fs/1000) #RRI index點數要換算回ms (%fs，1000是因為要換算成毫秒)
    x_rrinterval = np.cumsum(rrinterval)
    rrinterval_resample = interpolate(rrinterval, rr_resample_rate*epoch_len) #補點為rr_resample_rate HZ
    x_rrinterval_resample = np.linspace(0, epoch_len, len(rrinterval_resample))
    
    
    # median_filter = medfilt(rrinterval_resample, medianfilter_k)
    # rrinterval_resample_median = rrinterval_resample-median_filter
    rrinterval_resample_zeromean=rrinterval_resample-np.mean(rrinterval_resample)
    
    
    # FFT轉頻域
    y_f_Real, x_f = fft_power(rrinterval_resample_zeromean, rr_resample_rate, 'hanning')
    
    # 
    # re_x_f = []
    # re_y_f_Real = []
    # for i in range(len(x_f)//3):
    #     re_y_f_Real.append(np.sum(y_f_Real[i*3: (i+1)*3]))
    #     re_x_f.append(x_f[(i)*3])
    
    
    # -------畫圖----------
    plt_len = 5
    
    plt.figure()
    plt.subplot(plt_len,1,1)
    plt.plot(median_ecg)
    plt.scatter(rpeakindex, median_ecg[rpeakindex], color='red')
    
    plt.subplot(plt_len,1,2)
    plt.plot(x_rrinterval, rrinterval)
    
    
    plt.subplot(plt_len,1,3)
    plt.plot(x_rrinterval_resample, rrinterval_resample)
    
    plt.subplot(plt_len,1,4)
    plt.plot(x_f, y_f_Real/100)
    plt.xlim(0.0,0.5)
    
    # plt.subplot(plt_len,1,5)
    # plt.plot(re_x_f, re_y_f_Real)
    # plt.xlim(0.0,0.5)
    # plt.tight_layout()
    
    # 參數計算
    tp_index = []
    hf_index = []
    lf_index = []
    vlf_index = []
    ulf_index = []
        
    
    tp_index.append(np.where( (x_f<=0.4)))  
    hf_index.append(np.where( (x_f>=0.15) & (x_f<=0.4)))  
    lf_index.append(np.where( (x_f>=0.04) & (x_f<=0.15)))  
    vlf_index.append(np.where( (x_f>=0.003) & (x_f<=0.04)))   
    ulf_index.append(np.where( (x_f<=0.003)))   
    
    
    tp_index = tp_index[0][0]
    hf_index = hf_index[0][0]
    lf_index = lf_index[0][0]
    vlf_index = vlf_index[0][0]
    ulf_index = ulf_index[0][0]
    
    
    tp = np.sum(y_f_Real[tp_index[0]:tp_index[-1]])
    hf = np.sum(y_f_Real[hf_index[0]:hf_index[-1]])
    lf = np.sum(y_f_Real[lf_index[0]:lf_index[-1]])
    vlf = np.sum(y_f_Real[vlf_index[0]:vlf_index[-1]])
    ulf = np.sum(y_f_Real[ulf_index[0]:ulf_index[-1]])
    nLF = (lf/(tp-vlf))*100
    nHF = (hf/(tp-vlf))*100
    lfhf_ratio = np.log(lf/hf)
    
    
    tp_HRV.append(tp)
    hf_HRV.append(hf)
    lf_HRV.append(lf)
    vlf_HRV.append(vlf)
    ulf_HRV.append(ulf)
    nLF_HRV.append(nLF)
    nHF_HRV.append(nHF)
    lfhf_ratio_hrv.append(lfhf_ratio)




# Frequency Domain Epoch
plt_len = 8

plt.figure()

plt.subplot(plt_len,1,1)
plt.plot(tp_HRV)

plt.subplot(plt_len,1,2)
plt.plot(hf_HRV)

plt.subplot(plt_len,1,3)
plt.plot(lf_HRV)

plt.subplot(plt_len,1,4)
plt.plot(vlf_HRV)

plt.subplot(plt_len,1,5)
plt.plot(ulf_HRV)

plt.subplot(plt_len,1,6)
plt.plot(nLF_HRV)

plt.subplot(plt_len,1,7)
plt.plot(nHF_HRV)

plt.subplot(plt_len,1,8)
plt.plot(lfhf_ratio_hrv)




