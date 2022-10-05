#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:12:33 2022

@author: weien
"""
import def_getRpeak_main as rpeak_preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# n=50
# situation = 'Baseline'
# ecg_url = 'Data/ClipSituation_CSVfile/N{}/{}.csv'.format(n, situation)
# df = pd.read_csv(ecg_url)
# ecg= df['ECG']

n=217
# n=100
record = wfdb.rdsamp('mitdb/'+str(n))
annotation = wfdb.rdann('mitdb/'+str(n), 'atr')
ecg = record[0][:, 0]


fs = 250

medianfilter_size = 61
lowpass_fs = 20
highpass_fs = 10
gaussian_filter_sigma = 0.03*fs 
moving_average_ms = 2.5
detectR_maxvalue_range = (0.32*fs)*2
detectR_minvalue_range = (0.32*fs)*2
rpeak_close_range = 0.20*fs


median_filter_data = rpeak_preprocessing.medfilt(np.array(ecg), medianfilter_size)
median_ecg = ecg-median_filter_data
lowpass_data = rpeak_preprocessing.lowPassFilter(lowpass_fs, fs, median_ecg)  #低通
bandfilter_data = rpeak_preprocessing.highPassFilter(highpass_fs, fs, lowpass_data)    #高通
dy_data = rpeak_preprocessing.defivative(bandfilter_data) #一程微分
normalize_data = dy_data/np.max(dy_data) #正規化
see_data = (-1)*(normalize_data**2)*np.log((normalize_data**2)) #Shannon envelop
gaussian_data = rpeak_preprocessing.gaussian_filter(see_data, sigma=gaussian_filter_sigma)
hibert_data = np.imag(rpeak_preprocessing.hilbert(gaussian_data))  #Hilbert取複數
movingaverage_data = rpeak_preprocessing.movingaverage(hibert_data, moving_average_ms, fs) #moving average
hibertmoving_data = hibert_data-movingaverage_data
zero_index = rpeak_preprocessing.findZeroCross(hibertmoving_data)  #Positive zero crossing point

#Decision Rule: input分為三種 1.以RawECG找最大值 2.bandfilterECG找最大值 3.RawECG找最小值
detect_maxRpeak_index, _   = rpeak_preprocessing.findMaxvalue(median_ecg, zero_index, detectR_maxvalue_range)  # RawECG抓R peak 找範圍內的最大值 
detect_minRpeak_index, _   = rpeak_preprocessing.findMinvalue(median_ecg, zero_index, detectR_maxvalue_range)  # RawECG抓R peak 找範圍內的最大值 

maxRpeak_sum = np.sum(np.abs(ecg[detect_maxRpeak_index]))
minRpeak_sum = np.sum(np.abs(ecg[detect_minRpeak_index]))



detect_Rpeak_index = []
if maxRpeak_sum >= minRpeak_sum:
    detect_Rpeak_index = detect_maxRpeak_index
elif minRpeak_sum > maxRpeak_sum:
    detect_Rpeak_index = detect_minRpeak_index
detect_rpeakindex = rpeak_preprocessing.deleteCloseRpeak(detect_Rpeak_index, rpeak_close_range) #刪除rpeak間隔小於rpeak_close_range之值




# Evaluate performance
peak_samples = annotation.sample
peak_symbols = annotation.symbol

real_peaks=[]
for index, sym in enumerate(peak_symbols):
    # if sym == 'N' or sym == 'V' or sym=='A':
    if sym == '/' or sym == 'N' or sym == 'L' or sym == 'R' or sym == 'A' or sym == 'a' or sym == 'J' or sym == 'S' or sym == 'V' or sym == 'F' or sym == 'O' or sym == 'N' or sym == 'E' or sym == 'P' or sym == 'F' or sym == 'Q':
        real_peaks=np.append(real_peaks, index)
real_peaks= real_peaks.astype(int)
real_peaks_index=peak_samples[real_peaks]


true_positive=[]
false_positive=[]

HitR=np.ones(len(real_peaks_index), dtype= bool)
for indP, ValP in np.ndenumerate(detect_rpeakindex):
    Hit=0
    for indR, ValR in np.ndenumerate(real_peaks_index):
        if np.absolute(ValP-ValR) < 50:
            Hit=1
            HitR[indR[0]]=False
            true_positive=np.append(true_positive, indP[0])
            real_peaks_index= real_peaks_index[HitR]
            HitR=HitR[HitR]
            break
    if Hit==0:
        false_positive=np.append(false_positive, indP[0])
        
false_negative = len(HitR)
true_positive_rate=len(true_positive)/(len(true_positive)+false_negative)
positive_predictive_Value=len(true_positive)/(len(true_positive)+len(false_positive))
accuracy = len(true_positive)/len(detect_rpeakindex)


n_list = []
total_count_list = []
true_positive_list = []
false_positive_list = []
false_negative_list = []
sensitivity_list = []
precision_list = []
accuracy_list = []


n_list.append(n)
total_count_list.append(len(detect_rpeakindex))
true_positive_list.append(len(true_positive))
false_positive_list.append(len(false_positive))
false_negative_list.append(len(HitR))
sensitivity_list.append(round(true_positive_rate*100,2))
precision_list.append(round(positive_predictive_Value*100,2))
accuracy_list.append(round(accuracy*100,2))


plt.figure()
plt.subplot(3,1,1)
plt.plot(ecg,'black')
plt.subplot(3,1,2)
plt.plot(median_ecg, 'black')
plt.scatter(detect_rpeakindex, median_ecg[detect_rpeakindex], c='blue')
plt.scatter(peak_samples, median_ecg[peak_samples], c='red', alpha=0.5)
# plt.scatter(test_real_peaks_index, median_ecg[test_real_peaks_index], c='green', alpha=0.5)
plt.subplot(3,1,3)
plt.plot(median_filter_data, 'black')





