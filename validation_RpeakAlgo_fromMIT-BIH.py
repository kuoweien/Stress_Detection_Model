#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:53:14 2022

@author: weien
"""

import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import def_getRpeak_main as getRpeak
# from wfdb.processing.peaks import find_local_peaks
from wfdb.processing import ann2rr

# if os.path.isdir("mitdb"):
#     print('You already have the data.')
# else:
#     wfdb.dl_database('mitdb', 'mitdb')

fs = 360

medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.32*fs)*2  #Patch=0.4*fs*2 LTA3=0.35*fs*2
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 20
highpass_fq = 10


sig_len = 30

n_list = []
total_count_list = []
true_positive_list = []
false_positive_list = []
false_negative_list = []
sensitivity_list = []
precision_list = []
accuracy_list = []


# n = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
#       111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 
#       122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 
#       209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 
#       222, 223, 228, 230, 231, 232, 233, 234]
n=[219]

# n=[102]

for i in n:
    
    print(i)

    
    # Read data from MIT-BIH database
    record = wfdb.rdsamp('mitdb/'+str(i))
    annotation = wfdb.rdann('mitdb/'+str(i), 'atr')
    
    
    # R peak algorithm
    ecg_data = record[0][:, 0]
    # median_ecg, detect_rpeakindex = getRpeak.getRpeak_shannon(ecg_data, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
    
    median_filter_data = getRpeak.medfilt(np.array(ecg_data), medianfilter_size)
    # median_filter_data = medfilt(ecg, medianfilter_size)
    median_ecg = ecg_data-median_filter_data
    lowpass_data = getRpeak.lowPassFilter(20,fs,median_ecg)  #低通
    bandfilter_data = getRpeak.highPassFilter(10,fs,lowpass_data)    #高通
    dy_data = getRpeak.defivative(bandfilter_data) #一程微分
    normalize_data = dy_data/np.max(dy_data) #正規化
    see_data = (-1)*(normalize_data**2)*np.log((normalize_data**2)) #Shannon envelop
    # lmin_index, lmax_index = hl_envelopes_idx(see_data) #取上包絡線
    # lmax_data = see_data[lmax_index]
    # interpolate_data = interpolate(lmax_data,len(ecg))
    gaussian_data = getRpeak.gaussian_filter(see_data, sigma=gaussian_filter_sigma)
    hibert_data = np.imag(getRpeak.hilbert(gaussian_data))  #Hilbert取複數
    movingaverage_data = getRpeak.movingaverage(hibert_data, moving_average_ms, fs) #moving average
    hibertmoving_data = hibert_data-movingaverage_data
    zero_index = getRpeak.findZeroCross(hibertmoving_data)  #Positive zero crossing point
    zero_shift_index = getRpeak.shiftArray(zero_index, final_shift) #位移結果
    
    #Decision Rule
    detect_maxRpeak_index, _   = getRpeak.findMaxvalue(median_ecg, zero_shift_index, detectR_maxvalue_range)  # RawECG抓R peak 找範圍內的最大值 
    detect_minRpeak_index, _   = getRpeak.findMinvalue(median_ecg, zero_shift_index, detectR_maxvalue_range)  # RawECG抓R peak 找範圍內的最大值 
    
    # detect_rpeakindex = []
    
    # for i in range(len(detect_maxRpeak_index)):
    #     if (np.abs(ecg_data[detect_maxRpeak_index[i]]) >= np.abs(ecg_data[detect_minRpeak_index[i]])).all():
    #         detect_rpeakindex.append(detect_maxRpeak_index[i])
    #     elif (np.abs(ecg_data[detect_maxRpeak_index[i]]) < np.abs(ecg_data[detect_minRpeak_index[i]])).all():
    #         detect_rpeakindex.append(detect_minRpeak_index[i])
            
    maxRpeak_sum = np.sum(np.abs(ecg_data[detect_maxRpeak_index]))
    minRpeak_sum = np.sum(np.abs(ecg_data[detect_minRpeak_index]))
        
    detect_rpeakindex = []
    if maxRpeak_sum >= minRpeak_sum:
        detect_rpeakindex = detect_maxRpeak_index
    elif minRpeak_sum > maxRpeak_sum:
        detect_rpeakindex = detect_minRpeak_index
    
    
    
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
            if np.absolute(ValP-ValR) < 50: # 50個點前後都算準確
                Hit=1
                HitR[indR[0]]=False
                true_positive=np.append(true_positive, indP[0])
                real_peaks_index= real_peaks_index[HitR]
                HitR=HitR[HitR]
                break
        if Hit==0:
            false_positive=np.append(false_positive, indP[0])
            
    false_negative = HitR
    accuracy = len(true_positive)/(len(true_positive)+len(false_positive)+len(false_negative))
    sensitivity=len(true_positive)/(len(true_positive)+len(false_negative))
    precision=len(true_positive)/(len(true_positive)+len(false_positive))
    
    
    n_list.append(i)
    total_count_list.append(len(detect_rpeakindex))
    true_positive_list.append(len(true_positive))
    false_positive_list.append(len(false_positive))
    false_negative_list.append(len(HitR))
    accuracy_list.append(round(accuracy*100,2))
    sensitivity_list.append(round(sensitivity*100,2))
    precision_list.append(round(precision*100,2))


    print('Total Count: {}'.format(len(detect_rpeakindex)))
    print('True Positive Count: {0:5d}'.format(len(true_positive)))
    print('False Positive Count: {0:d}'.format(len(false_positive)))
    print('False Negative Count: {0:d}'.format(len(false_negative)))
    print('ACC: {}%'.format(round(accuracy*100,2)))
    print('SEN: {}%'.format(round(sensitivity*100,2)))
    print('PRE: {}%'.format(round(precision*100,2)))
    
    print('-----')
    

    plt.figure(figsize=(12,4))
    plt.plot(median_ecg, 'black')
    plt.scatter(detect_rpeakindex, median_ecg[detect_rpeakindex], c='blue', alpha=0.5)
    plt.scatter(peak_samples, median_ecg[peak_samples], c='red', alpha=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(range(-4, 5, 1), fontsize=14)
    plt.xlim(0, len(median_ecg))
    plt.ylim(-4,4)
    plt.title('Record 104', fontsize=16)
    

# df = pd.DataFrame({'ECG record':n_list, 'Total (beats)':total_count_list, 'TP (beats)':true_positive_list, 'FP (beats)':false_positive_list , 'FN (beats)':false_negative_list, 'ACC (%)':accuracy_list, 'SEN ()%':sensitivity_list, 'PRE (%)':precision_list})
# df.to_excel('220928_ECG_validation.xlsx')

'''
df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/ECG_validation.xlsx')
accuracy = df['Accuracy (%)']
# n = df['N']
n = np.linspace(0, len(accuracy), len(accuracy))

plt.figure(figsize=(6,4))
plt.scatter(n, accuracy, c='black')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('N', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.tight_layout

'''





