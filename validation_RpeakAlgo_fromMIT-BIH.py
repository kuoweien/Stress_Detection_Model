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
HitR_list = []
true_positive_rate_list = []
positive_predictive_Value_list = []


n = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
      111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 
      122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 
      209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 
      222, 223, 228, 230, 231, 232, 233, 234]


for i in n:
    
    print(i)

    
    # Read data from MIT-BIH database
    record = wfdb.rdsamp('mitdb/'+str(i))
    annotation = wfdb.rdann('mitdb/'+str(i), 'atr')
    
    
    # R peak algorithm
    ecg_data = record[0][:, 0]
    median_ecg, rpeakindex = getRpeak.getRpeak_shannon(ecg_data, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
    
    
    
    # Evaluate performance
    peak_samples = annotation.sample
    peak_symbols = annotation.symbol
    
    real_peaks=[]
    for index, sym in enumerate(peak_symbols):
        if sym == 'N' or sym == 'V' or sym=='A':
            real_peaks=np.append(real_peaks, index)
    real_peaks= real_peaks.astype(int)
    real_peaks_index=peak_samples[real_peaks]
    
    
    true_positive=[]
    false_positive=[]
    
    HitR=np.ones(len(real_peaks_index), dtype= bool)
    for indP, ValP in np.ndenumerate(rpeakindex):
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
    true_positive_rate=len(true_positive)/(len(true_positive)+len(HitR))
    positive_predictive_Value=len(true_positive)/(len(true_positive)+len(false_positive))
    
    
    n_list.append(i)
    total_count_list.append(len(rpeakindex))
    true_positive_list.append(len(true_positive))
    false_positive_list.append(len(false_positive))
    HitR_list.append(len(HitR))
    true_positive_rate_list.append(true_positive_rate*100)
    positive_predictive_Value_list.append(positive_predictive_Value*100)

    
    # print('Total Count: {}'.format(len(rpeakindex)))
    # print('True Positive Count: {0:5d}'.format(len(true_positive)))
    # print('False Positive Count: {0:d}'.format(len(false_positive)))
    # print('False Negative Count: {0:d}'.format(len(HitR)))
    # print('TPR: {0:.3f}%'.format(true_positive_rate*100))
    # print('PPV: {0:.3f}%'.format(positive_predictive_Value*100))
    


plt.figure(figsize=(12,3))
plt.plot(ecg_data, c='black')
plt.scatter(rpeakindex, ecg_data[rpeakindex])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-2,2)
plt.xlim(48000, 60000)

# df = pd.DataFrame({'N':n_list, 'Total':total_count_list, 'TP':true_positive_list, 'FP':false_positive_list , 'FN':HitR_list, 'TPR':true_positive_rate_list, 'PPV':positive_predictive_Value_list})
# df.to_excel('ECG_validation.xlsx')

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





