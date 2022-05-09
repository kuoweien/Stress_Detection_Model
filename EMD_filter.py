#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:31:42 2021

@author: weien
"""

from PyEMD import EMD
import numpy as np
import matplotlib.pyplot as plt
# import dogHRV
import csv
import pandas as pd
from scipy.signal import butter,filtfilt
from scipy import signal


#%%

# df=pd.read_csv('negative_scared.csv')
# negative_scared=df['RawData']
def useEMDfilterLowFQ(getECGlist):
    emd = EMD()
    IMFs = emd(getECGlist)
    
    # plt.figure()
    # plt.subplot(1,1,1)
    # plt.plot(getECGlist,color='black')
    # plt.title('EMG (33-125HZ)')
    # plt.ylim(-2000,2000)
# '''   
#     allplotLen=len(IMFs)+1
#     plt.figure(figsize=(12,9))
    
#     plt.subplot(allplotLen+1,1,1)
#     plt.plot(getECGlist,'black')
    
    
#     for i in range(0,allplotLen-1): 
#         plt.subplot(allplotLen,1,i+2)
#         plt.plot(IMFs[i],'gray')
#         plt.ylabel('IMF'+str(i+1))
#         plt.locator_params(axis='y', nbins=7)
        
#     plt.xlabel('Time')
#     plt.tight_layout()
# '''
    
    plt.figure(figsize=(12,9))
    plt.subplot(len(IMFs)+1, 1, 1)
    plt.plot(getECGlist, 'black')
    
    for n in range(len(IMFs)):
        plt.subplot(len(IMFs)+1, 1, n+2)
        plt.plot(IMFs[n], 'gray')
        plt.ylabel("IMFs %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)
    
    plt.xlabel("Time [s]")
    plt.tight_layout()
    
    
    
    fig, axes = plt.subplots(nrows=9, ncols=1)

    for i, ax in enumerate(axes.flat, start=1):
        ax.set_title('Test Axes {}'.format(i))
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
    
    fig.tight_layout()
    plt.show()
    
    
    IMF_lastindex=len(IMFs)-1
    #刪除IMF=1,及最後兩個
    # IMFs_filter = np.delete(IMFs,(0,IMF_lastindex,IMF_lastindex-1), axis = 0)
    IMFs_filter = IMFs[0:3]
    IMFs_Filter_SUM=sum(IMFs_filter)
    
    plt.figure()
    plt.plot(IMFs_Filter_SUM,color='black')
    plt.title('EMG (filter) ')
    plt.ylim(-2000,2000)
    plt.show()
    
    return IMFs_Filter_SUM
    
    
#%%

df=pd.read_csv('negative_scared.csv')
data_scared=df['RawData']
data_scared=data_scared[1800:2300]

emg_EMD=useEMDfilterLowFQ(np.array(data_scared))

fq=124
b, a = signal.butter(8, (2*fq)/250, 'lowpass')   #濾除125HZ以上的頻率，wn=2*125/250=1
data_lowfilter = signal.filtfilt(b, a, data_scared) 

fq=33
b, a = signal.butter(8, (2*fq)/250, 'highpass') #濾除33HZ以下的頻率，wn=2*33/250=1
data_scared_emg = signal.filtfilt(b, a, data_lowfilter)
# data_scared_emg=data_scared_emg[1800:16800]#1分鐘
data_scared_emg=data_scared_emg[1800:2300]

emg_EMD=useEMDfilterLowFQ(data_scared_emg)
emg_rms = np.sqrt(np.mean(emg_EMD**2))
print(round(emg_rms,3))


#%%#低通率波器

fs = 250     # sample rate, Hz
cutoff = 33      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = 15000 # total number of samples

# 實現濾波器
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#%%將Rawdata輸出存成csv檔
'''
ecgrawdata_file1,ecgfq_file1,updatetime_file1 = dogHRV.openRawFile('210902e.241')

Baseline1 = dogHRV.getConditionRawdata('14:52:00','14:54:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
Baseline1=np.array(Baseline1)

positive_touch = dogHRV.getConditionRawdata('14:54:00','14:57:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
positive_touch=np.array(positive_touch)

Baseline2 = dogHRV.getConditionRawdata('14:57:00','15:00:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
Baseline2=np.array(Baseline2)

negative_scared = dogHRV.getConditionRawdata('15:00:21','15:02:30',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
negative_scared=np.array(negative_scared)

Baseline3 = dogHRV.getConditionRawdata('15:02:30','15:04:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
Baseline3=np.array(Baseline3)

positive_play = dogHRV.getConditionRawdata('15:04:04','15:06:29',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
positive_play=np.array(positive_play)

Baseline4 = dogHRV.getConditionRawdata('15:07:00','15:10:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
Baseline4=np.array(Baseline4)

ecgrawdata_file2,ecgfq_file2,updatetime_file2 = dogHRV.openRawFile('210902f.241')

Baseline6 = dogHRV.getConditionRawdata('15:20:00','15:23:00',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
Baseline6=np.array(Baseline6)

Walk = dogHRV.getConditionRawdata('15:24:00','15:26:00',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
Walk=np.array(Walk)
'''

#%%濾各情境
'''
negative_scared_filterLowFQ = useEMDfilterLowFQ(negative_scared)
negative_scared_filter = butter_lowpass_filter(negative_scared_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'negative_scared_filter':negative_scared_filter})
df.to_csv('negative_scared_filter.csv')


Baseline1_filterLowFQ = useEMDfilterLowFQ(Baseline1)
Baseline1_filter = butter_lowpass_filter(Baseline1_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'Baseline1_filter':Baseline1_filter})
df.to_excel('Baseline1_filter.xls')


Baseline2_filterLowFQ = useEMDfilterLowFQ(Baseline2)
Baseline2_filter = butter_lowpass_filter(Baseline2_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'Baseline2_filter':Baseline2_filter})
df.to_excel('Baseline2_filter.xls')

Baseline3_filterLowFQ = useEMDfilterLowFQ(Baseline3)
Baseline3_filter = butter_lowpass_filter(Baseline3_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'Baseline3_filter':Baseline3_filter})
df.to_excel('Baseline3_filter.xls')

Baseline4_filterLowFQ = useEMDfilterLowFQ(Baseline4)
Baseline4_filter = butter_lowpass_filter(Baseline4_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'Baseline4_filter':Baseline4_filter})
df.to_excel('Baseline4_filter.xls')

Baseline6_filterLowFQ = useEMDfilterLowFQ(Baseline6)
Baseline6_filter = butter_lowpass_filter(Baseline6_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'Baseline6_filter':Baseline6_filter})
df.to_excel('Baseline6_filter.xls')

positive_touch_filterLowFQ = useEMDfilterLowFQ(positive_touch)
positive_touch_filter = butter_lowpass_filter(positive_touch_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'positive_touch_filter':positive_touch_filter})
df.to_excel('positive_touch_filter.xls')

positive_play_filterLowFQ = useEMDfilterLowFQ(positive_play)
positive_play_filter = butter_lowpass_filter(positive_play_filterLowFQ, cutoff, fs, order)

df = pd.DataFrame({'positive_play_filter':positive_play_filter})
df.to_excel('positive_play_filter.xls')
'''

# df = pd.DataFrame({'Baseline1_filter':Baseline1_filter,'Baseline2_filter':Baseline2_filter,
#                    'Baseline3_filter':Baseline3_filter,
#                    'Baseline4_filter':Baseline4_filter,'Baseline6_filter':Baseline6_filter,
#                    'positive_touch_filter':positive_touch_filter,'negative_scared_filter':negative_scared_filter,
#                    'positive_play_filter':positive_play_filter
#                    })

# df.to_excel('DogECGFilter.xls')


# ----可刪-------
# plt.plot(y)
# plt.title('Low and High Pass Filter/Negative_scared')
# plt.show()

# 濾過的訊號再拆成EMD
# IMFs = emd(y)

# allplotLen=len(IMFs)+1
# plt.figure(figsize=(6,10))

# plt.subplot(allplotLen+1,1,1)
# plt.plot(y)
# # plt.title('Original')

# for i in range(0,allplotLen-1): 
#     plt.subplot(allplotLen,1,i+2)
#     plt.plot(IMFs[i])
#     # plt.title('IMF'+str(i))

# plt.show()

#%%




