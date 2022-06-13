#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:31:42 2021

@author: weien
"""

'''EMD'''

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







