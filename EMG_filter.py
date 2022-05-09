#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:08:26 2021

@author: weien
"""

#先高通
import pandas as pd
from scipy.signal import butter,filtfilt
from scipy import signal
import matplotlib.pyplot as plt
import defgetRpeak as getRpeak
import numpy as np

df=pd.read_csv('negative_scared.csv')
data_scared=df['RawData']

fq=124
b, a = signal.butter(8, (2*fq)/250, 'lowpass')   #濾除125HZ以上的頻率，wn=2*125/250=1
data_lowfilter = signal.filtfilt(b, a, data_scared) 

fq=33
b, a = signal.butter(8, (2*fq)/250, 'highpass') #濾除33HZ以下的頻率，wn=2*33/250=1
data_scared_emg = signal.filtfilt(b, a, data_lowfilter)
# data_scared_emg=data_scared_emg[1800:16800]#1分鐘
data_scared_emg=data_scared_emg[1800:2300]

df = pd.DataFrame({'EMG':data_scared_emg})
rawdatalist=df.EMG
ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,1000)

plt.figure()
plt.plot(rawdatalist)
plt.title('EMG (33-125HZ)')
plt.ylim(-3500,3500)

# 畫rpeak原始圖位置圖
plt.figure()
plt.plot(x_time, df['EMG'])
plt.xlabel('Time')
plt.ylabel('EMG')
plt.title('EMG (33-125HZ)')
plt.ylim(-3500,3500)
plt.scatter(peaklist_time, ybeat, c='red') #Plot detected peaks

#將rpeak的點改成0->濾掉r peak的部分
for i in range(len(peaklist_time)):
    rpeak_index=peaklist_time[i]
    rawdatalist[rpeak_index]=0

plt.figure()
plt.plot(rawdatalist)
plt.title('EMG (33-125HZ and exclude R peak)')
plt.ylim(-3500,3500)



#RR interval
rrinterval_list=[]
for i in range(1,len(peaklist_time)):
    rrinterval_list.append((peaklist_time[i]-peaklist_time[i-1])*4)

#畫RR直方圖
# plt.figure()
# plt.hist(rrinterval_list)
# plt.title('Scared')
# plt.xlabel('R-R intervals')
# plt.ylim(0,40)
# plt.show()

