#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:19:01 2022

@author: weien
"""

#參考模板：https://pypi.org/project/py-ecg-detectors/

from ecgdetectors import Detectors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
from scipy import signal
import linearFunc

#包含T波一起刪除
def fillRTpeakwithLinear(rawdata, rpeakindex):
    rpeak_linear=rawdata

    #將rpeak的點改成0->濾掉r peak的部分
    for i in range(len(rpeakindex)):
        rpeak_index=rpeakindex[i]
        
        startX=rpeak_index-12
        startY=rawdata[rpeak_index-12]
        endX=rpeak_index+40
        
        if len(rawdata)<endX:
            endX=len(rawdata)
            endY=rawdata[len(rawdata)-1]
        elif len(rawdata)>=endX:
            endX=endX
            endY=rawdata[rpeak_index+40]
        
        linearOutput=linearFunc.linearFunc([startX,startY],[endX,endY]) #linearFunc.py引入 #共前後1秒
        firstindex=linearOutput[0][0]
        # lastindex=linearOutput[-1][0]
        
        for j in range(0,len(linearOutput)):
            rpeak_linear[(j+firstindex)] = linearOutput[j][1]
        
    return linearOutput,rpeak_linear


fs = 250
detectors = Detectors(fs)

df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/petted.csv').iloc[10000:12500]#10000:12500
# df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/isolation.csv').iloc[18000:20500]

ecg = df['petted']
ecg = ecg.reset_index(drop = True)
ecg_V=((ecg*(1.8/65535)-0.9)/500)*1000

r_peaks = detectors.pan_tompkins_detector(ecg_V)
peak_x = np.array(r_peaks)
r_peaks_y = ecg_V[peak_x]

plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
# plt.plot(r_peaks,ecg_V[r_peaks],'o')
plt.scatter(r_peaks, ecg_V[r_peaks] , c='red') 
plt.plot(ecg_V)
plt.ylim(-0.5,0.5)


linearOutput2,r_t_peakfillbylinear=fillRTpeakwithLinear(ecg_V, peak_x.tolist())

#畫刪除前後1秒的ECG資料
plt.subplot(3,1,2)
# excludeECG_V=(rawdata_Rpeakfillbylinear*(1.8/65535)-0.9)/500
plt.plot(r_t_peakfillbylinear)
# plt.plot(r_t_peakfillbylinear)
plt.ylim(-0.5,0.5)
plt.xlabel('Time (s)')
plt.title('EMG')

#高通濾波   
fq=10
b, a = signal.butter(5, (2*fq)/250, 'highpass')   #濾除33HZ以下的頻率，wn=2*125/250=1
excludeECG_EMG = signal.filtfilt(b, a, r_t_peakfillbylinear) 


plt.subplot(3,1,3)
plt.plot(excludeECG_EMG)
plt.xlabel('Time (s)')
plt.title('EMG ({}-250HZ)'.format(fq))
plt.ylim(-0.5,0.5)

plt.tight_layout()

