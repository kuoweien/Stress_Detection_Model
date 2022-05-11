#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:42:33 2022

@author: weien
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import def_passFilter as bandfilter
import def_linearFunc
import def_pantompkin as pantompkin
from scipy.interpolate import interp1d
import math



#%%

#nemo
situation = 'Shake'

length_s = 30
index = 0

fs = 250

clip_start = index*(length_s*fs) 
clip_end = (index+1)*(length_s*fs) 

url = '/Users/weien/Desktop/ECG穿戴/HRV實驗/人體/Rawdata/220510郭葦珊/'

df=pd.read_csv(url+situation+'.csv').iloc[clip_start:clip_end] #petted.iloc[10000:17500] scared[2500:10000]
ecg_dataraw = df[situation]

#參數調整：
lowpass = 15
highpass = 5
qrs_range = 30 #計算EMG時，在detect rpeak向左刪除之距離 （狗使用25 人使用30）
tpeak_range = 37 #計算EMG時，在detect rpeak向右刪除之距離 （狗使用37 人使用0）
rpeak_findmin_range = 100


interval_freq = 7 #rrinterval內差為7HZ
magnification = 500 #LTA3放大倍率


ecg_dataraw=ecg_dataraw.reset_index(drop=True)
rawdata_mV = ((ecg_dataraw*(1.8/65535)-0.9)/magnification)*1000  #轉換為電壓

median_adjustline = pantompkin.medfilt(rawdata_mV.values,61) #sliding window折照為一半 120ms->61
ecg_median = rawdata_mV-median_adjustline  #基線飄移
ecg_lowpass=bandfilter.lowPassFilter(lowpass,ecg_median)        #低通
ecg_bandpass = bandfilter.highPassFilter(highpass,ecg_lowpass)        #高通
ecg_defivative = pantompkin.defivative(ecg_bandpass)       #導數
ecg_square = np.square(ecg_defivative)       #平方
# movingwindow= pantompkin.movingaverage(ecg_defivative)     #moving average
peaks_x, peaks_y = pantompkin.findpeak(ecg_square)
detedted_rpeak_x,detedted_rpeak_y = pantompkin.detectRpeak(rawdata_mV, peaks_x, peaks_y)       #Pantompkin決策演算抓rpeak 資料來源：網路找的Github

newdetedted_rpeak_x, newdetedted_rpeak_y = pantompkin.ecgfindthemaxvalue(rawdata_mV, detedted_rpeak_x,rpeak_findmin_range) #找最小值

rrinterval = np.diff(newdetedted_rpeak_x)
rrinterval = rrinterval*1000/fs
x_time = np.linspace(0, 40, len(rrinterval))

#內差rrinterval為7HZ
f1 = interp1d(x_time, rrinterval) 
rrx_interpolate = np.linspace(0, 40, 40*interval_freq)
rry_interpolate = f1(rrx_interpolate)

#直方圖：峰值與偏度
[niu, sigma, skew, kurt] = pantompkin.calc_stat(rrinterval)


#%%

#畫直方圖
plt.figure()
plt.title(situation)
plt.hist(rrinterval,density=False, stacked=True,facecolor='grey',alpha=0.9)
plt.grid(True)
plt.xlabel('RR interval')
plt.ylim(0,30)
plt.xlim(400,1200)


print('RR Mean: '+str(np.mean(rrinterval))+' RR SD: '+str(np.std(rrinterval)))
print(situation+' Skewness: '+str(round(skew,2))+' Kurtosis: '+str(round(kurt,2)))



plt.figure(figsize=(16,12))
plt.subplot(6,1,1)
plt.plot(rawdata_mV)
plt.plot(newdetedted_rpeak_x, newdetedted_rpeak_y, "o", markersize=3, c='red')
plt.title('Raw')

plt.subplot(6,1,2)
plt.plot(ecg_median)
plt.title('MedianFilter')
plt.tight_layout()

plt.subplot(6,1,3)
plt.plot(ecg_bandpass)
plt.title('Band Pass Filter (5-15HZ)')
plt.tight_layout()

plt.subplot(6,1,4)
plt.plot(ecg_defivative)
# plt.plot(peaks_x, peaks_y, "o", markersize=3, c='red')
plt.title('Filtered with Derivative filter')

plt.tight_layout()

plt.subplot(6,1,5)
plt.plot(ecg_square)
# plt.plot(peaks_x, peaks_y, "o", markersize=3, c='red')
plt.title('Squared')
plt.tight_layout()

plt.subplot(6,1,6)
# plt.plot(movingwindow)
# plt.plot(peaks_x, peaks_y, "o", markersize=3, c='red')
plt.title('Moving Average')
plt.tight_layout()



#%% EMG
#取EMG
emg_mV  = pantompkin.fillRTpeakwithLinear(rawdata_mV,newdetedted_rpeak_x, qrs_range, tpeak_range) #刪除rtpeak並補線性點
emg_mV_linearwithzero, emg_list = pantompkin.deleteRTpeak(rawdata_mV,newdetedted_rpeak_x, qrs_range, tpeak_range) #刪除rtpeak並補0
emg_mV_withoutZero = pantompkin.deleteZero(emg_mV_linearwithzero) #


#emg轉成mV 並取rms
emg_rms = round(np.sqrt(np.mean(emg_mV_withoutZero**2)),3)

print('RMS'+str(emg_rms))


plt.figure(figsize = (12,4))
plt.title(situation)
plt.subplot(3,1,1)
plt.ylim(-0.3,0.5)
plt.plot(np.array(range(len(rawdata_mV)))/250,np.array(rawdata_mV),'black')
plt.plot(np.array(newdetedted_rpeak_x)/250, np.array(newdetedted_rpeak_y), "o", markersize=4, c='red')
plt.ylabel('ECG (mV)')

plt.subplot(3,1,2)#RRinterval
plt.ylim(400,1200)
plt.plot(rrx_interpolate,rry_interpolate,'black')
plt.ylabel('RR(ms)')

plt.subplot(3,1,3)
plt.ylim(-0.3,0.5)
plt.plot(np.array(range(len(emg_mV)))/250,emg_mV,'black')
plt.ylabel('EMG (mV)')
plt.xlabel('Time (s)')

plt.tight_layout()


