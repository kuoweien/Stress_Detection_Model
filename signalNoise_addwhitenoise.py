#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:53:56 2022

@author: weien
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import def_pantompkin as pantompkin
import def_passFilter as bandfilter
import def_Shannon as shannon
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert, chirp


def normalize(data, lowest_rms, highest_rms):
    
    point = ((data-lowest_rms)/(highest_rms-lowest_rms))*100
    
    return point

def ecgEpochScore(ecg):
    rms = np.sqrt(np.mean(np.array(ecg)**2))
    noise_score = normalize(rms, 1186.45, 32767)
    return noise_score

#使用pantomskin抓Peak
def getRpeak_pantompskin(ecg):
    median_adjustline = pantompkin.medfilt(ecg.values,61) #sliding window折照為一半 120ms->61
    ecg_median = ecg-median_adjustline  #基線飄移
    rawdata_mV = ecg_median
    ecg_lowpass=bandfilter.lowPassFilter(15,ecg_median)        #低通
    ecg_bandpass = bandfilter.highPassFilter(5,ecg_lowpass)        #高通
    ecg_defivative = pantompkin.defivative(ecg_bandpass)       #導數
    ecg_square = np.square(ecg_defivative)       #平方
    peaks_x, peaks_y = pantompkin.findpeak(ecg_square)
    detedted_rpeak_x,detedted_rpeak_y = pantompkin.detectRpeak(rawdata_mV, peaks_x, peaks_y)       #Pantompkin決策演算抓rpeak 資料來源：網路找的Github
    newdetedted_rpeak_x, newdetedted_rpeak_y = pantompkin.ecgfindthemaxvalue(ecg, detedted_rpeak_x, (0.35*fs)*2)
    
    return newdetedted_rpeak_x

#Shonnon抓Rpeak
def getRpeak_shannon(ecg):
    median_filter_data = shannon.medfilt(np.array(ecg), 61)
    median_ecg = ecg-median_filter_data
    lowpass_data = bandfilter.lowPassFilter(20, median_ecg)  #低通
    bandfilter_data = bandfilter.highPassFilter(10, lowpass_data)    #高通
    dy_data = shannon.defivative(bandfilter_data) #一程微分
    normalize_data = dy_data/np.max(dy_data) #正規化
    see_data = (-1)*(normalize_data**2)*np.log((normalize_data**2)) #Shannon envelop
    lmin_index, lmax_index = shannon.hl_envelopes_idx(see_data) #取上包絡線
    lmax_data = see_data[lmax_index]
    interpolate_data = shannon.interpolate(lmax_data,len(ecg))
    gaussian_data = gaussian_filter(interpolate_data, sigma=gaussian_filter_sigma)
    hibert_data = np.imag(hilbert(gaussian_data))  #Hilbert取複數
    movingaverage_data = shannon.movingaverage(hibert_data, moving_average_ms) #moving average
    hibertmoving_data = hibert_data-movingaverage_data
    zero_index = shannon.findZeroCross(hibertmoving_data)  #Positive zero crossing point
    zero_shift_index = shannon.shiftArray(zero_index, final_shift) #位移結果
    
    #Decision Rule: input分為三種 1.以RawECG找最大值 2.bandfilterECG找最大值 3.RawECG找最小值
    detect_Rpeak_index, _   = shannon.ecgfindthemaxvalue(median_ecg, zero_shift_index, detectR_maxvalue_range)  # RawECG抓R peak 找範圍內的最大值 
    re_detect_Rpeak_index = shannon.deleteCloseRpeak(detect_Rpeak_index, rpeak_close_range) #刪除rpeak間隔小於rpeak_close_range之值
    # re_detect_Rpeak_index = shannon.deleteLowerRpeak(re_detect_Rpeak_index, ecg, 2000)

    return median_ecg, re_detect_Rpeak_index


fs = 500
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5 
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.5*fs)*2  #草哥使用(0.3*fs)*2
detectR_minvalue_range = (0.5*fs)*2 
rpeak_close_range = 0.15*fs #0.1*fs


url_data = '/Users/weien/Desktop/ECG穿戴/測試ECG/標準心電圖(不同R的mV)(PATCH)/cleanECG_r=0.3mV.csv'
# url_data = '/Users/weien/Desktop/ECG穿戴/測試ECG/有雜訊心電圖/noise0.2.csv'
# url_data = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/Dataset/Rawdata/220517-5夢源/Patch/AllData.csv'

df_data = pd.read_csv(url_data)
ecg=df_data['ECG']
ecg = ecg-np.median(ecg)
lowest_rms = np.sqrt(np.mean(np.array(ecg)**2))

# plt.figure(figsize=(14,2))
# plt.plot(ecg,'black')


#%%建立標準化的演算法源頭


# white noise 全滿
mean = 0
std = 10000 
num_samples = 1000
whitenoise = np.random.normal(mean, std, size=num_samples)
rms_whitenoise_full = np.sqrt(np.mean(np.array(whitenoise)**2))


rms_initial = np.sqrt(np.mean(np.array(ecg)**2))

# para = np.linspace(0, 0.038, 380)
# rms_list = []
# for i in range(len(para)):
#     rms = np.sqrt(np.mean(np.array((whitenoise*para[i])+ecg)**2))
#     rms_list.append(rms)
    

#%%其他可能訊號之rms值
'''
rms_full = np.sqrt(np.mean(np.array((np.array([32767]*1000))**2)))

rms_zero = np.sqrt(np.mean(np.array((np.array([0]*1000))**2)))

halfECG = ([32767]*500)+list(ecg[0:500])
rms_half = np.sqrt(np.mean(np.array((np.array(halfECG))**2)))
'''

#%%
#畫圖

ecg = ecg+whitenoise*0.03

threshold_rms = np.sqrt(np.mean(np.array(ecg)**2))

threshold_score = normalize(threshold_rms, lowest_rms, 32767)
median_ecg, redetect_Rpeak_index = getRpeak_shannon(ecg)

plt.figure(figsize=(14,2))
plt.plot(median_ecg, c='black')
plt.scatter(np.array(redetect_Rpeak_index), median_ecg[redetect_Rpeak_index], alpha=0.5, c='r')
plt.ylabel('ECG (mV)')

#%%驗證Data


url_data = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/Dataset/Rawdata/220517-5夢源/Patch/AllData.csv'
df_data = pd.read_csv(url_data)
ecg=df_data['ECG']
plt.plot(ecg)
# ecg = ecg-np.median(ecg)
ecg_clip = ecg[640000:641000]
redetect_Rpeak_index = getRpeak_shannon(ecg_clip)
data_rms = np.sqrt(np.mean(np.array(ecg_clip)**2))
data_score = normalize(data_rms, lowest_rms, 32767)
print(data_score)

medianfilter_ecg, redetect_Rpeak_index = getRpeak_shannon(ecg_clip)
plt.figure(figsize=(14,2))
plt.plot(medianfilter_ecg, c='black')
plt.scatter(np.array(redetect_Rpeak_index), medianfilter_ecg[redetect_Rpeak_index], alpha=0.5, c='r')
plt.ylabel('ECG (mV)')
plt.ylim(0,1000)

