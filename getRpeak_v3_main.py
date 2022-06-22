#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:23:50 2022

@author: weien
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import def_getRpeak_main as getRpeak
import def_measureSQI as measureSQI
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import math

'''220622 還沒加Epoch'''

fs = 500

checknoiseThreshold = 0.3 #2秒Epoch刪雜訊時的閾值 Patch=0.419 LTA3=2.210

rri_epoch = 30 # Epoch

patch_baseline = 0
lta3_baseline = 0.9
patch_magnification = 120
lta3_magnification = 250

medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.4*fs)*2  #草哥使用(0.3*fs)*2 #Patch=0.4*fs*2 LTA3=0.35*fs*2
detectR_minvalue_range = (0.5*fs)*2 
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 30
highpass_fq = 5

#%%讀取Data

columns = ['Baseline', 'Stroop', 'Baseline_after_stroop', 'Arithmetic', 'Baseline_after_Arithmetic', 'Speech', 'Baseline_after_speech']

n = 10

for j in range(1, len(columns)):
# for j in range(0, 1):
    situation = columns[j]
    #讀取之ECG csv檔
    ecg_url = 'Data/N{}/{}.csv'.format(n,situation) 
    df = pd.read_csv(ecg_url)
    ecg_raw = df['ECG']
    df_parameter = pd.DataFrame()
    
    
    for i in range(0, 10): 
        ecg = ecg_raw[i*fs*rri_epoch : (i+1)*rri_epoch*fs]
        if len(ecg) == 0:
            break
        ecg_nonoise = measureSQI.splitEpochandisCleanSignal(ecg, fs, checknoiseThreshold) #兩秒為Epoch，將雜訊的Y值改為0
    
    #%%抓R peak位置
    
    #單位換算
        ecg_nonoise_mV = (((np.array(ecg_nonoise))*1.8/65535-patch_baseline)/patch_magnification)*1000
        ecg_mV = (((np.array(ecg))*1.8/65535-patch_baseline)/patch_magnification)*1000
    
    #抓Rpeak
        median_ecg, rpeakindex = getRpeak.getRpeak_shannon(ecg_nonoise_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
        
        # median_ecg, rpeakindex = getRpeak.getRpeak_pantompskin(ecg_mV, fs, medianfilter_size, lowpass_fq, highpass_fq)
       
    
    # 畫圖
        plt.figure(figsize=(14,2))
        plt.plot(median_ecg, 'black')
        # plt.ylim(-1.5, 1.8)
        plt.scatter(np.array(rpeakindex), median_ecg[rpeakindex], alpha=0.5, c='r')
    
    #%%計算RRI
    # RR interval
    
        rrinterval = np.diff(rpeakindex)
        rrinterval = rrinterval/(fs/1000) #RRI index點數要換算回ms (%fs，1000是因為要換算成毫秒)
    
        # plt.figure(figsize=(10,2))
        # plt.ylim(0,3600)
        # plt.xlim(0, len(rrinterval))
        # plt.ylabel('ms')
        # plt.plot(rrinterval, '-o', c='black')
    
    
        re_rrinterval = getRpeak.interpolate_rri(rrinterval, fs)
    
        # plt.figure(figsize=(10,2))
        # plt.ylim(0,3600)
        # plt.xlim(0, len(rrinterval_add))
        # plt.ylabel('ms')
        # plt.plot(rrinterval_add, '-o', c='black') 
    
        #RRI 相關參數
        rri_mean = np.mean(re_rrinterval)
        rri_sd = np.std(re_rrinterval)
        
        outlier_upper = rri_mean+(3*rri_sd) 
        outlier_lower = rri_mean-(3*rri_sd)
        
        re_rrinterval = re_rrinterval[re_rrinterval<outlier_upper]
        re_rrinterval = re_rrinterval[re_rrinterval>outlier_lower]  #刪除outlier的rrinterval
        
        #因有刪除outlier，所以重新計算平均與SD
        rri_mean = np.mean(re_rrinterval)
        rri_sd = np.std(re_rrinterval)
        [niu, sigma, rri_skew, rri_kurt] = getRpeak.calc_stat(re_rrinterval) #峰值與偏度
        rri_rmssd = math.sqrt(np.mean((np.diff(re_rrinterval)**2))) #RMSSD
        rri_nn50 = len(np.where(np.abs(np.diff(re_rrinterval))>50)[0]) #NN50 心跳間距超過50ms的個數，藉此評估交感
        rri_pnn50 = rri_nn50/len(re_rrinterval)
    
    #%%取EMG
        qrs_range = int(0.32*fs)
        tpeak_range = int(0.2*fs)
        
        ##emg_mV  = getRpeak.fillRTpeakwithLinear(median_ecg,rpeakindex, qrs_range, tpeak_range) #刪除rtpeak並補線性點
        emg_mV_linearwithzero, emg_list = getRpeak.deleteRTpeak(median_ecg,rpeakindex, qrs_range, tpeak_range) #刪除rtpeak並補0
        emg_mV_withoutZero = getRpeak.deleteZero(emg_mV_linearwithzero) 
        emg_rms = round(np.sqrt(np.mean(emg_mV_withoutZero**2)),4)
        
        
        # 建立新檔
        # df_parameter = df_parameter.append({'N':n, 'Mean':rri_mean , 'SD':rri_sd, 'RMSSD':rri_rmssd, 'NN50':rri_nn50, 'pNN50':rri_pnn50, 'Skewness':rri_skew, 'Kurtosis':rri_kurt , 'EMG_RMS':emg_rms, 'Situation':situation} ,ignore_index=True)
        # df_parameter.to_excel('Data/N{}/HRV.xlsx'.format(n))
    
    # 讀取已建立檔，並加入新資料
        df_parameter = df_parameter.append({'N':n, 'Mean':rri_mean , 'SD':rri_sd, 'RMSSD':rri_rmssd, 'NN50':rri_nn50, 'pNN50':rri_pnn50, 'Skewness':rri_skew, 'Kurtosis':rri_kurt , 'EMG_RMS':emg_rms, 'Situation':situation} ,ignore_index=True)
    df_hrv = pd.read_excel('Data/N{}/HRV.xlsx'.format(n))
    df_hrv = df_hrv.append(df_parameter,ignore_index=True)
    df_hrv.to_excel('Data/N{}/HRV.xlsx'.format(n))
    
    
