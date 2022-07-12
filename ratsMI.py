#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:37:48 2022

@author: weien
"""

import def_getRpeak_main as getRpeak
import matplotlib.pyplot as plt
import numpy as np
import def_readandget_Rawdata as getRawdata
import pandas as pd


# 每一秒算一個Var or SD
def onesecond_rrivar(x_rri, rri):
    threshold = 0
    rri_var_epoch = []
    index_list = []
    for i in range(len(x_rri)):
        
        if int(x_rri[i]) == threshold:
            index_list.append(rri[i])
        
        elif int(x_rri[i]) != threshold:
            rri_var_epoch.append(np.var(index_list))
            index_list = []
            threshold+=1
    
    return rri_var_epoch

# 將因雜訊刪除的RRI進行補點
def interpolate_rri(rawrrinterval, fs):
    rrinterval_add = rawrrinterval
    # 計算時需將先前因雜訊而刪除的地方做補點
    i = 0
    while i < len(rrinterval_add):
    # for i in range(len(rrinterval)):
        if rrinterval_add[i] >= 200 :    # 因為要把index值換算成ms
            insert_distance = (rrinterval_add[i-1] + rrinterval_add[i+1])/2
            n = int(rrinterval_add[i]/insert_distance)
            add_list = [insert_distance] * n
            rrinterval_add = np.append(np.append(rrinterval_add[:i], add_list), rrinterval_add[i+1:])
            i+=n
        i+=1
        
    return rrinterval_add

'''讀檔'''
# ecg, fs, time = getRawdata.openRawFile('/Users/weien/Desktop/kylab專案/勝杰的心肌梗塞大鼠分析/MIraw檔/MI前/wc1.RAW')
# ecg_clip = inputtimetoClipRawdata(ecg, fs, time, '22:00:00', '23:30:00')
# df = pd.DataFrame({'ECG': ecg_clip})
# df.to_csv('/Users/weien/Desktop/kylab專案/勝杰的心肌梗塞大鼠分析/MIraw檔/MI前/wc1_clipECG.csv')

fs = 500

df_helth = pd.read_csv('/Users/weien/Desktop/kylab專案/勝杰的心肌梗塞大鼠分析/MIraw檔/MI前/wc1_clipECG.csv')
ecg_Health = df_helth['ECG']

ecg_Health = ecg_Health[464*fs : 488*fs]
# ecg_Health = ecg_Health[25000:85000]
ecg_Health = ecg_Health.reset_index(drop = True)

df_MI = pd.read_csv('/Users/weien/Desktop/kylab專案/勝杰的心肌梗塞大鼠分析/MIraw檔/MI後/wc3_clipECG.csv')
ecg_MI = df_MI['ECG']
ecg_MI = ecg_MI[(45*60+12)*fs : (45*60+43)*fs] 
# ecg_MI = ecg_MI[0:60000]
ecg_MI = ecg_MI.reset_index(drop = True)


medianfilter_size = 61
gaussian_filter_sigma =  0.005*fs #20
moving_average_ms = 0.5#1.25 
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.08*fs)*2  #草哥使用(0.3*fs)*2
rpeak_close_range = 0.04*fs #0.1*fs
lowpass_fq = 100
highpass_fq = 10


# ecg_Health = ecg_Health[0:5000]

# Health
medianEcg_Health, rpeak_index_Health = getRpeak.getRpeak_shannon(ecg_Health, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
rri_Health = np.diff(rpeak_index_Health)
rri_Health = rri_Health/(fs/1000)



# MI後
medianEcg_MI, rpeak_index_MI = getRpeak.getRpeak_shannon(ecg_MI, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
rri_MI = np.diff(rpeak_index_MI)
rri_MI = rri_MI/(fs/1000)


x_rri_Health = []
temp = 0
for i in range(len(rri_Health)):
    temp = temp+rri_Health[i]
    x_rri_Health.append(temp)
    
x_rri_MI = []
temp = 0
for i in range(len(rri_MI)):
    temp = temp+rri_MI[i]
    x_rri_MI.append(temp)
    
x_ecg_Health = np.linspace(0, len(medianEcg_Health)/fs, len(medianEcg_Health))
x_ecg_MI = np.linspace(0, len(medianEcg_MI)/fs, len(medianEcg_MI))

x_rri_Health = np.array(x_rri_Health)/1000
x_rri_MI = np.array(x_rri_MI)/1000

# 每秒為一Epoch，計算variance
rri_var_health = onesecond_rrivar(x_rri_Health, rri_Health)
rri_var_MI = onesecond_rrivar(x_rri_MI, rri_MI)


# 畫圖
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14,4)) 

ax1 = plt.subplot(3,1,1) 
ax1.set_title('Health (PS Stage)')
ax1.plot(x_ecg_Health, medianEcg_Health, 'black')
ax1.scatter(np.array(rpeak_index_Health)/fs, medianEcg_Health[rpeak_index_Health], alpha=0.5, c='r')
ax1.set_ylabel('ECG')
ax1.set_ylim(-200, 200)

rr_Health_add = interpolate_rri(rri_Health,fs)

ax2 = plt.subplot(3,1,2, sharex=ax1) 
ax2.plot(x_rri_Health[1:], rri_Health[1:], 'black')
# ax2.plot(rr_Health_add, 'black')
ax2.set_ylabel('RRI [ms]')
ax2.set_ylim(100,250)

ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(rri_var_health[1:], '-ko')
ax3.set_ylabel('RRI Var (1s)')
ax3.set_ylim(0,20)
ax3.set_xlabel('Time (s)')

plt.tight_layout()

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14,4)) 

ax1 = plt.subplot(3,1,1)
ax1.set_title('MI (PS Stage)')
ax1.plot(x_ecg_MI, medianEcg_MI, 'black')
ax1.scatter(np.array(rpeak_index_MI)/fs, medianEcg_MI[rpeak_index_MI], alpha=0.5, c='r')
ax1.set_ylabel('ECG')
ax1.set_ylim(-200, 200)

ax2 = plt.subplot(3,1,2, sharex=ax1)
ax2.plot(x_rri_MI[1:], rri_MI[1:], 'black')
ax2.set_ylabel('RRI [ms]')
ax2.set_ylim(100,250)

ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(rri_var_MI[1:], '-ko')
ax3.set_ylabel('RRI Var (1s)')
ax3.set_ylim(0,20)
ax3.set_xlabel('Time (s)')

plt.tight_layout()






