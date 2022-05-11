#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 23:31:45 2022

@author: weien
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 30秒為一點 畫圖
# 2. 將每個情境做平均 ->一個情境一個點
# 3. 前後相減
# 4. 單執行一個情境

#%% 30秒為一點 畫圖

'''
df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/HRV實驗/人體/220510分析資料.xlsx')
# df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/HRV實驗/人體/220510Shake.xlsx')

mean = df['Mean']
sd = df['SD']
rmssd = df['RMSSD']
skewness = df['Skewness']
kurtosis = df['Kurtosis']
nn50 = df['NN50']
pnn50 = df['pNN50']
EMG_RMS = df['EMG_RMS']

time = np.linspace(0, 31*30, 31)
# time = np.linspace(0, 4*30, 4)


ylabel_xlocation = -0.04
ylabel_ylocation = 0.5

# plt.figure(figsize=(16,14))
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12,12))  

ax1 = plt.subplot(8,1,1)
ax1.plot(time, mean, c='black', marker = '.')
ax1.set_ylim(500,1000)
ax1.set_ylabel('Mean')
ax1.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax2 = plt.subplot(8,1,2)
ax2.plot(time, sd, c='black', marker = '.')
ax2.set_ylim(0,100)
ax2.set_ylabel('SD')
ax2.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax3 = plt.subplot(8,1,3)
ax3.plot(time, rmssd, c='black', marker = '.')
ax3.set_ylim(0,100)
ax3.set_ylabel('RMSSD')
ax3.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax4 = plt.subplot(8,1,4)
ax4.plot(time, skewness, c='black', marker = '.')
ax4.set_ylim(-5,5)
ax4.set_ylabel('Skew')
ax4.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax5 = plt.subplot(8,1,5)
ax5.plot(time, kurtosis, c='black', marker = '.')
ax5.set_ylim(-25,25)
ax5.set_ylabel('Kurt')
ax5.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax6 = plt.subplot(8,1,6)
ax6.plot(time, nn50, c='black', marker = '.')
ax6.set_ylim(0,20)
ax6.set_ylabel('NN50')
ax6.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax7 = plt.subplot(8,1,7)
ax7.plot(time, pnn50, c='black', marker = '.')
ax7.set_ylim(0,0.5)
ax7.set_ylabel('pNN50')
ax7.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax8 = plt.subplot(8,1,8)
ax8.plot(time, EMG_RMS, c='black', marker = '.')
ax8.set_ylim(0,0.1)
ax8.set_ylabel('EMG-RMS')
ax8.get_yaxis().set_label_coords(ylabel_xlocation,0.5)

plt.xlabel('Time(s)')


plt.tight_layout()
'''

#%%將每個情境做平均 ->一個情境一個點
'''
df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/HRV實驗/人體/220510分析資料.xlsx')

situation_list = ['Baseline1', 'Stroop', 'Baseline2', 'Arithmetic', 'Baseline3', 'Speech', 'Baseline4']
mean =[]
sd = []
rmssd = []
skewness = []
kurtosis = []
nn50 = []
pnn50 = []
EMG_RMS = []

for i in range(len(situation_list)):
               
    df_situation = df[df['Situation'] == situation_list[i]]  
    mean_mean = df_situation['Mean'].mean()
    sd_mean = df_situation['SD'].mean()
    rmssd_mean = df_situation['RMSSD'].mean()
    skewness_mean = df_situation['Skewness'].mean()
    kurtosis_mean = df_situation['Kurtosis'].mean()
    nn50_mean = df_situation['NN50'].mean()
    pnn50_mean = df_situation['pNN50'].mean()
    EMG_RMS_mean = df_situation['EMG_RMS'].mean()
    
    mean.append(mean_mean)
    sd.append(sd_mean)
    rmssd.append(rmssd_mean)
    skewness.append(skewness_mean)
    kurtosis.append(kurtosis_mean)
    nn50.append(nn50_mean)
    pnn50.append(pnn50_mean)
    EMG_RMS.append(EMG_RMS_mean)
    

time = np.linspace(0, 120*7, 7)
x_ticks = ['Baseline', 'Stroop', 'Baseline', 'Arithmetic', 'Baseline', 'Speech', 'Baseline']


ylabel_xlocation = -0.07
ylabel_ylocation = 0.5

# plt.figure(figsize=(16,14))
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8,8))  

ax1 = plt.subplot(8,1,1)
ax1.plot(time, mean, c='black', marker = '.')
ax1.set_ylim(500,1000)
ax1.set_ylabel('Mean')
ax1.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
plt.xticks(time, x_ticks)

ax2 = plt.subplot(8,1,2)
ax2.plot(time, sd, c='black', marker = '.')
ax2.set_ylim(0,100)
ax2.set_ylabel('SD')
ax2.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
plt.xticks(time, x_ticks)

ax3 = plt.subplot(8,1,3)
ax3.plot(time, rmssd, c='black', marker = '.')
ax3.set_ylim(0,100)
ax3.set_ylabel('RMSSD')
ax3.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
plt.xticks(time, x_ticks)

ax4 = plt.subplot(8,1,4)
ax4.plot(time, skewness, c='black', marker = '.')
ax4.set_ylim(-5,5)
ax4.set_ylabel('Skew')
ax4.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
plt.xticks(time, x_ticks)

ax5 = plt.subplot(8,1,5)
ax5.plot(time, kurtosis, c='black', marker = '.')
ax5.set_ylim(-25,25)
ax5.set_ylabel('Kurt')
ax5.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
plt.xticks(time, x_ticks)

ax6 = plt.subplot(8,1,6)
ax6.plot(time, nn50, c='black', marker = '.')
ax6.set_ylim(0,20)
ax6.set_ylabel('NN50')
ax6.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
plt.xticks(time, x_ticks)

ax7 = plt.subplot(8,1,7)
ax7.plot(time, pnn50, c='black', marker = '.')
ax7.set_ylim(0,0.5)
ax7.set_ylabel('pNN50')
ax7.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
plt.xticks(time, x_ticks)

ax8 = plt.subplot(8,1,8)
ax8.plot(time, EMG_RMS, c='black', marker = '.')
ax8.set_ylim(0,0.1)
ax8.set_ylabel('EMG-RMS')
ax8.get_yaxis().set_label_coords(ylabel_xlocation,0.5)

plt.xlabel('Time(s)')
plt.xticks(time, x_ticks)


plt.tight_layout()

'''

#%% 前後相減


'''
df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/HRV實驗/人體/220510分析資料.xlsx')

mean = df['Mean']
sd = df['SD']
rmssd = df['RMSSD']
skewness = df['Skewness']
kurtosis = df['Kurtosis']
nn50 = df['NN50']
pnn50 = df['pNN50']
EMG_RMS = df['EMG_RMS']

mean = np.diff(mean)
sd = np.diff(sd)
rmssd = np.diff(rmssd)
skewness = np.diff(skewness)
kurtosis = np.diff(kurtosis)
nn50 = np.diff(nn50)
pnn50 = np.diff(pnn50)
EMG_RMS = np.diff(EMG_RMS)

time = np.linspace(0, 26, 26)


ylabel_xlocation = -0.05
ylabel_ylocation = 0.5

# plt.figure(figsize=(16,14))
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12,12))  

ax1 = plt.subplot(8,1,1)
ax1.plot(time, mean, c='black', marker = '.')
ax1.set_ylim(-200,200)
ax1.set_ylabel('Mean')
ax1.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax2 = plt.subplot(8,1,2)
ax2.plot(time, sd, c='black', marker = '.')
ax2.set_ylim(-100,100)
ax2.set_ylabel('SD')
ax2.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax3 = plt.subplot(8,1,3)
ax3.plot(time, rmssd, c='black', marker = '.')
ax3.set_ylim(-100,100)
ax3.set_ylabel('RMSSD')
ax3.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax4 = plt.subplot(8,1,4)
ax4.plot(time, skewness, c='black', marker = '.')
ax4.set_ylim(-5,5)
ax4.set_ylabel('Skew')
ax4.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax5 = plt.subplot(8,1,5)
ax5.plot(time, kurtosis, c='black', marker = '.')
ax5.set_ylim(-20,20)
ax5.set_ylabel('Kurt')
ax5.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax6 = plt.subplot(8,1,6)
ax6.plot(time, nn50, c='black', marker = '.')
ax6.set_ylim(-20,20)
ax6.set_ylabel('NN50')
ax6.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax7 = plt.subplot(8,1,7)
ax7.plot(time, pnn50, c='black', marker = '.')
ax7.set_ylim(-0.5,0.5)
ax7.set_ylabel('pNN50')
ax7.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)

ax8 = plt.subplot(8,1,8)
ax8.plot(time, EMG_RMS, c='black', marker = '.')
ax8.set_ylim(-0.05,0.05)
ax8.set_ylabel('EMG-RMS')
ax8.get_yaxis().set_label_coords(ylabel_xlocation,0.5)

plt.xlabel('Time(s)')


plt.tight_layout()

'''

#%% 單執行一個情境

df = pd.read_excel('/Users/weien/Desktop/ECG穿戴/HRV實驗/人體/220510分析資料.xlsx')

situation = 'Shake'

mean =[]
sd = []
rmssd = []
skewness = []
kurtosis = []
nn50 = []
pnn50 = []
EMG_RMS = []


               
df_situation = df[df['Situation'] == situation]  
mean = df_situation['Mean']
sd = df_situation['SD']
rmssd = df_situation['RMSSD']
skewness = df_situation['Skewness']
kurtosis = df_situation['Kurtosis']
nn50 = df_situation['NN50']
pnn50 = df_situation['pNN50']
EMG_RMS = df_situation['EMG_RMS']
    

time = np.linspace(0, 30*4, 4)
# time = [1,2,3,4]

ylabel_xlocation = -0.12
ylabel_ylocation = 0.5

# plt.figure(figsize=(16,14))
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5,8))  



ax1 = plt.subplot(8,1,1)
ax1.plot(time, mean, c='black', marker = '.')
ax1.set_ylim(500,1000)
ax1.set_ylabel('Mean')
ax1.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
ax1.get_xaxis().set_visible(False)
ax1.set_title(situation)

ax2 = plt.subplot(8,1,2)
ax2.plot(time, sd, c='black', marker = '.')
ax2.set_ylim(0,200)
ax2.set_ylabel('SD')
ax2.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
ax2.get_xaxis().set_visible(False)

ax3 = plt.subplot(8,1,3)
ax3.plot(time, rmssd, c='black', marker = '.')
ax3.set_ylim(0,300)
ax3.set_ylabel('RMSSD')
ax3.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
ax3.get_xaxis().set_visible(False)

ax4 = plt.subplot(8,1,4)
ax4.plot(time, skewness, c='black', marker = '.')
ax4.set_ylim(-5,5)
ax4.set_ylabel('Skew')
ax4.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
ax4.get_xaxis().set_visible(False)

ax5 = plt.subplot(8,1,5)
ax5.plot(time, kurtosis, c='black', marker = '.')
ax5.set_ylim(-30,30)
ax5.set_ylabel('Kurt')
ax5.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
ax5.get_xaxis().set_visible(False)

ax6 = plt.subplot(8,1,6)
ax6.plot(time, nn50, c='black', marker = '.')
ax6.set_ylim(0,20)
ax6.set_ylabel('NN50')
ax6.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
ax6.get_xaxis().set_visible(False)

ax7 = plt.subplot(8,1,7)
ax7.plot(time, pnn50, c='black', marker = '.')
ax7.set_ylim(0,0.5)
ax7.set_ylabel('pNN50')
ax7.get_yaxis().set_label_coords(ylabel_xlocation, ylabel_ylocation)
ax7.get_xaxis().set_visible(False)

ax8 = plt.subplot(8,1,8)
ax8.plot(time, EMG_RMS, c='black', marker = '.')
ax8.set_ylim(0,0.2)
ax8.set_ylabel('EMG-RMS')
ax8.get_yaxis().set_label_coords(ylabel_xlocation,0.5)

plt.xlabel('Time(s)')


plt.tight_layout()








