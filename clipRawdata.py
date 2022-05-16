#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:21:18 2021

@author: weien
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import signal
import numpy as np
import def_decodeECGdata as decodeECGdata
import def_clipConditionRawdata as clipConditionRawdata
import def_passFilter
import def_getRpeak as getRpeak



url = '/Users/weien/Desktop/ECG穿戴/HRV實驗/人體/Rawdata/220510郭葦珊'
filename = '220516B.RAW'
situation = 'Shake'

cliptime_start = '17:12:12'
cliptime_end = '17:14:52'

outputurl = url+'/'+situation+'.csv'
ecgrawdata_file1,ecgfq_file1,updatetime_file1 = decodeECGdata.openRawFile(url+'/'+filename) #minus_V = -0.9
ecgdata = decodeECGdata.get_data_complement(ecgrawdata_file1) #如需執行此行，rawdata換算成mV時則不須-0.9 minus_V = 0
minus_V = 0

# ecgdata = clipConditionRawdata.getConditionRawdata(cliptime_start, cliptime_end, ecgrawdata_file1, ecgfq_file1,updatetime_file1)
df = pd.DataFrame({situation:ecgdata})
# df.to_csv(outputurl)

rawdata_mV = ((np.array(ecgdata)*(1.8/65535)-minus_V)/500)*1000
plt.figure(figsize=(16,3))
plt.plot(ecgdata, c='black')
# plt.ylim(-0.2,0.5)


'''
baseline_1 = clipConditionRawdata.getConditionRawdata('16:41:14','16:54:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
stroop = clipConditionRawdata.getConditionRawdata('16:44:36','16:46:25',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_2 = clipConditionRawdata.getConditionRawdata('16:46:26','16:49:26',ecgrawdata_file1,ecgfq_file1,updatetime_file1)

df = pd.DataFrame({'ECG': baseline_2})
df.to_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/TestingData/yu_strooptest_base2.csv')

'''
'''
#210902資料分割
ecgrawdata_file1,ecgfq_file1,updatetime_file1 = decodeECGdata.openRawFile('210902e.241')
baseline_1 = clipConditionRawdata.getConditionRawdata('14:52:00','14:54:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
positive_touch = clipConditionRawdata.getConditionRawdata('14:54:00','14:57:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_2 = clipConditionRawdata.getConditionRawdata('14:57:00','15:00:21',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
negative_scared = clipConditionRawdata.getConditionRawdata('15:00:21','15:02:30',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_3 = clipConditionRawdata.getConditionRawdata('15:02:30','15:04:04',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
positive_play = clipConditionRawdata.getConditionRawdata('15:04:04','15:07:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_4 = clipConditionRawdata.getConditionRawdata('15:07:00','15:10:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
negative_seperate = clipConditionRawdata.getConditionRawdata('15:10:00','15:13:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_5 = clipConditionRawdata.getConditionRawdata('15:13:00','15:15:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
positive_eat = clipConditionRawdata.getConditionRawdata('15:15:00','15:19:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
ecgrawdata_file2,ecgfq_file2,updatetime_file2 = openRawFile('210902f.241')
baseline_6 = clipConditionRawdata.getConditionRawdata('15:20:00','15:24:00',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
walk = clipConditionRawdata.getConditionRawdata('15:24:00','15:26:00',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
'''


'''
#211014資料分割

ecgrawdata_file1,ecgfq_file1,updatetime_file1 = decodeECGdata.openRawFile('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/Rawdata/211014h.240')
baseline_1 = clipConditionRawdata.getConditionRawdata('18:31:40','18:34:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
petted = clipConditionRawdata.getConditionRawdata('18:34:00','18:36:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_2 = clipConditionRawdata.getConditionRawdata('18:36:16','18:38:16',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
scared = clipConditionRawdata.getConditionRawdata('18:38:17','18:40:17',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_3 = clipConditionRawdata.getConditionRawdata('18:40:17','18:42:17',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
play = clipConditionRawdata.getConditionRawdata('18:42:28','18:44:30',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_4 = clipConditionRawdata.getConditionRawdata('18:45:30','18:51:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
isolation = clipConditionRawdata.getConditionRawdata('18:51:00','18:53:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_5 = clipConditionRawdata.getConditionRawdata('18:54:40','18:56:40',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
eat = clipConditionRawdata.getConditionRawdata('18:57:00','18:59:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
baseline_6 = clipConditionRawdata.getConditionRawdata('19:00:00','19:02:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)
walk = clipConditionRawdata.getConditionRawdata('19:02:14','19:04:00',ecgrawdata_file1,ecgfq_file1,updatetime_file1)

#%%
ecgrawdata_file1,ecgfq_file1,updatetime_file1 = decodeECGdata.openRawFile('210902e.241')
negative_scared = clipConditionRawdata.getConditionRawdata('15:00:21','15:02:30',ecgrawdata_file1,ecgfq_file1,updatetime_file1)

'''
'''
#220215 Peace收案
ecgrawdata_file1,ecgfq_file1,updatetime_file1 = decodeECGdata.openRawFile('/Users/weien/Desktop/狗狗穿戴/HRV實驗/收案RawData/210215 Peace/220215a.240')
baseline_1 = clipConditionRawdata.getConditionRawdata('11:07:00','11:09:50',ecgrawdata_file1,ecgfq_file1,updatetime_file1)

ecgrawdata_file2,ecgfq_file2,updatetime_file2 = decodeECGdata.openRawFile('/Users/weien/Desktop/狗狗穿戴/HRV實驗/收案RawData/210215 Peace/220215b.240')
petted = clipConditionRawdata.getConditionRawdata('11:10:50','11:15:20',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
baseline_2 = clipConditionRawdata.getConditionRawdata('11:15:20','11:16:40',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
scared = clipConditionRawdata.getConditionRawdata('11:17:30','11:22:30',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
baseline_3 = clipConditionRawdata.getConditionRawdata('11:23:00','11:25:00',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
play = clipConditionRawdata.getConditionRawdata('11:25:00','11:30:00',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
baseline_4 = clipConditionRawdata.getConditionRawdata('11:31:00','11:33:30',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
isolation = clipConditionRawdata.getConditionRawdata('11:34:10','11:39:10',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
baseline_5 = clipConditionRawdata.getConditionRawdata('11:41:30','11:44:00',ecgrawdata_file2,ecgfq_file2,updatetime_file2)
eat = clipConditionRawdata.getConditionRawdata('11:44:40','11:49:40',ecgrawdata_file2,ecgfq_file2,updatetime_file2)

# plt.figure()
# a=range(0,len(ecgrawdata_file1))
# a=np.array(a)/250
# plt.plot(a,ecgrawdata_file1)

# plt.figure()
# b=range(0,len(ecgrawdata_file2))
# b=np.array(b)/250
# plt.plot(b,ecgrawdata_file2)

'''
#%%畫原始資料圖
'''
condition_list=[baseline_1,petted,baseline_2,scared,baseline_3,play,
                    baseline_4,isolation,baseline_5,eat]
title_list=['Baseline','Petted','Baseline','Scared','Baseline','Play','Baseline','Isolation','Baseline','Eat']

# condition_list=[baseline_1,petted,baseline_2,scared,baseline_3]
# title_list=['Baseline','Petted','Baseline','Scared','Baseline']

plt.figure(figsize=(8,6))
for i in range(len(condition_list)):
    plt.subplot(4,3,i+1)
    plt.plot(condition_list[i])
    plt.title(title_list[i])
    plt.ylim(0,80000)
plt.tight_layout()
'''
    
#%%切割一分鐘 自動判找訊號看起來比較好的
'''
#B1,Petted,B2,B5,Eat, 10000-20000
#Scared 5000-15000
#B3 20000-30000
#Play 15000-25000
#B4 55000-65000
#isolation 12000-22000
#B6 20000-30000
#walk超慘 17600-19000好一點 但還是偏雜

c_base1=baseline_1[10000:20000]
c_base2=baseline_2[10000:20000]
c_base3=baseline_3[20000:30000]
c_base4=baseline_4[55000:65000]
c_base5=baseline_5[10000:20000]
c_base6=baseline_6[20000:30000]

c_petted=petted[10000:20000]
c_scared=scared[5000:15000]

c_play=play[15000:25000]
c_isolation=isolation[12000:22000]
c_eat=eat[10000:20000]
c_walk=walk[10000:20000]


clip_negative_scared=negative_scared[31250:32250]
'''
#畫圖 因為內容太長所以分兩張畫
'''
cliprawdata_list=[c_base1,c_petted,c_base2,c_scared,c_base3,c_play,c_base4,c_isolation,c_base5,c_eat,c_base6,c_walk]

plt.figure(figsize=(12,24))
for i in range(6):
    plt.subplot(6,1,i+1)
    plt.plot(cliprawdata_list[i])
    plt.title(title_list[i])
    plt.ylim(10000,50000)
    plt.xlim(0,10000)
plt.tight_layout()

plt.figure(figsize=(12,24))
for i in range(6,len(cliprawdata_list)):
    plt.subplot(6,1,i-5)
    plt.plot(cliprawdata_list[i])
    plt.title(title_list[i])
    plt.ylim(10000,50000)
    plt.xlim(0,10000)
plt.tight_layout()
'''

#%%濾波取出0-33HZ
'''
lowpassdata_list=[]
for i in range(len(cliprawdata_list)):
    lowpass=passFilter.lowPassFilter(33,cliprawdata_list[i])
    lowpassdata_list.append(lowpass)
'''
    
'''
plt.figure(figsize=(12,24))
for i in range(6):
    plt.subplot(6,1,i+1)
    plt.plot(lowpassdata_list[i])
    plt.title(title_list[i])
    plt.ylim(10000,50000)
    plt.xlim(0,10000)
plt.tight_layout()

plt.figure(figsize=(12,24))
for i in range(6,len(lowpassdata_list)):
    plt.subplot(6,1,i-5)
    plt.plot(lowpassdata_list[i])
    plt.title(title_list[i])
    plt.ylim(10000,50000)
    plt.xlim(0,10000)
plt.tight_layout()
'''

#%%0-33HZ後取R peak 

# df = pd.DataFrame({'Petted':lowpassdata_list[1], 'Scared':lowpassdata_list[3],'Play':lowpassdata_list[5], 'Isolation':lowpassdata_list[7], 'Eat':lowpassdata_list[9]})


#touch(1):0.9 scared(3):找不到 play(5):1.05(吧) Isolation(7): 1.04 Eat(9):1.04
'''
data=lowpassdata_list[9]
# touch=touch.reset_index(drop=True)
df = pd.DataFrame({'ECG':data})
rawdatalist=df.ECG
ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,1.04)
'''
#找rpeak參數的方式
# for i in np.arange(0,2,0.1):

#     ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,1.5)
    
#     if len(ybeat)>1:
#         print('參數為{} beat長度為{}'.format(i,len(ybeat)))
#     elif len(ybeat)==0:
#         print('0')
        
'''
# 畫rpeak原始圖位置圖
plt.figure(figsize=(16,3))
plt.plot(x_time,rawdatalist)
plt.xlabel('Time')
plt.ylabel('ECG')
plt.title('Eat (0-33HZ)')
plt.xlim(0,10000)
plt.ylim()
plt.scatter(peaklist_time, ybeat, c='red') #Plot detected peaks
'''

#%%
'''
data_based=np.array(data)-np.mean(data)

#將rpeak的點改成0->濾掉r peak的部分
for i in range(len(peaklist_time)):
    rpeak_index=peaklist_time[i]
    
    for j in range(rpeak_index-12,rpeak_index+13):       
        if j<len(data_based):
            data_based[j]=0
        elif j>=len(data_based):
            break

#EMG再經過濾波
emg=passFilter.highPassFilter(33,data_based)




#畫刪除前後1秒的ECG資料
plt.figure(figsize=(16,3))
plt.plot(emg)
plt.ylabel('EMG')
plt.ylim(-5000,5000)
plt.xlabel('Time')
plt.title('Eat')
'''



    
    

    



