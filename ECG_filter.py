#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:29:55 2021

@author: weien
"""

import pandas as pd
from scipy.signal import butter,filtfilt
from scipy import signal
import matplotlib.pyplot as plt
import defgetRpeak as getRpeak
import numpy as np

df=pd.read_csv('negative_scared.csv')
data_scared=df['RawData']

df=pd.read_csv('positive_touch.csv')
data_touch=df['RawData']


#%%

#濾害怕
#低通濾波
#滤除33hz以上频率成分，即截至频率为33hz，则wn=2*33/250=0.02
#滤除10hz以上频率成分，即截至频率为10hz，则wn=2*10/250=0.08
b, a = signal.butter(8, 0.08, 'lowpass')   #配置濾波器 8 表示濾波器的階數
data_lowfilter = signal.filtfilt(b, a, data_scared) 
# data_lowfilter=data_lowfilter[30550:32250]
#高通濾波
# 滤除33hz以下频率成分，即截至频率为33hz，则wn=2*33/250=0.02
# 滤除1hz以下频率成分，即截至频率为1hz，则wn=2*1/250=0.008
b, a = signal.butter(8, 0.008, 'highpass')
data_scared_1to33 = signal.filtfilt(b, a, data_lowfilter)
# data_scared_1to33=data_scared_1to33[29750:32250]#10秒
data_scared_1to33=data_scared_1to33[1800:16800]#1分鐘

#濾撫摸
b, a = signal.butter(8, 0.08, 'lowpass')   #配置濾波器 8 表示濾波器的階數
data_lowfilter = signal.filtfilt(b, a, data_touch) 
b, a = signal.butter(8, 0.008, 'highpass')
data_touch_1to33 = signal.filtfilt(b, a, data_lowfilter)
# data_touch_1to33=data_touch_1to33[22500:25000]#10秒
data_touch_1to33=data_touch_1to33[25000:40000]#1分鐘

#%%濾完的data抓Rpeak

df = pd.DataFrame({'ECG':data_touch_1to33})
rawdatalist=df.ECG
ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,20)


#畫rpeak原始圖位置圖
# plt.figure()
# plt.plot(x_time, df['ECG'])
# plt.xlabel('Time')
# plt.ylabel('ECG')
# plt.title('Scared (1-10HZ)')
# plt.ylim(-2000,3000)
# plt.scatter(peaklist_time, ybeat, c='red') #Plot detected peaks

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



def deleteExtremeHistValue(rrinterval_list):#dict新增被刪除的value dic

    hist=plt.hist(rrinterval_list,bins=10,color='black')
    plt.xlim(100,1000)
    plt.ylim(0,40)
    plt.title('Touch')
    hist_dict = {'Condition':'scared','count': hist[0], 'value': hist[1]}

    max_index=hist_dict['count'].tolist().index(max(hist_dict['count']))
    max_value=hist_dict['value'][max_index]
    range_value=[max_value-0.5*max_value,max_value+0.5*max_value]
    
    delvalue=[]
    
    for i in range(len(rrinterval_list)):
        if rrinterval_list[i]<range_value[0] or rrinterval_list[i]>range_value[1]:
            delvalue.append(rrinterval_list[i])
    
    
    rri_keep_value=set(rrinterval_list)-set(delvalue)
    
    return rri_keep_value
    

rri_keep_value_scared=deleteExtremeHistValue(rrinterval_list)
plt.figure()
plt.hist(rri_keep_value_scared,bins=5,color='black')
plt.xlim(100,1000)
plt.ylim(0,40)
plt.title('Touch (filter by hist)')




'''
max_index=hist_scared_dict['count'].tolist().index(max(hist_scared_dict['count']))
max_value=hist_scared_dict['value'][max_index]
range_value=[max_value-0.5*max_value,max_value+0.5*max_value]

delvalue=[]

for i in range(len(hist_scared_dict['value'])):
    value=hist_scared_dict['value'][i]
    if value<range_value[0] or value>range_value[1]:
        delvalue.append(value)


keep_value=set(hist_scared_dict['value'])-set(delvalue)
dic2={'keep_value':keep_value}
hist_scared_dict.update(dic2)

'''


        
'''     
#畫positive撫摸

df = pd.DataFrame({'ECG':data_touch_1to33})
rawdatalist=df.ECG
ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,10)

plt.figure()
plt.plot(x_time, df['ECG'])
plt.xlabel('Time')
plt.ylabel('ECG')
plt.title('Touch (1-10HZ)')
plt.ylim(-2000,3000)
plt.scatter(peaklist_time, ybeat, c='red') #Plot detected peaks

#RR interval
rrinterval_list=[]
for i in range(1,len(peaklist_time)):
    rrinterval_list.append((peaklist_time[i]-peaklist_time[i-1])*4)

#畫RR直方圖
plt.figure()
plt.hist(rrinterval_list)
plt.title('Touch')
plt.xlabel('R-R intervals')
plt.ylim(0,8)
plt.show()
'''
