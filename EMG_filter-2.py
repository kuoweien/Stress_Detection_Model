#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:17:04 2021

@author: weien
"""
import pandas as pd
from scipy.signal import butter,filtfilt
from scipy import signal
import numpy as np
import datetime
import matplotlib.pyplot as plt
import defgetRpeak as getRpeak
import def_linearFunc

#220208: 要先將原始訊號轉換成電壓 (電壓要再1000變mV) 再做後續動作如抓rpeak等

#將R peak點改成以線性y=ax+b的方式補點 輸入原始資料及rpeak的位置點
def fillRpeakwithLinear(rawdata, rpeakindex): #只刪除rpeak
    
    rpeak_linear=rawdata
    
    #將rpeak的點改成0->濾掉r peak的部分
    for i in range(len(rpeakindex)):
        rpeak_index=rpeakindex[i]
        
        startX=rpeak_index-12
        startY=rawdata[rpeak_index-12]
        endX=rpeak_index+13
        
        if len(rawdata)<endX:
            endX=len(rawdata)
            endY=rawdata[-1]
        elif len(rawdata)>=endX:
            endX=endX
            endY=rawdata[rpeak_index+13]
        
        linearOutput=linearFunc.linearFunc([startX,startY],[endX,endY]) #linearFunc.py引入 #共前後1秒
        firstindex=linearOutput[0][0]
        # lastindex=linearOutput[-1][0]
        
        for j in range(0,len(linearOutput)):
            rpeak_linear[(j+firstindex)] = linearOutput[j][1]
        
    return linearOutput,rpeak_linear

#包含T波一起刪除
def fillRTpeakwithLinear(rawdata, rpeakindex): #刪除rtpeak
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
        
        linearOutput=def_linearFunc.linearFunc([startX,startY],[endX,endY]) #linearFunc.py引入 #共前後1秒
        firstindex=linearOutput[0][0]
        # lastindex=linearOutput[-1][0]
        
        for j in range(0,len(linearOutput)):
            rpeak_linear[(j+firstindex)] = linearOutput[j][1]
        
    return linearOutput,rpeak_linear #輸出linear(忘記是啥)跟已刪除rt波的圖EMG



#原始訊號-從ECG中抓的Rpeak  
# df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/scared.csv').iloc[5000:15000]
df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/petted.csv').iloc[10000:12500]
# df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/isolation.csv').iloc[18000:20500]
touch=df['petted']
touch=touch.reset_index(drop=True)


#經過低通
fq=33
b, a = signal.butter(8, (2*fq)/250, 'lowpass')  #濾除125HZ以上的頻率，wn=2*125/250=1
ecg = signal.filtfilt(b, a, touch) 
df = pd.DataFrame({'ECG':ecg})
rawdatalist=df.ECG
# ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,1.063) #for scared[31250:32250]

rawdatalist_V=((rawdatalist*(1.8/65535)-0.9)/500)*1000

ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist_V,2.5) #1.052->2.5

x_datetime=[]
for i in x_time:
    x_datetime.append(datetime.timedelta(seconds=(x_time[i]/250))) #x軸點數轉成秒

 

# 畫rpeak原始圖位置圖
pltylim_high=0.5
pltylim_low=-0.5

plt.figure(figsize=(12,6))
plt.subplot(3,1,1)     
plt.ylim(pltylim_low,pltylim_high)
# rawdata_V=(rawdatalist*(1.8/65535)-0.9)/500 
# xdata=(rawdata_V-np.mean(rawdata_V))
plt.plot(np.array(x_time)/250, rawdatalist_V)
plt.xlabel('Time (s)')
plt.ylabel('ECG')
plt.title('ECG (0-33HZ)')
plt.scatter(np.array(peaklist_time)/250, ybeat , c='red') 

# linearOutput,rpeakfillbylinear=fillRpeakwithLinear(rawdatalist_V, peaklist_time)
linearOutput2,r_t_peakfillbylinear=fillRTpeakwithLinear(rawdatalist_V, peaklist_time)

        

#畫刪除前後1秒的ECG資料
plt.subplot(3,1,2)
# excludeECG_V=(rawdata_Rpeakfillbylinear*(1.8/65535)-0.9)/500
plt.plot(np.array(x_time)/250,r_t_peakfillbylinear)
# plt.plot(r_t_peakfillbylinear)
plt.ylim(pltylim_low,pltylim_high)
plt.xlabel('Time (s)')
plt.title('EMG')

# 針對補0：為了計算RMS 需排除補0的值
# EMG_withoutzero=excludeECG[excludeECG!=0] 
# rms = np.sqrt(np.mean(EMG_withoutzero**2))
    

#高通濾波   
fq=10
b, a = signal.butter(5, (2*fq)/250, 'highpass')   #濾除33HZ以下的頻率，wn=2*125/250=1
excludeECG_EMG = signal.filtfilt(b, a, r_t_peakfillbylinear) 


plt.subplot(3,1,3)
plt.plot(np.array(x_time)/250,excludeECG_EMG)
plt.xlabel('Time (s)')
plt.ylim(pltylim_low,pltylim_high)
plt.title('EMG ({}-250HZ)'.format(fq))

plt.tight_layout()





