#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:57:44 2021

@author: weien
"""

# import statistics
# import dogHRV
# import tkinter as tk  
# from tkinter import filedialog  
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt

# ecgrawdata_file1,ecg_fq,update_datetime=dogHRV.openRawFile('210902e.241')
# rawdata = dogHRV.getConditionRawdata('14:52:00','15:04:00',ecgrawdata_file1,ecg_fq,update_datetime)

def useFilterDatadrawHist(filename,column):
    df_1=pd.read_excel(filename)
    rawdata=df_1[column+'_filter']
    df = pd.DataFrame({'HR':rawdata})
    df = df.dropna()
    rawdatalist=df.HR
    ybeat,x_time,peaklist_time=getYValueofRPeak(df,rawdatalist)
    
    #畫直方圖
    plt.hist(ybeat)
    plt.title(column+' (0-33HZ)')
    plt.show()
    
    ybeat_median=np.median(ybeat)
    
    return ybeat_median


#-----detect R-peak-----#

def getYValueofRPeak(df,rawdatalist):
    HR_rollingmean=[]
    
    hrw = 0.3 #One-sided window size, as proportion of the sampling frequency
    fs = 250 #The example dataset was recorded at 100Hz
    mov_avg = rawdatalist.rolling(int(hrw*fs)).mean() #Calculate moving average
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    
    avg_hr = (np.mean(rawdatalist))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.03 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    df['HR_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of interest
    window = []
    peaklist = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in rawdatalist:
        rollingmean = df.HR_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            #maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    ybeat = [rawdatalist[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
    
    x_time=range(len(rawdatalist))
    
    peaklist_time = [x_time[j] for j in peaklist]
    return ybeat,x_time,peaklist_time
#Note參數調整:
    #filter:scared:1.03
    #touch:1.0515

#%%
#抓原始資料
df_1=pd.read_csv('negative_scared.csv')
rawdata=df_1['RawData']
# rawdata=rawdata[1800:16800]
rawdata=rawdata[22700:25700]
rawdata=rawdata.reset_index(drop=True)

'''
#Note:選取data:（一分鐘）
    #scared:[1800:16800]
    #touch:[25000:40000]
'''

#找Rpeak
df = pd.DataFrame({'ECG':rawdata})
rawdatalist=df.ECG
ybeat,x_time,peaklist_time=getYValueofRPeak(df,rawdatalist)



#%%

#RR interval
rrinterval_list=[]
for i in range(1,len(peaklist_time)):
    rrinterval_list.append(peaklist_time[i]-peaklist_time[i-1])


#畫rpeak原始圖位置圖
plt.figure()
plt.plot(x_time, df['ECG'])
plt.xlabel('Time')
plt.ylabel('ECG')
plt.title('Scared (0-33HZ)')
plt.scatter(peaklist_time, ybeat, c='red') #Plot detected peaks

#畫R-R直方圖
plt.figure()
plt.hist(rrinterval_list)
plt.title('Scared (0-33HZ)')
plt.xlabel('R-R intervals')
plt.show()

#畫R振幅直方圖
plt.figure()
plt.hist(ybeat)
plt.title('Scared (0-33HZ)')
plt.xlabel('R amplitude')
plt.show()


#直方圖的值
# filtery=useFilterDatadrawHist('Condition_filter.xls','negative_scared')
# useFilterDatadrawHist('Condition_filter.xls','Walk')

#撈negative_scared 算r peak
# df_1=pd.read_excel('Condition_filter.xls')
# df_1=df_1.dropna(subset=['negative_scared_filter'])

# rawdata=(df_1['negative_scared_filter'])[27000:30000]
# df = pd.DataFrame({'HR':rawdata})
# df = df.dropna()
# rawdatalist=df.HR
# ybeat,x_time,peaklist_time=getYValueofRPeak(df,rawdatalist)



# df_1=pd.read_excel('Condition_filter.xls')
# rawdata=df_1['negative_scared_filter'][22700:25700]
# df = pd.DataFrame({'HR':rawdata})
# rawdatalist=df.HR
# ybeat_noise,x_time_noise,peaklist_time_noise=getYValueofRPeak(df,rawdatalist)


#畫rpeak位置圖
# plt.plot(x_time, df.HR)
# plt.xlabel('Time')
# plt.ylabel('ECG')
# plt.scatter(peaklist_time, ybeat, c='red') #Plot detected peaks


# df=pd.read_excel('Condition_filter.xls')
# df_scared=df[['negative_scared_filter']]
# rawdatalist=df_scared['negative_scared_filter']
# ybeat,x_time,peaklist_time=getYValueofRPeak(df_scared,rawdatalist)




