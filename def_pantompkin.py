#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:40:51 2022

@author: weien
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import def_passFilter as bandfilter
import def_linearFunc
import math
    

def pantomskin(rawdata):
    rawdata=rawdata.reset_index(drop=True)
    ecg_lowpass=bandfilter.lowPassFilter(15,rawdata)        #低通
    ecg_bandpass = bandfilter.highPassFilter(5,ecg_lowpass)        #高通
    ecg_defivative = defivative(ecg_bandpass)       #導數
    ecg_square = np.square(ecg_defivative)       #平方
    movingwindow= movingaverage(ecg_defivative)     #moving average
    peaks_x, peaks_y = findpeak(ecg_defivative)
    detedted_rpeak_x,detedted_rpeak_y = detectRpeak(rawdata, peaks_x, peaks_y)  
    return detedted_rpeak_x,detedted_rpeak_y
    
    
def defivative(data_y): #微分
    x=range(len(data_y))
    
    dy = np.zeros(data_y.shape,np.float)
    dy[0:-1] = np.diff(data_y)/np.diff(x) #每一格的斜率 diff是前後相減  [0:-1]是最後一個不取(i.e.從0取到倒數第二個)
    dy[-1] = (data_y[-1] - data_y[-2])/(x[-1] - x[-2])
    
    return dy

def medfilt (x, k): #x是訊號 k是摺照大小 
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

def reversesignal(dataraw): #反轉訊號
    dataraw = dataraw.values
    median = np.median(dataraw)
    reverse_ecg = median-(dataraw-median)    
    return reverse_ecg


def movingaverage(ecg_square): #sliding window
    #sliding window
    moving_average = []
    for i in range(len(ecg_square)):
        moving_average.append(np.mean(ecg_square[i:i+30])) #視窗大小為120ms(30個點)
    moving_average=np.array(moving_average) 
    
    return moving_average

def findpeak(data):
    peaks_x, _ = find_peaks(data, distance=120) #findpeak 依照論文應該要150ms 因為取樣頻率不同（論文為200HZ, 我們為250HZ） 為了取整數所以取120ms 250*0.12 = 30  觀察結果圖 嘗試抓150
    peaks_y = data[peaks_x] #findpeak的y值
    
    return peaks_x, peaks_y
    
    
    

# 參考資料 https://github.com/c-labpl/qrs_detector/blob/master/QRSDetectorOffline.py
def detectRpeak(ecg_raw, peaks_x, peaks_y): #決策Threshold找Rpeak

    qrs_peaks_indices = np.array([], dtype=int)
    noise_peaks_indices = np.array([], dtype=int)
    refractory_period = 120
    threshold_value = 0.0
    qrs_peak_filtering_factor = 0.125
    qrs_peak_value = 0.0
    noise_peak_filtering_factor = 0.125
    qrs_noise_diff_weight = 0.25
    noise_peak_value = 0.0
    
    for detected_peak_index, detected_peaks_value in zip(peaks_x, peaks_y):
    
        try:
            last_qrs_index = qrs_peaks_indices[-1]
        except IndexError:
            last_qrs_index = 0
        
        # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
        if detected_peak_index - last_qrs_index > refractory_period or not qrs_peaks_indices.size:
            # Peak must be classified either as a noise peak or a QRS peak.
            # To be classified as a QRS peak it must exceed dynamically set threshold value.
            if detected_peaks_value > threshold_value:
                qrs_peaks_indices = np.append(qrs_peaks_indices, detected_peak_index)
        
                # Adjust QRS peak value used later for setting QRS-noise threshold.
                qrs_peak_value = qrs_peak_filtering_factor * detected_peaks_value + \
                                      (1 - qrs_peak_filtering_factor) * qrs_peak_value
            else:
                noise_peaks_indices = np.append(noise_peaks_indices, detected_peak_index)
        
                # Adjust noise peak value used later for setting QRS-noise threshold.
                noise_peak_value = noise_peak_filtering_factor * detected_peaks_value + \
                                        (1 - noise_peak_filtering_factor) * noise_peak_value
        
            # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
            threshold_value = noise_peak_value + \
                                   qrs_noise_diff_weight * (qrs_peak_value - noise_peak_value)
        
    # Create array containing both input ECG measurements data and QRS detection indication column.
    # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
    measurement_qrs_detection_flag = np.zeros([len(ecg_raw), 1])
    measurement_qrs_detection_flag[qrs_peaks_indices] = 1
    detedted_rpeak_x = qrs_peaks_indices #自己加：最後為Rpeak的index點
    detedted_rpeak_y = ecg_raw[detedted_rpeak_x] #自己加：最後為Rpeak的value
    # ecg_data_detected = np.append(ecg_dataraw, measurement_qrs_detection_flag, 1) #自己註解掉
    
    return detedted_rpeak_x,detedted_rpeak_y


#%%決策演法

def ecgfindthemaxvalue(rawdata, rpeak_x, range_n): #Decision rule找最大值(因前面會抓錯)在rpeak附近100點找最大值 input原始data, detected rpeak, 找尋最大值範圍
    newrpeak = pd.Series()
    for i in range(len(rpeak_x)):
        if rpeak_x[i]-int(range_n/2)<0:
            range_list = rawdata[0:rpeak_x[i]+int(range_n/2)]
        elif rpeak_x[i]+int(range_n/2)>len(rawdata):
            range_list = rawdata[rpeak_x[i]-int(range_n/2):len(rawdata)]
        else:  
            range_list = rawdata[rpeak_x[i]-int(range_n/2):rpeak_x[i]+int(range_n/2)]
        min_location = range_list.nlargest(1) #Series取最小值 取最大值為nlargest 最小值為nsmallest
        newrpeak = newrpeak.append(min_location)

    newdetedted_rpeak_x = newrpeak.index.values.tolist() 
    newdetedted_rpeak_y = newrpeak.tolist()
    
    return newdetedted_rpeak_x, newdetedted_rpeak_y

def ecgfindtheminvalue(rawdata, rpeak_x, range_n): #因前面會抓錯 所以直接找在附近100點的最小值
    newrpeak = pd.Series()
    for i in range(len(rpeak_x)):
        if rpeak_x[i]-int(range_n/2) < 0: #若第一點小於扣除的長度
            range_list = rawdata[0:rpeak_x[i]+int(range_n/2)] #以0為起點
            
        elif rpeak_x[i]+int(range_n/2) > len(rawdata): #若最後一點超過總長度
            range_list = rawdata[rpeak_x[i]-int(range_n/2):len(rawdata)]
        
        else:  
            range_list = rawdata[rpeak_x[i]-int(range_n/2):rpeak_x[i]+int(range_n/2)]
        
        min_location = range_list.nsmallest(1) #Series取最小值 取最大值為nlargest 最小值為nsmallest
        newrpeak = newrpeak.append(min_location)

    newdetedted_rpeak_x = newrpeak.index.values.tolist() 
    newdetedted_rpeak_y = newrpeak.tolist()
    
    return newdetedted_rpeak_x, newdetedted_rpeak_y
#包含T波一起刪除

def fillRTpeakwithLinear(rawdata, rpeakindex, qrs_range, tpeak_range): #原始資料跟rpeak_x #刪除rtpeak
    emgwithlinear = rawdata
    
    pre_range = math.floor(qrs_range/2)
    after_range = round(qrs_range/2)
    

    #將rpeak的點改成0->濾掉r peak的部分
    for i in range(len(rpeakindex)):
        rpeak_index=rpeakindex[i]
        if rpeak_index<pre_range:
            startX=0
            startY=emgwithlinear[0]
        
        elif rpeak_index>=pre_range: 
            startX=rpeak_index-pre_range
            startY=emgwithlinear[rpeak_index-pre_range]
            
        endX=rpeak_index+after_range+tpeak_range
        
        if len(emgwithlinear)<endX:
            endX=len(emgwithlinear)
            endY=emgwithlinear[len(emgwithlinear)-1]
        elif len(emgwithlinear)>=endX:
            endX=endX
            endY=emgwithlinear[rpeak_index+after_range+tpeak_range]
        
        linearOutput=def_linearFunc.linearFunc([startX,startY],[endX,endY]) #linearFunc.py引入 #共前後1秒
        firstindex=linearOutput[0][0]
        
        for j in range(0,len(linearOutput)):
            emgwithlinear[(j+firstindex)] = linearOutput[j][1]
        
    return emgwithlinear     # Output已刪除rt波的圖EMG, 有線性補點之值

def deleteRTpeak(rawdata, rpeakindex, qrs_range, tpeak_range): #與def fillRTpeakwithLinear相同 可以放在一起寫（要再修）
    emg_nolinear = rawdata
    
    sliplist_emg = []
    
    pre_range = math.floor(qrs_range/2)
    after_range = round(qrs_range/2)
    
    #將rpeak的點改成0->濾掉r peak的部分
    emg_startX = 0
    for i in range(len(rpeakindex)):
        
        rpeak_index=rpeakindex[i]
        if rpeak_index<pre_range:
            startX=0
            startY=emg_nolinear[0]
        
        elif rpeak_index>=pre_range: 
            startX=rpeak_index-pre_range
            startY=emg_nolinear[rpeak_index-pre_range]
            
        endX=rpeak_index+after_range+tpeak_range
        
        if len(emg_nolinear)<endX:
            endX=len(emg_nolinear)
            endY=emg_nolinear[len(emg_nolinear)-1]
        elif len(emg_nolinear)>=endX:
            endX=endX
            endY=emg_nolinear[rpeak_index+after_range+tpeak_range]
        
        linearOutput=def_linearFunc.linearFunc([startX,startY],[endX,endY]) #linearFunc.py引入 #共前後1秒
        firstindex=linearOutput[0][0]
        
        for j in range(0,len(linearOutput)):
            emg_nolinear[(j+firstindex)] = 0
            
        sliplist_emg.append(rawdata[emg_startX:startX])
        emg_startX = endX

            
    return emg_nolinear, sliplist_emg    # emg_nolinear為已刪除rt波的圖EMG, 沒有補點之值（計算rms用）; sliplist_emg為把有emg部分切成list


def deleteZero(data): #把0值刪除
    df = pd.DataFrame({'data':data})
    df_withoutzero = df[df['data'] != 0]
    data_withoutzero = df_withoutzero['data']

    return data_withoutzero

#計算期望值和方差
def calc(data):
    n=len(data) 
    niu=0.0 # niu表示平均值,即期望.
    niu2=0.0 # niu2表示平方的平均值
    niu3=0.0 # niu3表示三次方的平均值
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu /= n  
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu*niu)
    return [niu,sigma,niu3]

#計算鋒度、偏度
def calc_stat(data):
    [niu, sigma, niu3]=calc(data)
    n=len(data)
    niu4=0.0 # niu4計算峰度計算公式的分子
    for a in data:
        a -= niu
        niu4 += a**4
    niu4 /= n

    skew =(niu3 -3*niu*sigma**2-niu**3)/(sigma**3) # 偏度計算公式
    kurt=niu4/(sigma**4) # 峰度計算公式:下方為方差的平方即為標準差的四次方
    return [niu, sigma,skew,kurt] #skew偏度 kurt峰度




