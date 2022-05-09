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


def pantomskin(rawdata):
    rawdata=rawdata.reset_index(drop=True)
    ecg_lowpass=bandfilter.lowPassFilter(15,rawdata)        #低通
    ecg_bandpass = bandfilter.highPassFilter(5,ecg_lowpass)        #高通
    ecg_defivative = defivative(ecg_bandpass)       #導數
    ecg_square = np.square(ecg_defivative)       #平方
    movingwindow= movingaverage(ecg_defivative)     #moving average
    peaks_x, peaks_y = findpeak(ecg_defivative)
    detedted_rpeak_x,detedted_rpeak_y = detectRpeak(ecg_dataraw, peaks_x, peaks_y)  
    return detedted_rpeak_x,detedted_rpeak_y
    
    
    
    
    




