#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:56:54 2022

@author: weien
"""

"""
包含Pantompkin Algorithm 跟 Shannon Algorithm兩種Rpeak抓取的演算法
1. Pantompkin Algorithm: J Pan, WJ Tompkins - IEEE transactions on biomedical engineering, 1985
2. Shannon Algorithm: MS Manikandan, KP Soman - Biomedical Signal Processing and Control, 2012

最終分析使用Shannon Algorithm
"""

"""以下Fuction分類為:
1. for Two Algorithm: 兩種演算法都會使用的Function
2. for Shannon Algorithm: 只有Shannon Algo會使用的Function
3. for Pantompkin Algorithm: 只有Pantompkin Algo會使用的Function
4. for Decision Rules: 校正Rpeak的決策規則
5. for EMG Signal: 處理EMG訊號的Function
6. for Statistics: 統計參數相關的Function
7. Main Function: 最主要的Function，會呼叫前面函式，來取得Rpeak的函式
"""


import numpy as np
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import pandas as pd
import math
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


'''------- 1. for Two Algorithm--------'''
def get_linear(start, end):  # EX: a=linearFunc([3,2],[7,4])

    """
    # 線性函式，且x刻度為1
    input: [list], [list], start:[x, y], end:[x, y]
    example: get_linear([3,2],[7,4])

    output: [list]
    """

    output = []
    
    a = (end[1]-start[1])/(end[0]-start[0])
    b = start[1]-a*start[0]
    
    for i in range(start[0]+1, end[0]):
        y = a*i+b
        tempoutput = [i,y]
        output.append(tempoutput)
    
    return output

def defivative(data):
    x = range(len(data))
    dy = np.zeros(data.shape, float)
    dy[0:-1] = np.diff(data)/np.diff(x)  # 每一格的斜率(diff)是前後相減, [0:-1]是最後一個不取(i.e.從0取到倒數第二個)
    dy[-1] = (data[-1] - data[-2])/(x[-1] - x[-2])
    
    return dy

def medfilt(data, k):  # 基線飄移 x是訊號 k是摺照大小
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert data.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(data), k), dtype=data.dtype)
    y[:, k2] = data
    for i in range(k2):
        j = k2 - i
        y[j:, i] = data[:-j]
        y[:j, i] = data[0]
        y[:-j, -(i+1)] = data[j:]
        y[-j:, -(i+1)] = data[-1]

    return np.median(y, axis=1)  # 得到output之後還要再用原始訊號減此值

def lowpass_filter(cutfq, fs, data):
    b, a = signal.butter(8, 2*cutfq/fs, 'lowpass')   # 濾除cutfq(HZ)以上的頻率
    data_lowfilter = signal.filtfilt(b, a, data)

    return data_lowfilter

def highpass_filter(cutfq, fs, data):
    b, a = signal.butter(8, 2*cutfq/fs, 'highpass')  # 濾除fq(HZ)以下的頻率
    data_highfilter = signal.filtfilt(b, a, data)

    return data_highfilter


'''--------2. for Shannon Algorithm-----------'''
# 畫上下包絡線
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin, lmax

# 線性插值
def interpolate(signal, n):
    """
    input:
    signal: 訊號
    n: 插入產生為多少長度之訊號
    """
    x = np.linspace(0, len(signal)-1, num=len(signal), endpoint=True)
    f = interp1d(x, signal, kind='cubic')
    xnew = np.linspace(0, len(signal)-1, num=n, endpoint=True)
    
    return f(xnew)

def movingaverage(signal, s, fs): #sliding window #s是window大小為幾秒
    """
    input:
    signal: 訊號
    s: window大小為幾秒
    fs: 頻率
    """
    #sliding window
    window = int(s*fs)
    moving_average = []
    for i in range(len(signal)):
        moving_average.append(np.mean(signal[i:i+window]))
    moving_average = np.array(moving_average)
    
    return moving_average

# 找交零index
def find_zerocross_index(signal):
    cross_zero_index = []
    for i in range(len(signal)-1):
        if (signal[i]*signal[i+1]) < 0 and signal[i] < signal[i+1]:
            if abs(signal[i]) <= abs(signal[i+1]):    #把接近0的加進陣列
                cross_zero_index.append(i)
            elif abs(signal[i]) > abs(signal[i+1]):
                cross_zero_index.append(i+1)

    return cross_zero_index

# 陣列位移 (同時減值，且若第一筆小於0則為0)
def shift_array(signal, shift_value):
    """
    signal:
    shift_value: 向左位移的格數
    """
    data_output = np.array(signal)-shift_value
    for i in range(len(data_output)):
        if data_output[i] < 0:  # 若位移後小於0
            data_output[i] = signal[0]  # 則等於第一個點
    return data_output


'''----------3. Def function for Pantompkin Algorithm---------'''
def reversesignal(signal):  # 反轉訊號
    dataraw = signal.values
    median = np.median(dataraw)
    reverse_ecg = median-(dataraw-median)    
    return reverse_ecg

def findpeak(signal):
    # findpeak 依照論文應該要150ms 因為取樣頻率不同（論文為200HZ, 我們為250HZ） 為了取整數所以取120ms 250*0.12 = 30 觀察結果圖 測試取150
    peaks_x, _ = find_peaks(signal, distance=120)
    peaks_y = signal[peaks_x]  # findpeak的y值
    
    return peaks_x, peaks_y

def detect_rpeak_baseon_pantompkin(ecg_raw, peaks_x, peaks_y):  # 決策Threshold找Rpeak
    """
    參考資料 https://github.com/c-labpl/qrs_detector/blob/master/QRSDetectorOffline.py
    """
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
    rpeak_x = qrs_peaks_indices  # Wayne added index of rpeak
    rpeak_y = ecg_raw[rpeak_x]  # Wayne added value of rpeak
    
    return rpeak_x, rpeak_y


'''------------4. for Decision Rules------------'''
# relocated rpeak
def relocated_rpeak(signal, rpeak_x, range_n, method):
    """
    校正R peak: 在range_n範圍內找最小/小值
    input:
    method: [str] min, max
    """
    newrpeak = pd.Series()
    re_location = 0
    signal = pd.Series(signal)
    for i in range(len(rpeak_x)):
        if rpeak_x[i]-int(range_n/2) < 0:  # 若第一點小於扣除的長度
            range_list = signal[0:rpeak_x[i]+int(range_n/2)]  # 以0為起點
            
        elif rpeak_x[i]+int(range_n/2) > len(signal): # 若最後一點超過總長度
            range_list = signal[rpeak_x[i]-int(range_n/2):len(signal)]
        
        else:  
            range_list = signal[rpeak_x[i]-int(range_n/2):rpeak_x[i]+int(range_n/2)]


        if method == 'min':
            re_location = range_list.nsmallest(1)  # Series取最小值用nsmallest
        elif method == 'max':
            re_location = range_list.nlargest(1)  # Series取最大值用nlargest
        else:
            print('function of relocated_rpeak had invalid input')
        newrpeak = newrpeak._append(re_location)

    newdetedted_rpeak_x = newrpeak.index.values.tolist() 
    newdetedted_rpeak_y = newrpeak.tolist()
    
    return newdetedted_rpeak_x, newdetedted_rpeak_y

# relocated rpeak
def delete_close_rpeak(rpeak_index, distance_range):
    """
    刪除過接近的Rpeak點 閾值為distance_range
    """
    diff_data = np.diff(rpeak_index)
    for i in range(len(diff_data)-1, 0, -1):
        if diff_data[i] < distance_range:  # 若距離小於此參數
            del rpeak_index[i+1]  # 則刪除後一個點
    return rpeak_index

# relocated rpeak
def delete_lower_rpeak(rpeak_index, signal, lower_range):
    """
    刪除y值太小的Rpeak點 閾值為distance_range
    """
    for i in range(len(rpeak_index)-1, -1, -1):
        if signal[rpeak_index[i]] < lower_range:  # 若距離小於此參數
            del rpeak_index[i]  # 則刪除後一個點
    return rpeak_index

def delete_extreme_value_baseonhist(rrinterval_list):
    """
    刪除直方圖最高值前後超過0.5之RRI
    """
    hist = plt.hist(rrinterval_list, bins=10, color='black')
    plt.xlim(100, 1000)
    plt.ylim(0, 40)
    plt.title('Touch')
    hist_dict = {'Condition': 'scared', 'count': hist[0], 'value': hist[1]}

    max_index = hist_dict['count'].tolist().index(max(hist_dict['count']))
    max_value = hist_dict['value'][max_index]
    range_value = [max_value-0.5*max_value,max_value+0.5*max_value]
    
    delvalue = []
    
    for i in range(len(rrinterval_list)):
        if rrinterval_list[i] < range_value[0] or rrinterval_list[i] > range_value[1]:
            delvalue.append(rrinterval_list[i])

    rri_keep_value = set(rrinterval_list)-set(delvalue)
    
    return rri_keep_value


'''-------------5. for EMG Signal-----------'''
def fill_rtpeak_bylinear(signal, rpeakindex, qrs_range, tpeak_range):
    """
    取得EMG: 刪除RT波，並線性補點
    output: 刪除rt波的EMG, 並線性補點被刪除之值
    """
    start_x, start_y, end_x, end_y = None, None, None, None
    signalwithlinear = signal
    pre_range = math.floor(qrs_range/2)
    after_range = round(qrs_range/2)

    # 將rpeak的點改成0
    for i in range(len(rpeakindex)):
        rpeak_index = rpeakindex[i]
        if rpeak_index < pre_range:
            start_x = 0
            start_y = signalwithlinear[0]
        
        elif rpeak_index >= pre_range:
            start_x = rpeak_index-pre_range
            start_y = signalwithlinear[rpeak_index-pre_range]
            
        end_x = rpeak_index+after_range+tpeak_range
        
        if len(signalwithlinear) < end_x:
            end_x = len(signalwithlinear)
            end_y = signalwithlinear[len(signalwithlinear)-1]
        elif len(signalwithlinear) >= end_x:
            end_x = end_x
            end_y = signalwithlinear[rpeak_index+after_range+tpeak_range]
        
        linear_output = get_linear([start_x, start_y],[end_x, end_y])  # linearFunc.py引入 #共前後1秒
        firstindex = linear_output[0][0]
        
        for j in range(0, len(linear_output)):
            signalwithlinear[(j+firstindex)] = linear_output[j][1]
        
    return signalwithlinear

def delete_rtpeak(signal, rpeakindex, qrs_range, tpeak_range):  # qrs_range, tpeak_range須為int
    """
    取得EMG，刪除RT波，且不補點
    input: qrs_range [int], tpeak_range [int]
    """
    emg_nolinear = signal
    sliplist_emg = []
    start_x, start_y, end_x, end_y, emg_start_x = None, None, None, None, None

    pre_range = math.floor(qrs_range/2)
    after_range = round(qrs_range/2)
    
    # 將rpeak的點改成0
    for i in range(len(rpeakindex)):
        
        rpeak_index = rpeakindex[i]
        if rpeak_index < pre_range:  # 若要刪除的位置小於Rpeak的位置，則以第一點為起始
            start_x = 0
            start_y = emg_nolinear[0]
        
        elif rpeak_index >= pre_range: # 若pre_range沒有小於rpeak位置，則直接刪減
            start_x = rpeak_index-pre_range
            start_y = emg_nolinear[rpeak_index-pre_range]
            
        end_x = rpeak_index+after_range+tpeak_range
        
        if len(emg_nolinear) <= end_x:
            end_x = len(emg_nolinear)
            end_y = emg_nolinear[len(emg_nolinear)-1]
        elif len(emg_nolinear) > end_x:
            end_y = emg_nolinear[end_x]
            
        
        linear_output = get_linear([start_x, start_y], [end_x, end_y])
        firstindex = linear_output[0][0]
        
        for j in range(0, len(linear_output)):
            emg_nolinear[(j+firstindex)] = 0
            
        sliplist_emg.append(signal[emg_start_x: start_x])

    # emg_nolinear為已刪除rt波的圖EMG, 沒有補點之值（計算rms用）; sliplist_emg為把有emg部分切成list 
    # 且輸入為將rawdata轉為pd.Series(rawdata)
    return emg_nolinear, sliplist_emg    


'''--------6. Def function for Statistics-------'''
def calc(data):
    """
    計算期望值和方差
    """
    n = len(data)
    niu = 0.0  # niu表示平均值,即期望.
    niu2 = 0.0  # niu2表示平方的平均值
    niu3 = 0.0  # niu3表示三次方的平均值
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu /= n  
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu*niu)
    return [niu, sigma, niu3]

def calc_stat(data):
    """
    計算鋒度、偏度
    """
    [niu, sigma, niu3] = calc(data)
    n = len(data)
    niu4 = 0.0  # niu4計算峰度計算公式的分子
    for a in data:
        a -= niu
        niu4 += a**4
    niu4 /= n

    skew = (niu3 - 3*niu*sigma**2-niu**3)/(sigma**3)  # 偏度計算公式
    kurt = niu4/(sigma**4)  # 峰度計算公式:下方為方差的平方即為標準差的四次方
    return [niu, sigma, skew, kurt]  # skew偏度, kurt峰度


'''-------7.  Main Function--Get R peak------------'''
def getRpeak_pantompskin(ecg, fs):
    """
    使用pantomskin取Peak
    """
    # ------Set variables-------
    medianfilter_size = 61
    lowpass_fq = 10
    highpass_fq = 20
    # ------End of setting variables-------

    median_adjustline = medfilt(np.array(ecg), medianfilter_size)  # sliding window折照為一半 120ms->61
    ecg_median = ecg-median_adjustline  # 基線飄移
    rawdata_mv = ecg_median
    ecg_lowpass = lowpass_filter(lowpass_fq, fs, ecg_median)  # 低通
    ecg_bandpass = highpass_filter(highpass_fq, fs, ecg_lowpass)   # 高通
    ecg_defivative = defivative(ecg_bandpass)  # 導數
    ecg_square = np.square(ecg_defivative)  # 平方
    peaks_x, peaks_y = findpeak(ecg_square)
    detedted_rpeak_x, detedted_rpeak_y = detect_rpeak_baseon_pantompkin(rawdata_mv, peaks_x, peaks_y)   #  Pantompkin決策演算取rpeak
    final_rpeak_x, _ = relocated_rpeak(ecg, detedted_rpeak_x, (0.35*fs)*2, 'max')
    
    return ecg_median, final_rpeak_x

def getRpeak_shannon(ecg, fs):
    """
    使用Shonnon取Rpeak
    input:
    ecg [list],
    fs=250
    """
    # ------Set variables-------
    medianfilter_size = 61
    gaussian_filter_sigma = 0.03 * fs
    moving_average_ms = 2.5
    final_shift = 0
    detectR_maxvalue_range = (0.32 * fs) * 2
    rpeak_close_range = 0.15 * fs
    detect_Rpeak_index = []
    relocated_rpeak_method = 'all_choose_max'  # all_choose_max, oneepoch_choose_highorlow, onebeat_choose_highorlow
    '''
    校正rpeak位置的方法:
    M1: all_choose_max, All choose high amplitude
    M2: oneepoch_choose_highorlow, 在一個epoch中計算標準，判斷要找最大/小值
    M3: onebeat_choose_highorlow, 每一個beat都判斷要找最大/小值
    '''

    # ------End of setting variables-------

    median_filter_data = medfilt(np.array(ecg), medianfilter_size)
    median_ecg = ecg-median_filter_data
    lowpass_data = lowpass_filter(20, fs, median_ecg)  # 低通
    bandfilter_data = highpass_filter(10, fs, lowpass_data)    # 高通
    dy_data = defivative(bandfilter_data)  # 一程微分
    normalize_data = dy_data/np.max(dy_data)  # 正規化
    see_data = (-1)*(normalize_data**2)*np.log((normalize_data**2))  # Shannon envelop
    # lmin_index, lmax_index = hl_envelopes_idx(see_data) #取上包絡線
    # lmax_data = see_data[lmax_index]
    # interpolate_data = interpolate(lmax_data,len(ecg))
    gaussian_data = gaussian_filter(see_data, sigma=gaussian_filter_sigma)
    hibert_data = np.imag(hilbert(gaussian_data))  # Hilbert取複數
    movingaverage_data = movingaverage(hibert_data, moving_average_ms, fs)  # moving average
    hibertmoving_data = hibert_data-movingaverage_data
    zero_index = find_zerocross_index(hibertmoving_data)  # Positive zero crossing point
    zero_shift_index = shift_array(zero_index, final_shift)  # 位移結果
    # Decision Rule
    detect_maxRpeak_index, _ = relocated_rpeak(median_ecg, zero_shift_index, detectR_maxvalue_range, 'max')
    detect_minRpeak_index, _ = relocated_rpeak(median_ecg, zero_shift_index, detectR_maxvalue_range, 'min')
    # Relocated r peak
    if relocated_rpeak_method == 'all_choose_max':
        detect_Rpeak_index = detect_maxRpeak_index
    elif relocated_rpeak_method == 'oneepoch_choose_highorlow':
        for i in range(len(detect_maxRpeak_index)):
            if (np.abs(detect_maxRpeak_index[i]) >= np.abs(detect_minRpeak_index)).all():
                detect_Rpeak_index.append(detect_maxRpeak_index[i])
            elif (np.abs(detect_maxRpeak_index[i]) < np.abs(detect_minRpeak_index)).all():
                detect_Rpeak_index.append(detect_minRpeak_index[i])
    elif relocated_rpeak_method == 'onebeat_choose_highorlow':
        maxRpeak_sum = np.sum(np.abs(ecg[detect_maxRpeak_index]))
        minRpeak_sum = np.sum(np.abs(ecg[detect_minRpeak_index]))

        if maxRpeak_sum >= minRpeak_sum:
            detect_Rpeak_index = detect_maxRpeak_index
        elif minRpeak_sum > maxRpeak_sum:
            detect_Rpeak_index = detect_minRpeak_index
    rpeakindex = delete_close_rpeak(detect_Rpeak_index, rpeak_close_range)  # 刪除rpeak間隔小於rpeak_close_range之值

    return median_ecg, rpeakindex


'''-------8. Def function for RRI---------'''
def interpolate_rri(rawrrinterval, fs):
    """
    將因雜訊刪除的RRI進行補點
    """
    rrinterval_add = rawrrinterval
    # 計算時需將先前因雜訊而刪除的地方做補點
    i = 0
    while i < len(rrinterval_add):
        if rrinterval_add[i] >= 2*fs/(fs/1000):    # 因為要把index值換算成ms 2是指刪除的兩秒
            try:
                insert_distance = (rrinterval_add[i-1] + rrinterval_add[i+1])/2
                n = int(rrinterval_add[i]/insert_distance)
                add_list = [insert_distance] * n
                rrinterval_add = np.append(np.append(rrinterval_add[:i], add_list), rrinterval_add[i+1:])
                i += n
            except IndexError:
                pass
        i += 1
        
    return rrinterval_add