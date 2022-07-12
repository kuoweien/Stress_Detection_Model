#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:56:54 2022

@author: weien
"""

'''包含Pantompkin Algorithm 跟 Shannon Algorithm兩種Rpeak抓取的演算法
**Pantompkin Algorithm: J Pan, WJ Tompkins - IEEE transactions on biomedical engineering, 1985
**Shannon Algorithm: MS Manikandan, KP Soman - Biomedical Signal Processing and Control, 2012
'''

'''以下Fuction分類為:
1. Def for Two Algorithm: 兩種演算法都會使用的Function
2. Def for Shannon Algorithm: 只有Shannon Algo會使用的Function
3. Def function for Pantompkin Algorithm: 只有Pantompkin Algo會使用的Function
4. Def for Decision Rules: 校正Rpeak的決策規則
5. Def function for EMG Signal: 處理EMG訊號的Function
6. Def function for Statistics: 統計參數相關的Function
7. Main Function: 最主要的Function，會呼叫前面函式，來取得Rpeak的函式
'''


import numpy as np
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import pandas as pd
import math
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


#%%
'''------- 1. Def for Two Algorithm--------'''

#線性函式 且x刻度為1
def linearFunc(start,end): # EX: a=linearFunc([3,2],[7,4])
    output=[]
    
    a=(end[1]-start[1])/(end[0]-start[0])
    b=start[1]-a*start[0]
    
    for i in range(start[0]+1,end[0]):
        y=a*i+b
        tempoutput=[i,y]
        output.append(tempoutput)
    
    return output


#微分
def defivative(data_y):
    x=range(len(data_y))
    
    dy = np.zeros(data_y.shape,np.float)
    dy[0:-1] = np.diff(data_y)/np.diff(x) #每一格的斜率 diff是前後相減  [0:-1]是最後一個不取(i.e.從0取到倒數第二個)
    dy[-1] = (data_y[-1] - data_y[-2])/(x[-1] - x[-2])
    
    return dy

def medfilt (x, k): #基線飄移 x是訊號 k是摺照大小
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
    return np.median (y, axis=1)  #做完之後還要再用原始訊號減此值

#低通
def lowPassFilter(fq,fs,data):
    b, a = signal.butter(8, 2*fq/fs, 'lowpass')   #濾除 fq HZ以上的頻率
    data_lowfilter = signal.filtfilt(b, a, data) 
    return data_lowfilter
  
#高通  
def highPassFilter(fq,fs,data):
    b, a = signal.butter(8, 2*fq/fs, 'highpass') #濾除 fq HZ以下的頻率
    data_highfilter = signal.filtfilt(b, a, data)
    return data_highfilter

#%%
'''--------2. Def for Shannon Algorithm-----------'''


#畫上下包絡線
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
    
    return lmin,lmax

#線性插值
def interpolate(raw_signal,n):   #signal為原始訊號 n為要插入產生為多少長度之訊號

    x = np.linspace(0, len(raw_signal)-1, num=len(raw_signal), endpoint=True)
    f = interp1d(x, raw_signal, kind='cubic')
    xnew = np.linspace(0, len(raw_signal)-1, num=n, endpoint=True)  
    
    return f(xnew)


def movingaverage(ecg_square, s, fs): #sliding window #s是window大小為幾秒
    #sliding window
    win = int(s*fs) #250HZ
    moving_average = []
    for i in range(len(ecg_square)):
        moving_average.append(np.mean(ecg_square[i:i+win])) #視窗大小為120ms(30個點)
    moving_average=np.array(moving_average) 
    
    return moving_average


#找交零
def findZeroCross(raw_signal):
    cross_zero_index = []
    for i in range(len(raw_signal)-1):
        if (raw_signal[i]*raw_signal[i+1])<0 and raw_signal[i]<raw_signal[i+1]:
            if abs(raw_signal[i])<=abs(raw_signal[i+1]):    #把接近0的加進陣列
                cross_zero_index.append(i)
            elif abs(raw_signal[i])>abs(raw_signal[i+1]):
                cross_zero_index.append(i+1)          
            
    return cross_zero_index

#陣列位移，(同時減值，且若第一筆小於0則為0)
def shiftArray(data, shift_value):  #data為陣列，shift_value為向左位移的格數
    data_output = np.array(data)-shift_value
    for i in range(len(data_output)):
        if data_output[i] <0:  # 若位移後小於0
            data_output[i] = data[0] # 則等於第一個點
    return data_output


#%%
'''----------3. Def function for Pantompkin Algorithm---------'''


def reversesignal(dataraw): #反轉訊號
    dataraw = dataraw.values
    median = np.median(dataraw)
    reverse_ecg = median-(dataraw-median)    
    return reverse_ecg


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

#%%
'''------------4. Def for Decision Rules------------'''

#校正Rpeak位置，找最小值   
def ecgfindtheminvalue(rawdata, rpeak_x, range_n): #因前面會抓錯 所以直接找在附近range_n的最小值
    newrpeak = pd.Series()
    rawdata = pd.Series(rawdata)
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

#校正Rpeak位置，找最大值
def ecgfindthemaxvalue(rawdata, rpeak_x, range_n): #Decision rule找最大值(因前面會抓錯)在rpeak附近range_n點找最大值 input原始data, detected rpeak, 找尋最大值範圍
    newrpeak = pd.Series()
    rawdata = pd.Series(rawdata)
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

#校正Rpeak，刪除過於接近的Rpeak
def deleteCloseRpeak(data_index, distance_range): #刪除太接近的Rpeak點 閾值為distance_range
    diff_data = np.diff(data_index)
    for i in range(len(diff_data)-1,0,-1):
        if diff_data[i] < distance_range:  # 若距離小於此參數
            del data_index[i+1]  # 則刪除後一個點
    return data_index

#校正Rpeak，刪除振幅過小的Rpeak
def deleteLowerRpeak(data_index, ecg, lower_range): #刪除y值太小的Rpeak點 閾值為distance_range
    for i in range(len(data_index)-1,-1,-1):
        if ecg[data_index[i]] < lower_range:  # 若距離小於此參數
            del data_index[i]  # 則刪除後一個點
    return data_index

# 刪除直方圖最高值前後超過0.5之RRI
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


#%% EMG
'''-------------5. Def function for EMG Signal-----------'''

#取得EMG: 刪除RT波，並線性補點
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
        
        linearOutput=linearFunc([startX,startY],[endX,endY]) #linearFunc.py引入 #共前後1秒
        firstindex=linearOutput[0][0]
        
        for j in range(0,len(linearOutput)):
            emgwithlinear[(j+firstindex)] = linearOutput[j][1]
        
    return emgwithlinear     # Output已刪除rt波的圖EMG, 有線性補點之值

#取得EMG: 刪除RT波，並不補點
def deleteRTpeak(rawdata, rpeakindex, qrs_range, tpeak_range): #與def fillRTpeakwithLinear相同 可以放在一起寫（要再修） #qrs_range, tpeak_range須為int
    emg_nolinear = rawdata
    
    sliplist_emg = []
    
    pre_range = math.floor(qrs_range/2)
    after_range = round(qrs_range/2)
    
    #將rpeak的點改成0->濾掉r peak的部分
    emg_startX = 0
    for i in range(len(rpeakindex)):
        
        rpeak_index=rpeakindex[i]
        if rpeak_index<pre_range:  #若要刪除的位置小於Rpeak的位置，則以第一點為起始
            startX=0
            startY=emg_nolinear[0]
        
        elif rpeak_index>=pre_range: #若pre_range沒有小於rpeak位置，則直接刪減
            startX=rpeak_index-pre_range
            startY=emg_nolinear[rpeak_index-pre_range]
            
        endX=rpeak_index+after_range+tpeak_range
        
        if len(emg_nolinear)<=endX:
            endX=len(emg_nolinear)
            endY=emg_nolinear[len(emg_nolinear)-1]
        # if endX>len(rawdata):
        #     endX = len(rawdata)
        elif len(emg_nolinear)>endX:
            # endX=endX
            endY=emg_nolinear[endX]
            
        
        linearOutput=linearFunc([startX,startY],[endX,endY]) #linearFunc.py引入 #共前後1秒
        firstindex=linearOutput[0][0]
        
        for j in range(0,len(linearOutput)):
            emg_nolinear[(j+firstindex)] = 0
            
        sliplist_emg.append(rawdata[emg_startX:startX])
        emg_startX = endX

            
    return emg_nolinear, sliplist_emg    # emg_nolinear為已刪除rt波的圖EMG, 沒有補點之值（計算rms用）; sliplist_emg為把有emg部分切成list

#%%
'''--------6. Def function for Statistics-------'''
#刪除0值
def deleteZero(data): 
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


#%% 

'''-------7.  Main Function--Get R peak------------'''
#使用pantomskin取Peak
def getRpeak_pantompskin(ecg ,fs, medianfilter_size, lowpass_fq, highpass_fq):
    median_adjustline = medfilt(np.array(ecg), medianfilter_size) #sliding window折照為一半 120ms->61
    ecg_median = ecg-median_adjustline  #基線飄移
    rawdata_mV = ecg_median
    ecg_lowpass = lowPassFilter(lowpass_fq,fs,ecg_median)        #低通
    ecg_bandpass = highPassFilter(highpass_fq,fs,ecg_lowpass)        #高通
    ecg_defivative = defivative(ecg_bandpass)       #導數
    ecg_square = np.square(ecg_defivative)       #平方
    peaks_x, peaks_y = findpeak(ecg_square)
    detedted_rpeak_x,detedted_rpeak_y = detectRpeak(rawdata_mV, peaks_x, peaks_y)       #Pantompkin決策演算抓rpeak 資料來源：網路找的Github
    newdetedted_rpeak_x, _ = ecgfindthemaxvalue(ecg, detedted_rpeak_x, (0.35*fs)*2)
    
    return ecg_median, newdetedted_rpeak_x

#Shonnon取Rpeak
def getRpeak_shannon(ecg, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range):
    median_filter_data = medfilt(np.array(ecg), medianfilter_size)
    # median_filter_data = medfilt(ecg, medianfilter_size)
    median_ecg = ecg-median_filter_data
    lowpass_data = lowPassFilter(20,fs,median_ecg)  #低通
    bandfilter_data = highPassFilter(10,fs,lowpass_data)    #高通
    dy_data = defivative(bandfilter_data) #一程微分
    normalize_data = dy_data/np.max(dy_data) #正規化
    see_data = (-1)*(normalize_data**2)*np.log((normalize_data**2)) #Shannon envelop
    # lmin_index, lmax_index = hl_envelopes_idx(see_data) #取上包絡線
    # lmax_data = see_data[lmax_index]
    # interpolate_data = interpolate(lmax_data,len(ecg))
    gaussian_data = gaussian_filter(see_data, sigma=gaussian_filter_sigma)
    hibert_data = np.imag(hilbert(gaussian_data))  #Hilbert取複數
    movingaverage_data = movingaverage(hibert_data, moving_average_ms, fs) #moving average
    hibertmoving_data = hibert_data-movingaverage_data
    zero_index = findZeroCross(hibertmoving_data)  #Positive zero crossing point
    zero_shift_index = shiftArray(zero_index, final_shift) #位移結果
    
    #Decision Rule: input分為三種 1.以RawECG找最大值 2.bandfilterECG找最大值 3.RawECG找最小值
    detect_Rpeak_index, _   = ecgfindthemaxvalue(median_ecg, zero_shift_index, detectR_maxvalue_range)  # RawECG抓R peak 找範圍內的最大值 
    re_detect_Rpeak_index = deleteCloseRpeak(detect_Rpeak_index, rpeak_close_range) #刪除rpeak間隔小於rpeak_close_range之值
    # re_detect_Rpeak_index = deleteLowerRpeak(re_detect_Rpeak_index, ecg, 0.001)

    return median_ecg, re_detect_Rpeak_index

#%%
'''-------8. Def function for RRI------------'''
# 將因雜訊刪除的RRI進行補點
def interpolate_rri(rawrrinterval, fs):
    rrinterval_add = rawrrinterval
    # 計算時需將先前因雜訊而刪除的地方做補點
    i = 0
    while i < len(rrinterval_add):
    # for i in range(len(rrinterval)):
        if rrinterval_add[i] >= 2*fs/(fs/1000) :    # 因為要把index值換算成ms
            insert_distance = (rrinterval_add[i-1] + rrinterval_add[i+1])/2
            n = int(rrinterval_add[i]/insert_distance)
            add_list = [insert_distance] * n
            rrinterval_add = np.append(np.append(rrinterval_add[:i], add_list), rrinterval_add[i+1:])
            i+=n
        i+=1
        
    return rrinterval_add
