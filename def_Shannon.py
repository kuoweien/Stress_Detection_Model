#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:56:54 2022

@author: weien
"""

import numpy as np
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import pandas as pd
import math
import def_linearFunc

#%%ECG
#微分
def defivative(data_y): #微分
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

def interpolate(raw_signal,n):  #線性插值 signal為原始訊號 n為要插入產生為多少長度之訊號

    x = np.linspace(0, len(raw_signal)-1, num=len(raw_signal), endpoint=True)
    f = interp1d(x, raw_signal, kind='cubic')
    xnew = np.linspace(0, len(raw_signal)-1, num=n, endpoint=True)  
    
    return f(xnew)


def movingaverage(ecg_square, s): #sliding window #s是window大小為幾秒
    #sliding window
    win = int(s*250) #250HZ
    moving_average = []
    for i in range(len(ecg_square)):
        moving_average.append(np.mean(ecg_square[i:i+win])) #視窗大小為120ms(30個點)
    moving_average=np.array(moving_average) 
    
    return moving_average


def gaussian_filter1d(raw_signal, sigma):
    # filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter_output = [1 / (raw_signal * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in raw_signal]  
    
    return gaussian_filter_output


def findZeroCross(raw_signal):
    cross_zero_index = []
    for i in range(len(raw_signal)-1):
        if (raw_signal[i]*raw_signal[i+1])<0 and raw_signal[i]<raw_signal[i+1]:
            if abs(raw_signal[i])<=abs(raw_signal[i+1]):    #把接近0的加進陣列
                cross_zero_index.append(i)
            elif abs(raw_signal[i])>abs(raw_signal[i+1]):
                cross_zero_index.append(i+1)          
            
    return cross_zero_index

def decisionRule(raw_signal):
    detect_Rpeak = []
    for i in range(1,len(raw_signal),2):
        detect_Rpeak.append(raw_signal[i])
    return detect_Rpeak

def shiftArray(data, shift_value):  #陣列同時減值，且若第一筆小於
    data_output = np.array(data)-shift_value
    for i in range(len(data_output)):
        if data_output[i] <0:  # 若位移後小於0
            data_output[i] = data[0] # 則等於第一個點
    return data_output

    
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


def deleteCloseRpeak(data_index, distance_range): #刪除太接近的Rpeak點 閾值為distance_range
    diff_data = np.diff(data_index)
    for i in range(len(diff_data)-1,0,-1):
        if diff_data[i] < distance_range:  # 若距離小於此參數
            del data_index[i+1]  # 則刪除後一個點
    return data_index

#%% EMG
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


#畫圖
'''
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)
fig.tight_layout()
'''