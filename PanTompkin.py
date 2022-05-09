#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:42:33 2022

@author: weien
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import def_passFilter as bandfilter
import def_linearFunc
from scipy.interpolate import interp1d
import math

#Note: 323行那邊Error，沒有刪到0，會是NoneType

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

def ecgfindthemaxvalue(rawdata, rpeak_x, range_n): #Decision rule找最大值(因前面會抓錯)在rpeak附近100點找最大值 input原始data, detected rpeak, 找尋最大值範圍
    newrpeak = pd.Series()
    for i in range(len(rpeak_x)):
        range_list = rawdata[rpeak_x[i]-int(range_n/2):rpeak_x[i]+int(range_n/2)]
        min_location = range_list.nlargest(1) #Series取最小值 取最大值為nlargest 最小值為nsmallest
        newrpeak = newrpeak.append(min_location)

    newdetedted_rpeak_x = newrpeak.index.values.tolist() 
    newdetedted_rpeak_y = newrpeak.tolist()
    
    return newdetedted_rpeak_x, newdetedted_rpeak_y

def ecgfindtheminvalue(rawdata, rpeak_x, range_n): #Decision rule找最小值(因前面會抓錯)在rpeak附近100點找最小值 input原始data, detected rpeak, 找尋最大值範圍
    newrpeak = pd.Series()
    for i in range(len(detedted_rpeak_x)):
        range_list = ecg_dataraw[detedted_rpeak_x[i]-int(range_n/2):detedted_rpeak_x[i]+int(range_n/2)]
        min_location = range_list.nsmallest(1) #Series取最小值 取最大值為nlargest 最小值為nsmallest
        newrpeak = newrpeak.append(min_location)

    newdetedted_rpeak_x = newrpeak.index.values.tolist() 
    newdetedted_rpeak_y = newrpeak.tolist()
    
    return newdetedted_rpeak_x, newdetedted_rpeak_y

#包含T波一起刪除

def fillRTpeakwithLinear(rawdata, rpeakindex): #原始資料跟rpeak_x #刪除rtpeak
    emgwithlinear = rawdata
    qrs_range = 25  #dog = 25 小魚30
    tpeak_range = 37  #dog = 37 小魚0
    
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
        
    return emgwithlinear     # Output已刪除rt波的圖EMG, 沒有補點之值（計算rms用）

def deleteRTpeak(rawdata, rpeakindex): #與def fillRTpeakwithLinear相同 可以放在一起寫（要再修）
    emg_mV = np.array(rawdata)/65535*1.8-0.9
    emg_nolinear = emg_mV
    
    qrs_range = 25  #dog = 25 小魚30
    tpeak_range = 37  #dog = 37 小魚0
    
    pre_range = math.floor(qrs_range/2)
    after_range = round(qrs_range/2)
    
    #將rpeak的點改成0->濾掉r peak的部分
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
            
    return emg_nolinear     # Output已刪除rt波的圖EMG, 沒有補點之值（計算rms用）


# myArray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# indexes = [3, 5, 7]
# modifiedArray = np.delete(myArray, indexes)

def deleteZero(x):
    for i in range(len(x)-1,-1,-1):
        if x[i] == 0:
            x = np.delete(x, i)


#計算期望值和方差
def calc(data):
    n=len(data) # 10000個數
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

#nemo
situation = 'scared'
clip_range_start = 2500 #petted:10000 scared:2500
clip_range_end = 10000 #petted:17500 scared:10000

df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/'+situation+'.csv').iloc[clip_range_start:clip_range_end] #petted.iloc[10000:17500] scared[2500:10000]
# df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/scared.csv').iloc[2500:10000]
ecg_dataraw = df[situation]
# ecg_dataraw = df['scared']

'''
#小魚stroop
situation = 'Baseline2'
clip_range_start = 2500 #Baseline1: [20125:27625] 
clip_range_end = 17500 

url = '/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/220413小魚Stroop Test/yu_strooptest_base2.csv'
df = pd.read_csv(url)
ecg_dataraw = df['ECG'].iloc[clip_range_start:clip_range_end] 
'''

fs = 250

ecg_dataraw=ecg_dataraw.reset_index(drop=True)
median_adjustline = medfilt(ecg_dataraw.values,61) #sliding window折照為一半 120ms->61
ecg_median = ecg_dataraw-median_adjustline
ecg_lowpass=bandfilter.lowPassFilter(15,ecg_dataraw)        #低通
ecg_bandpass = bandfilter.highPassFilter(5,ecg_lowpass)        #高通
ecg_defivative = defivative(ecg_bandpass)       #導數
ecg_square = np.square(ecg_defivative)       #平方
# movingwindow= movingaverage(ecg_defivative)     #moving average
peaks_x, peaks_y = findpeak(ecg_square)
detedted_rpeak_x,detedted_rpeak_y = detectRpeak(ecg_dataraw, peaks_x, peaks_y)       #抓rpeak

newdetedted_rpeak_x, newdetedted_rpeak_y = ecgfindtheminvalue(ecg_dataraw, detedted_rpeak_x,100) #找最小值

rrinterval = np.diff(newdetedted_rpeak_x)
rrinterval = rrinterval*1000/fs
# x_time= np.array(newdetedted_rpeak_x[0:len(detedted_rpeak_x)-1])/250 #RRinterval畫圖時的x軸
x_time = np.linspace(0, 40, len(rrinterval))
#內差rrinterval為7HZ
f1 = interp1d(x_time, rrinterval) 
rrx_interpolate = np.linspace(0, 40, 280)
rry_interpolate = f1(rrx_interpolate)

plt.figure(figsize = (12,4))
plt.title(situation)
plt.subplot(3,1,1)
plt.ylim(-0.3,0.5)
plt.plot(np.array(range(len(ecg_dataraw)))/250,np.array(ecg_dataraw)/65535*1.8-0.9,'black')
plt.plot(np.array(newdetedted_rpeak_x)/250, np.array(newdetedted_rpeak_y)/65535*1.8-0.9, "o", markersize=4, c='red')
plt.ylabel('ECG (mV)')


emg  = fillRTpeakwithLinear(ecg_dataraw,newdetedted_rpeak_x) #刪除rtpeak並補線性點
emg_mV_withoutLinear = deleteRTpeak(ecg_dataraw,newdetedted_rpeak_x)

emg_mV_withoutLinear_fillnan = deleteZero(emg_mV_withoutLinear)  #Error 沒有刪去0 output會變NoneType

for i in range(len(emg_mV_withoutLinear),-1):
    if emg_mV_withoutLinear[i] == 0:
        del emg_mV_withoutLinear[i]
        

#emg轉成mV 並取rms
emg_mV = np.array(emg)/65535*1.8-0.9
emg_rms = round(np.sqrt(np.mean(emg_mV_withoutLinear**2)),3)

#直方圖：峰值與偏度
[niu, sigma, skew, kurt] = calc_stat(rrinterval)
info = r'$\ skew=%.2f,\ kurt=%.2f$' %(round(skew,2), round(kurt,2))

#%%
#畫圖

plt.subplot(3,1,2)#RRinterval
plt.ylim(400,1200)
plt.plot(rrx_interpolate,rry_interpolate,'black')
plt.ylabel('RR(ms)')

plt.subplot(3,1,3)
plt.ylim(-0.3,0.5)
plt.plot(np.array(range(len(emg)))/250,emg_mV,'black')
plt.ylabel('EMG (mV)')
plt.xlabel('Time (s)')

plt.tight_layout()

#畫直方圖
plt.figure()
plt.title(situation)
plt.hist(rrinterval,density=False, stacked=True,facecolor='grey',alpha=0.9)
plt.grid(True)
plt.xlabel('RR interval')
plt.ylim(0,30)
plt.xlim(400,1200)

print('RMS'+str(emg_rms))
print('RR Mean: '+str(np.mean(rrinterval))+' RR SD: '+str(np.std(rrinterval)))
print(situation+' Skewness: '+str(round(skew,2))+' Kurtosis: '+str(round(kurt,2)))



plt.figure(figsize=(16,12))
plt.subplot(6,1,1)
plt.plot(ecg_dataraw)
plt.plot(detedted_rpeak_x, detedted_rpeak_y, "o", markersize=3, c='red')
plt.title('Raw')

plt.subplot(6,1,2)
plt.plot(ecg_median)
plt.title('MedianFilter')
plt.tight_layout()

plt.subplot(6,1,3)
plt.plot(ecg_bandpass)
plt.title('Band Pass Filter (5-15HZ)')
plt.tight_layout()

plt.subplot(6,1,4)
plt.plot(ecg_defivative)
# plt.plot(peaks_x, peaks_y, "o", markersize=3, c='red')
plt.title('Filtered with Derivative filter')

plt.tight_layout()

plt.subplot(6,1,5)
plt.plot(ecg_square)
# plt.plot(peaks_x, peaks_y, "o", markersize=3, c='red')
plt.title('Squared')
plt.tight_layout()

plt.subplot(6,1,6)
# plt.plot(movingwindow)
# plt.plot(peaks_x, peaks_y, "o", markersize=3, c='red')
plt.title('Moving Average')
plt.tight_layout()




#%% 自己嘗試的Detective Rpeak Algo
'''
#Rpeak決策

pki = peaks_x,peaks_y #經過帶通濾波、微分、平方與移動平方處理的訊號中，可能為Rpeak的標記，位置與振幅表示
pkf = peaks_x-30, peaks_y #經過帶通濾波處理的訊號中，可能為Rpeak的標記，位置與振幅表示

pki_x = 0.0
pki_y = 0.0
pky_x = 0.0
pky_y = 0.0
spkf = 0.0
#初始化設定
Thr_I1 = np.max(ecg_square[0:500])/3
Thr_I2 = np.mean(ecg_square[0:500])/2
Thr_F1 = np.max(ecg_bandpass[0:500])/3
Thr_F2 = np.mean(ecg_bandpass[0:500])/2

# pki中計算RRn與RRavg1
rr = np.diff(peaks_x)


for i in range(len(rr)):
    if i <8:
        rravg=(np.mean(rr[0:i+1]))
    elif i>=8:        
        rravg=(np.mean(rr[i-7:i+1])) #RRavg1=(RRn-7 + RRn-6 + ... + RRn)/8
                
        
    if rr[i] > rravg*0.92 and rr[i] < rravg*1.16: #Step4 判斷是否在範圍內 公式：0.92RRavg1 < RRn < 1.16RRavg2
        if i == 0:
            continue
        else:
            index = i
            pkirr_x = (rr[i])
            pkirr_index = (i)
            pki_x = pki[0][i]
            pki_y = pki[1][i]
            break
            # print(str(rr[i])+'is in the range')
    else:
        # print(str(rr[i])+'is not! in the range')
        continue

real_rpeak_x = []
real_rpeak_y = []

if pki_y>Thr_I1: #Step6
    print('判斷yi是否大於THR_I1，此PKI可能為R波，但仍需檢查PKF是否也符合條件')
    if pkirr_x>360: #Step7
        pkf_y = pkf[1][pkirr_index-30] 
        pkf_x = pkf[0][pkirr_index-30]        
        if pkf_y > Thr_F1: #Step 8
            real_rpeak_x.append(pkf_x)
            real_rpeak_y.append(pkf_y)
            
            #spkf = 0.125*
            
            

                        
    elif pkirr_x<=360: #Step7
        print('步驟7 rr間距太短 必須確定是否誤把T波當成R波')
        
        
    

pki_newpeaks_x = []        
pki_newpeaks_y = []
spki = Thr_I1
for i in range(len(pki[0])):
    if pki[1][i] > spki: #步驟(6)
        pki_newpeaks_x.append(pki[0][i])
        pki_newpeaks_y.append(pki[1][i])
        spki = 0.125*pki[1][i]+0.875*spki
    
pkf_newpeaks_x = []        
pkf_newpeaks_y = []
spkf = Thr_F1
for i in range(len(pkf[0])):
    if pki[1][i] > spkf: #步驟(6)
        pkf_newpeaks_x.append(pkf[0][i])
        pkf_newpeaks_y.append(pkf[1][i])
        spkf = 0.125*pkf[1][i]+0.875*spkf

        
#(6)


    # result_x.append()
    
#(7) >360ms 也就是250*0.36 = 90個點
'''
