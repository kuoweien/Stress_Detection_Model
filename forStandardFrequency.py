#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:04:12 2022

@author: weien
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:01:57 2022

@author: weien
"""

import pandas as pd
import numpy as np
import def_getRpeak_main as getRpeak
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import scipy.fft
import def_readandget_Rawdata
import def_measureSQI as measureSQI

def interpolate(raw_signal,n):   #signal為原始訊號 n為要插入產生為多少長度之訊號

    x = np.linspace(0, len(raw_signal)-1, num=len(raw_signal), endpoint=True)
    f = interp1d(x, raw_signal, kind='cubic')
    xnew = np.linspace(0, len(raw_signal)-1, num=n, endpoint=True)  
    
    return f(xnew)

def window_function(window_len,window_type='hanning'):
    if window_type=='hanning':
        return np.hanning(window_len)
    elif window_type=='hamming':
        return np.hamming(window_len)

def fft_power(input_signal,sampling_rate,window_type):
    w=window_function(len(input_signal))
    window_coherent_amplification=sum(w)/len(w)
    y_f = np.fft.fft(input_signal*w)
    y_f_Real= 2.0/len(input_signal) * np.abs(y_f[:len(input_signal)//2])/window_coherent_amplification
    x_f = np.linspace(0.0, 1.0/(2.0*1/sampling_rate), len(input_signal)//2)        
    return y_f_Real,x_f

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
'''
t = np.linspace( 0, 10, 1000, endpoint = False ) # 定義時間陣列
x = np.sin( 2 * np.pi * 0.2 * t ) 
y_f_Real, x_f = fft_power(x, 100, 'hamming')

plt.figure()
plt.subplot(211)
plt.plot(t,x)
plt.subplot(212)
plt.plot(x_f,y_f_Real)
'''

n = 31

# Read data parameters
lta3_baseline = 0.9
lta3_magnification = 250
fs = 250

# Sliding window parameters
epoch_len = 150 # seconds
rr_resample_rate = 7
slidingwidow_len = 30 #seconds
epoch = 2.5 # minutes
minute_to_second = 60

# Noise threshold
checknoiseThreshold = 30

# 抓Rpeak的參數
medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.32*fs)*2  #草哥使用(0.3*fs)*2 #Patch=0.4*fs*2 LTA3=0.35*fs*2
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 20
highpass_fq = 10

# EMG參數
qrs_range = int(0.32*fs)    # Human: int(0.32*fs)
tpeak_range = int(0.2*fs)   # Human: int(0.2*fs)



input_N_start = 31
input_N_end = 31
df_output_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/HRV/LTA3/Features/220801_FrequencyDomain.xlsx'


df_HRV_fqdomain = pd.DataFrame()

tp_HRV = []
hf_HRV = []
lf_HRV = []
vlf_HRV = []
nLF_HRV = []
nHF_HRV = []
lfhf_ratio_hrv = []
rpeakindex_list = []
median_ecg_list = []


    
ecg_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/ClipSituation_eachN/N{}/'.format(n)
filename_baseline = 'Baseline.csv'
filename_stroop = 'Stroop.csv'
filename_b2 = 'Baseline_after_stroop.csv'
filename_arithmetic = 'Arithmetic.csv'
filename_b3 =  'Baseline_after_Arithmetic.csv'
filename_speech = 'Speech.csv'
filename_b4 = 'Baseline_after_speech.csv'


df_baseline1 = pd.read_csv(ecg_url+filename_baseline)
df_stroop = pd.read_csv(ecg_url+filename_stroop)
df_baseline2 = pd.read_csv(ecg_url+filename_b2)
df_arithmetic = pd.read_csv(ecg_url+filename_arithmetic)
df_baseline3= pd.read_csv(ecg_url+filename_b3)
df_speech = pd.read_csv(ecg_url+filename_speech)
df_baseline4 = pd.read_csv(ecg_url+filename_b4)


ecg_baseline1 =  np.array(df_baseline1['ECG'])
ecg_stroop = np.array(df_stroop['ECG'])
ecg_baseline2 = np.array(df_baseline2['ECG'])
ecg_arithmetic = np.array(df_arithmetic['ECG'])
ecg_baseline3 = np.array(df_baseline3['ECG'])
ecg_speech = np.array(df_speech['ECG'])
ecg_baseline4 = np.array(df_baseline4['ECG'])

# Rebuild protocal data
ecg_raw = np.concatenate((ecg_baseline1, ecg_stroop, ecg_baseline2, ecg_arithmetic, ecg_baseline3, ecg_speech, ecg_baseline4))
ecg_without_noise = measureSQI.splitEpochandisCleanSignal(ecg_raw, fs, checknoiseThreshold) #兩秒為Epoch，將雜訊的Y值改為0
ecg_mV = (((np.array(ecg_without_noise))*1.8/65535-lta3_baseline)/lta3_magnification)*1000


#%%計算頻域

data_n = int(len(ecg_raw)/(slidingwidow_len*250))
    
for i in range(65, data_n): # one stress situation have 10 data
    print(i)

    # overlapping: (i*slidingwidow_len*fs) 
    # sliding window: (2.5*minute_to_second*fs)
    input_ecg = ecg_mV[(i*slidingwidow_len*fs) :  int((epoch*minute_to_second*fs)+(i*slidingwidow_len*fs))]
        
    # Get R peak from ECG by using shannon algorithm
    median_ecg, rpeakindex = getRpeak.getRpeak_shannon(input_ecg, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
    
    
    if len(rpeakindex)<=2: #若只抓到小於等於2點的Rpeak，會無法算HRV參數，因此將參數設為0
                tp_log =0
                hf_log = 0
                vlf_log = 0
                nLF = 0
                nHF = 0
                lfhf_ratio_log = 0
                mnf = 0
                mdf = 0
               
    else: #若Rpeak有多於2個點，進行HRV參數計算
                
        # RRI計算
        rrinterval = np.diff(rpeakindex)
        rrinterval = rrinterval/(fs/1000) #RRI index點數要換算回ms (%fs，1000是因為要換算成毫秒)
        rrinterval_resample = interpolate(rrinterval, rr_resample_rate*epoch_len) #補點為rr_resample_rate HZ
        x_rrinterval_resample = np.linspace(0, epoch_len, len(rrinterval_resample))
        
        rrinterval_resample_zeromean=rrinterval_resample-np.mean(rrinterval_resample)
        
        # EMG計算
        emg_mV_linearwithzero, _ = getRpeak.deleteRTpeak(median_ecg,rpeakindex, qrs_range, tpeak_range) #刪除rtpeak並補0       
           
        # FFT轉頻域
        y_f_ECG, x_f_ECG = fft_power(rrinterval_resample_zeromean, rr_resample_rate, 'hanning')
        y_f_EMG, x_f_EMG = fft_power(emg_mV_linearwithzero, fs, 'hanning')
        
        
    
        ## Calculate HRV frequency domain parameters
        tp_index = []
        hf_index = []
        lf_index = []
        vlf_index = []
        ulf_index = []
            
        
        tp_index.append(np.where( (x_f_ECG<=0.4)))  
        hf_index.append(np.where( (x_f_ECG>=0.15) & (x_f_ECG<=0.4)))  
        lf_index.append(np.where( (x_f_ECG>=0.04) & (x_f_ECG<=0.15)))  
        vlf_index.append(np.where( (x_f_ECG>=0.003) & (x_f_ECG<=0.04)))   
        ulf_index.append(np.where( (x_f_ECG<=0.003)))   
        
        
        tp_index = tp_index[0][0]
        hf_index = hf_index[0][0]
        lf_index = lf_index[0][0]
        vlf_index = vlf_index[0][0]
        ulf_index = ulf_index[0][0]
        
        
        tp = np.sum(y_f_ECG[tp_index[0]:tp_index[-1]])
        hf = np.sum(y_f_ECG[hf_index[0]:hf_index[-1]])
        lf = np.sum(y_f_ECG[lf_index[0]:lf_index[-1]])
        vlf = np.sum(y_f_ECG[vlf_index[0]:vlf_index[-1]])
        # ulf = np.log(np.sum(y_f_ECG[ulf_index[0]:ulf_index[-1]]))
        nLF = (lf/(tp-vlf))*100
        nHF = (hf/(tp-vlf))*100
        lfhf_ratio_log = np.log(lf/hf)
        
        tp_log = np.log(tp)
        hf_log = np.log(hf)
        lf_log = np.log(lf)
        vlf_log = np.log(vlf)
        
        tp_HRV.append(tp_log)
        hf_HRV.append(hf_log)
        lf_HRV.append(lf_log)
        vlf_HRV.append(vlf_log)
        nLF_HRV.append(nLF)
        nHF_HRV.append(nHF)
        lfhf_ratio_hrv.append(lfhf_ratio_log)
        
        
        ## Calculate EMG frequency domain parameters
        mnf = np.sum(y_f_EMG)/len(x_f_EMG)
        y_f_EMG_integral = np.cumsum(y_f_EMG)
        mdf_median_index = (np.where(y_f_EMG_integral>np.max(y_f_EMG_integral)/2))[0][0] # Array is bigger than (under area)/2
        mdf = y_f_EMG[mdf_median_index]
        
        
        df_HRV_fqdomain = df_HRV_fqdomain.append({'N':n, 'Epoch':i+1, 
                                                  'TP':tp_log , 'HF':hf_log, 'LF':lf_log, 'VLF':vlf_log,
                                                  'nLF':nLF, 'nHF':nHF , 'LF/HF':lfhf_ratio_log, 
                                                  'MNF': mnf, 'MDF': mdf
                                                 } ,ignore_index=True)
        
        plt_len = 5
        
        plt.figure(figsize=(6,10))
        plt.subplot(plt_len,1,1)
        plt.plot(median_ecg, 'black')
        plt.scatter(rpeakindex, median_ecg[rpeakindex], color='red')
        plt.ylabel('Raw ECG\n(mV)')
        plt.ylim(-2,2)
        plt.xlim(0,150)
        plt.xlabel('Time (s)')
        

        plt.subplot(plt_len,1,2)
        plt.plot(np.linspace(0,150,37500), emg_mV_linearwithzero, 'black')
        plt.ylabel('EMG\n(mV)')
        plt.ylim(-2,2)
        plt.xlim(0,150)
        plt.xlabel('Time (s)')
        
              
        plt.subplot(plt_len,1,3)
        plt.plot(x_rrinterval_resample, rrinterval_resample, 'black')
        plt.xlim(0,150)
        plt.ylabel('RR\n(ms)')
        plt.xlabel('Time (s)')
        
        
        plt.subplot(plt_len,1,4)
        plt.plot(x_f_ECG, y_f_ECG/100, 'black')
        plt.xlim(0.0,0.5)
        plt.ylabel('PSDRR\n($ms^2$/Hz)')
        plt.xlabel('Frequency (Hz)')
        
        plt.subplot(plt_len,1,5)
        plt.plot(x_f_EMG, y_f_EMG/100 ,'black')
        plt.xlim(0.0,125)
        plt.ylabel('PSDEMG\n($ms^2$/Hz)')
        plt.xlabel('Frequency (Hz)')
        
        plt.tight_layout()
    


# Frequency Domain Epoch
plt_len = 9

plt.subplot(plt_len,1,1)
plt.plot(ecg_mV, 'black')
plt.scatter(rpeakindex_list, median_ecg_list[rpeakindex_list], 'red')

plt.subplot(plt_len,1,2)
plt.plot(rrinterval_resample, 'black')

plt.subplot(plt_len,1,3)
plt.plot(tp_HRV, 'black')
plt.ylabel('TP\n$[ln(m{s^2)}$]')


plt.subplot(plt_len,1,4)
plt.plot(hf_HRV, 'black')
plt.ylabel('HF\n$[ln(m{s^2)}$]')

plt.subplot(plt_len,1,5)
plt.plot(lf_HRV, 'black')
plt.ylabel('LF\n$[ln(m{s^2)}$]')

plt.subplot(plt_len,1,6)
plt.plot(vlf_HRV, 'black')
plt.ylabel('VLF\n$[ln(m{s^2)}$]')

plt.subplot(plt_len,1,7)
plt.plot(nLF_HRV, 'black')
plt.ylabel('LF%\n(%)')

plt.subplot(plt_len,1,8)
plt.plot(nHF_HRV, 'black')
plt.ylabel('HF%\n(%)')

plt.subplot(plt_len,1,9)
plt.plot(lfhf_ratio_hrv, 'black')
plt.ylabel('LF/HF\n[ln(ratio)]')





