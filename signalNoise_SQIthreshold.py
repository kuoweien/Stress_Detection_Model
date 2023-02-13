#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:53:56 2022

@author: weien
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import def_getRpeak_main as getRpeak
import math


def normalize(data, lowest_rms, highest_rms):
    
    point = ((data-lowest_rms)/(highest_rms-lowest_rms))*100
    
    return point

def ecgEpochScore(ecg):
    rms = np.sqrt(np.mean(np.array(ecg)**2))
    noise_score = normalize(rms, 1186.45, 32767)
    return noise_score


#%% Add white noise to define SQI threshold

fs = 250
gaussian_filter_sigma =  0.03*fs 
medianfilter_size = 61
moving_average_ms = 2.5 
final_shift = 0 
detectR_maxvalue_range = (0.32*fs)*2  
rpeak_close_range = 0.15*fs 
lowpass_fq = 20
highpass_fq = 10

lta3_baseline = 0.9
lta3_magnification = 250

lta3_max_noise = 65535


url_data = 'Data/SQI/cleanECG_LTA3_r=1mV.csv'

# Create clean ECG
df_data = pd.read_csv(url_data)
ecg_clean=df_data['cleanECG']
ecg_clean_mV = (((np.array(ecg_clean))*1.8/65535-lta3_baseline)/lta3_magnification)*1000
ecg_clean_median = ecg_clean_mV-np.median(ecg_clean_mV)
lowest_rms = np.sqrt(np.mean(np.array(ecg_clean)**2))


# Create white noise
mean = 0
std = 10000 
num_samples = 500
whitenoise = np.random.normal(mean, std, size=num_samples)
rms_whitenoise_full = np.sqrt(np.mean(np.array(whitenoise)**2))

rms_cleanECG = np.sqrt(np.mean(np.array(ecg_clean)**2))
  
  
# Other signals may happended

rms_full = np.sqrt(np.mean(np.array((np.array([lta3_max_noise]*1000))**2))) # if noise is full
# rms_zero = np.sqrt(np.mean(np.array((np.array([0]*1000))**2))) # if noise is empty
# rms_half = np.sqrt(np.mean(np.array((np.array(([32767]*(num_samples/2))+list(ecg[0:(num_samples/2)]))**2)))  # if signal a hlaf is full and a half is empty


# Finding the Noise threshold

ecg_noise = ecg_clean+whitenoise*0.22
ecg_noise_mV = (((np.array(ecg_noise))*1.8/65535-lta3_baseline)/lta3_magnification)*1000
threshold_rms = np.sqrt(np.mean(np.array(ecg_noise)**2))
threshold_score = normalize(threshold_rms, lowest_rms, rms_full)
median_ecg_noise, ecg_noise_Rpeak_index = getRpeak.getRpeak_shannon(ecg_noise_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)


# Validation Data

url_data = '/Users/weien/Documents/GitHub/Stress_Detection_python/Data/ClipSituation_CSVfile/N23/Baseline.csv'
df_data = pd.read_csv(url_data)
ecg=df_data['ECG']
ecg_clip = ecg[0:500]
ecg_cleaninput_mV = (((np.array(ecg_clip))*1.8/65535-lta3_baseline)/lta3_magnification)*1000

ecg_median_inputclean, clean_redetect_Rpeak_index = getRpeak.getRpeak_shannon(ecg_cleaninput_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
data_rms = np.sqrt(np.mean(np.array(ecg_clip)**2))
input_cleanecg_score = normalize(data_rms, lowest_rms, rms_full)

print('Clean RMS:{}'.format(round(input_cleanecg_score,2)))



url_data = '/Users/weien/Documents/GitHub/Stress_Detection_python/Data/ClipSituation_CSVfile/Validation_Data/Shake_shoulder.csv'
df_data = pd.read_csv(url_data)
ecg_noise=df_data['ECG'][1000:1500]
ecg_cnoiseinput_mV = (((np.array(ecg_noise))*1.8/65535-lta3_baseline)/lta3_magnification)*1000

ecg_median_inputnoise, noise_redetect_Rpeak_index = getRpeak.getRpeak_shannon(ecg_cnoiseinput_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
noise_rms = np.sqrt(np.mean(np.array(ecg_noise)**2))
input_noiseecg_score = normalize(noise_rms, lowest_rms, rms_full)

print('Noise RMS:{}'.format(round(input_noiseecg_score,2)))



# plot

x_time = np.linspace(0, 2, 2*fs)

plt.figure(figsize=(8,6))
plt.subplot(4,1,1)
plt.plot(x_time, ecg_clean_median, c='black')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.ylim(-2, 2)
plt.xlim(0,2)
plt.title('Clean ECG', fontsize=16)
plt.yticks(fontsize=14)


plt.subplot(4,1,2)
plt.plot(x_time, median_ecg_noise ,c='black')
plt.scatter(np.array(ecg_noise_Rpeak_index)/fs, median_ecg_noise[ecg_noise_Rpeak_index], alpha=0.8, c='steelblue')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.ylim(-2, 2)
plt.xlim(0,2)
plt.title('Clean ECG with white noise', fontsize=16)
plt.yticks(fontsize=14)


plt.subplot(4,1,3)
plt.plot(x_time, ecg_median_inputclean, color='black')
plt.scatter(np.array(clean_redetect_Rpeak_index)/fs, ecg_median_inputclean[clean_redetect_Rpeak_index], color='steelblue', alpha=0.8)
plt.ylim(-2, 2)
plt.xlim(0,2)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.title('Input clean ECG', fontsize=16)
plt.yticks(fontsize=14)


plt.subplot(4,1,4)
plt.plot(x_time, ecg_median_inputnoise, color='black')
plt.scatter(np.array(noise_redetect_Rpeak_index)/fs, ecg_median_inputnoise[noise_redetect_Rpeak_index], color='steelblue', alpha=0.8)
plt.ylim(-2, 2)
plt.xlim(0,2)
plt.title('Input noisy ECG', fontsize=16)
plt.xlabel('Time (s)', fontsize=16)
ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()

plt.savefig('Noisy.png', dpi=800)


#%% Reference methods: measure statistic SQI (Leave the bandpass filter(5-30) as noise signal)

# measure skewness and kurtosis

'''
url = '/Users/weien/Desktop/ECG穿戴/HRV實驗/狗/Dataset/2110Nimo/scared.csv'
# url = '/Users/weien/Desktop/人體壓力測試/220426陳云/Stroop+Radio.csv'
situation = 'scared'


fs = 250
length_s = 10
index = 6
magnification = 500
lowpass_fs = 30
highpass_fs = 5
medianfilter_size = 61


for index in range(0,12):

    clip_start = index*(length_s*fs) #scared 2500 #petted 10000
    clip_end = (index+1)*(length_s*fs)  #scared 5000 #petted 12500
    
    df=pd.read_csv(url).iloc[clip_start:clip_end]
    rawdata = df[situation].reset_index(drop = True)
    rawdata_mV = ((rawdata*(1.8/65535)-0.9)/magnification)*1000  #轉換為電壓
    median_filter_data = getRpeak.medfilt(np.array(rawdata_mV), medianfilter_size)
    rawdata_mV_medianfilter = rawdata_mV - median_filter_data
    lowpass_data = getRpeak.lowPassFilter(lowpass_fs, rawdata_mV_medianfilter)  #低通
    bandfilter_data = getRpeak.highPassFilter(highpass_fs, lowpass_data)    #高通
    
    signal = bandfilter_data
    noise = rawdata_mV-bandfilter_data
    
    signal_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    rmsSQI = signal_rms/noise_rms
    print('RMS')
    print('index='+str(index))
    print(round(signal_rms,3))
    print(round(noise_rms,3))
    print(round(rmsSQI,3))
    print('')
       
    #Signal Quality Indices
    y = rawdata_mV  
    mean = np.mean(y)
    std = np.std(y)
    n = len(y)    
    #SQI
    sqi_snr =  (std**2) / (np.std(np.abs(y))**2)
    sqi_skew = np.sum(((y-mean)/std)**3)/n
    sqi_snr = 10*math.log10(sqi_snr)
    sqi_kur = np.sum(((y-mean)/std)**4)/n
    
    plt.subplot(6,2,index+1)
    plt.plot(rawdata_mV, color='black')
    plt.ylim(-1,1)
    plt.title('rmsSQI='+str(round(rmsSQI,3)))
    plt.tight_layout()

    

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(rawdata_mV, color='black')
plt.ylim(-1,1)

plt.subplot(3,1,2)
plt.plot(signal, color='black')
plt.ylim(-1,1)

plt.subplot(3,1,3)
plt.plot(noise, color='black')
plt.ylim(-1,1)

plt.tight_layout()

plt.title('Measure Noise')


#SQI
y = rawdata_mV
mean = np.mean(y)
std = np.std(y)
n = len(y)


sqi_snr =  (std**2) / (np.std(np.abs(y))**2)
sqi_skew = np.sum(((y-mean)/std)**3)/n
sqi_snr = 10*math.log10(sqi_snr)
sqi_kur = np.sum(((y-mean)/std)**4)/n
sqi_hos = np.abs(sqi_skew) * (sqi_kur/5)



print('sqi_skew: {}'.format(round(sqi_skew,3)))
print('sqi_snr: {}'.format(round(sqi_snr,3)))
print('sqi_kur: {}'.format(round(sqi_kur,3)))


# #畫直方圖
# sns.set_style("white")
# plt.figure(figsize=(10,7), dpi= 80)
# hist_binvalue = []
# sns.histplot(y, alpha=0.6, linewidth=2, color='black', kde=True, label='RR Interval') #kde畫常態線
# plt.grid(True)
# plt.xlim(-1,1)
# plt.ylim(0,200)
# plt.legend()
'''




