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

def interpolate(signal, output_signal_len):   # signal:原始訊號, output_signal_len:產生為多少長度之訊號
    x = np.linspace(0, len(signal)-1, num=len(signal), endpoint=True)
    f = interp1d(x, signal, kind='cubic')
    xnew = np.linspace(0, len(signal)-1, num=output_signal_len, endpoint=True)
    return f(xnew)

def window_function(window_len, window_type='hanning'):
    if window_type == 'hanning':
        return np.hanning(window_len)
    elif window_type == 'hamming':
        return np.hamming(window_len)

def fft_power(signal, sampling_rate, window_type):
    w = window_function(len(signal), window_type)
    window_coherent_amplification = sum(w)/len(w)
    y_f = np.fft.fft(signal*w)
    y_f_Real = 2.0/len(signal) * np.abs(y_f[:len(signal)//2])/window_coherent_amplification
    x_f = np.linspace(0.0, 1.0/(2.0*1/sampling_rate), len(signal)//2)
    return y_f_Real, x_f

def medfilt(x, k):  # 基線飄移, x:訊號 k:摺照大小
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)  # 做完之後還要再用原始訊號減此值

def sliding_meanSD_to_filter_RRI(rri_list, epoch):
    
    clean_rri = rri_list[0:epoch]
    
    standard_rri_epoch = rri_list[0*epoch:1*epoch]
    
    for i in range(int(len(rri_list)/epoch)-2):
        
        # rri_standard_epoch=rri_list[i*epoch : (i+1)*epoch]
        input_rri_epoch = rri_list[(i+1)*epoch:(i+2)*epoch]
        
        standard_mean = np.mean(standard_rri_epoch)
        standard_sd = np.std(standard_rri_epoch)
        
        input_mean = np.mean(input_rri_epoch)
        input_sd = np.std(input_rri_epoch)
        
        if (standard_sd*5 < input_sd) and (standard_mean+50 < input_mean) and (standard_mean+50 < input_mean):  # defined as noisy RRI
            continue
        
        else:  # Normal RRI
            clean_rri = list(clean_rri) + list(input_rri_epoch)
            standard_rri_epoch = rri_list[(i+1)*epoch:(i+2)*epoch]
    
    return clean_rri

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def plot_fft(emg, rri, x_f_ECG, y_f_ECG, x_f_EMG, y_f_EMG):
    plt_len = 5
    plt.subplot(plt_len, 1, 1)
    plt.plot(np.linspace(0,150,37500), emg, 'black')
    plt.ylabel('EMG\n(mV)')
    plt.xlabel('Time (s)')
    plt.ylim(-2,2)
    plt.xlim(0,150)

    plt.subplot(plt_len, 1, 2)
    plt.plot(rri, 'black', marker='o')
    plt.ylim(200, 1000)
    plt.ylabel('RRI\n(ms)')
    plt.xlabel('Time (s)')

    plt.subplot(plt_len, 1, 3)
    plt.ylabel('Re RRI\n(ms)')
    plt.xlabel('Time (s)')
    plt.ylim(200,1000)

    plt.subplot(plt_len, 1, 4)
    plt.plot(x_f_ECG, y_f_ECG/100, 'black')
    plt.xlim(0.0, 0.5)
    plt.ylim(0, 1)
    plt.ylabel('PSDRR\n($ms^2$/Hz)')
    plt.xlabel('Frequency (Hz)')

    plt.subplot(plt_len, 1, 5)
    plt.plot(x_f_EMG, y_f_EMG/100,'black')
    plt.xlim(0.0, 125)
    plt.ylabel('PSDEMG\n($ms^2$/Hz)')
    plt.xlabel('Frequency (Hz)')

    plt.tight_layout()

def plot_fq_epoch(tp, hf, lf, vlf, nlf, nhf, lfhf_ratio):

    # Frequency Domain Epoch
    plt_len = 7
    plt.figure()
    plt.subplot(plt_len, 1, 1)
    plt.plot(tp)
    plt.subplot(plt_len, 1, 2)
    plt.plot(hf)
    plt.subplot(plt_len, 1, 3)
    plt.plot(lf)
    plt.subplot(plt_len, 1, 4)
    plt.plot(vlf)
    plt.subplot(plt_len, 1, 5)
    plt.plot(nlf)
    plt.subplot(plt_len, 1, 6)
    plt.plot(nhf)
    plt.subplot(plt_len, 1, 7)
    plt.plot(lfhf_ratio)


def concatenate_ecg(n, situations):
    ecg_allsituation = []

    for situation in situations:
        ecg_url = 'Data/ClipSituation_CSVfile/N{}/{}.csv'.format(n, situation)  # 讀取之ECG csv檔
        df = pd.read_csv(ecg_url)
        ecg_situation = df['ECG']
        ecg_allsituation.extend(ecg_situation)

    return ecg_allsituation

def get_frequencydomian_features(n):

    """
    input: [int] a participant number
    return: [dataframe] the participant's frequency domain features
    """

    # ------Setting variables--------
    # Read data parameters
    fs = 250
    lta3_baseline = 0.9
    lta3_magnification = 250

    # Sliding window
    epoch_len = 150  # seconds
    rr_resample_rate = 7  # HZ
    slidingwidow_len = 30  # seconds
    minute_to_second = 60
    situation_time = 300  # sec
    epoch_time = 30  # sec

    # SQI
    checknoise_threshold = 20

    # Parameters of r peak detection
    medianfilter_size = 61
    gaussian_filter_sigma = 0.03 * fs
    moving_average_ms = 2.5
    final_shift = 0
    detectR_max_range = (0.32 * fs) * 2
    rpeak_close_range = 0.4 * fs  # 0.1*fs

    # Parameters of EMG
    qrs_range = int(0.32 * fs)  # Human: int(0.32*fs)
    tpeak_range = int(0.2 * fs)  # Human: int(0.2*fs)

    situations = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
    situations_offset = ['Baseline', 'Stroop', 'Baseline_after_stroop', 'Arithmetic', 'Baseline_after_Arithmetic',
                         'Speech_3m', 'Baseline_after_speech']
    df_onen_fqdomain = pd.DataFrame()
    # ------End of setting variables--------


    # ------Load data--------
    ecg = concatenate_ecg(n, situations_offset)
    ecg_without_noise = measureSQI.replace_noisy_ecg_tozero(ecg, fs, checknoise_threshold)  # 兩秒為Epoch，將雜訊的Y值改為0
    ecg_mV = (((np.array(ecg_without_noise)) * 1.8 / 65535 - lta3_baseline) / lta3_magnification) * 1000

    # %% 計算頻域
    baseline_strart_index = 0 * minute_to_second * fs
    stroop_start_index = 5 * minute_to_second * fs
    arithmetic_start_index = 15 * minute_to_second * fs
    speech_start_index = 25 * minute_to_second * fs
    columns_index = [baseline_strart_index, stroop_start_index, arithmetic_start_index, speech_start_index]

    # Run for situations
    for situation in range(len(situations)):

        # Run for epochs
        for i in range(0, int(situation_time / slidingwidow_len)):
            print('Paricipant:{} Situation: {} Epoch:{}'.format(n, situations[situation], i))

            # overlapping: (i*slidingwidow_len*fs), sliding window: (2.5*minute_to_second*fs)
            input_ecg = ecg_mV[columns_index[situation] + (i * slidingwidow_len * fs): int(
                (columns_index[situation] + (2.5 * minute_to_second * fs)) + (i * slidingwidow_len * fs))]
            # Get R peak from ECG by using shannon algorithm
            median_ecg, rpeakindex = getRpeak.getRpeak_shannon(input_ecg, fs, medianfilter_size, gaussian_filter_sigma,
                                                               moving_average_ms, final_shift, detectR_max_range,
                                                               rpeak_close_range)

            # ------ECG (RRI) features--------
            if len(rpeakindex) <= 2:  # 若只取到<=2點的Rpeak，會無法算HRV參數，因此將參數設為0
                tp_log, hf_log, vlf_log, nlf, nhf, lfhf_ratio_log, mnf, mdf = 0, 0, 0, 0, 0, 0, 0, 0

            else:  # 若Rpeak有多於2個點，進行HRV參數計算

                rrinterval = np.diff(rpeakindex)
                rrinterval = rrinterval / (fs / 1000)  # RRI index點數要換算回ms (fs/1000是因為要換算成毫秒)
                re_rrinterval = getRpeak.interpolate_rri(rrinterval, fs)  # 對因雜訊刪除的RRI進行補點

                # RRI 相關參數
                re_rri_mean = np.mean(re_rrinterval)
                re_rri_sd = np.std(re_rrinterval)

                outlier_upper = re_rri_mean + (3 * re_rri_sd)
                outlier_lower = re_rri_mean - (3 * re_rri_sd)

                re_rrinterval = re_rrinterval[re_rrinterval < outlier_upper]
                re_rrinterval = re_rrinterval[re_rrinterval > outlier_lower]  # 刪除outlier的rrinterval

                rrinterval_resample = interpolate(re_rrinterval, rr_resample_rate * epoch_len)  # 補點為rr_resample_rate HZ
                x_rrinterval_resample = np.linspace(0, epoch_len, len(rrinterval_resample))
                rrinterval_resample_zeromean = rrinterval_resample - np.mean(rrinterval_resample)

                # 刪除RTpeak，並以差值補點結果視為EMG
                emg_mV_linearwithzero, _ = getRpeak.deleteRTpeak(median_ecg, rpeakindex, qrs_range, tpeak_range)  # 刪除rtpeak並補0

                # Get frequency domain by using FFT
                y_f_ECG, x_f_ECG = fft_power(rrinterval_resample_zeromean, rr_resample_rate, 'hanning')
                y_f_EMG, x_f_EMG = fft_power(emg_mV_linearwithzero, fs, 'hanning')

                # Calculate ECG frequency domain parameters
                tp_index, hf_index, lf_index, vlf_index, ulf_index = [], [], [], [], []
                tp_index.append(np.where((x_f_ECG <= 0.4)))
                hf_index.append(np.where((x_f_ECG >= 0.15) & (x_f_ECG <= 0.4)))
                lf_index.append(np.where((x_f_ECG >= 0.04) & (x_f_ECG <= 0.15)))
                vlf_index.append(np.where((x_f_ECG >= 0.003) & (x_f_ECG <= 0.04)))
                ulf_index.append(np.where((x_f_ECG <= 0.003)))
                tp_index, hf_index, lf_index, vlf_index, ulf_index = tp_index[0][0], hf_index[0][0], lf_index[0][0], vlf_index[0][0], ulf_index[0][0]
                tp = np.sum(y_f_ECG[tp_index[0]:tp_index[-1]])
                hf = np.sum(y_f_ECG[hf_index[0]:hf_index[-1]])
                lf = np.sum(y_f_ECG[lf_index[0]:lf_index[-1]])
                vlf = np.sum(y_f_ECG[vlf_index[0]:vlf_index[-1]])
                # ulf = np.log(np.sum(y_f_ECG[ulf_index[0]:ulf_index[-1]]))  # ulf usually got 0
                nlF = (lf / (tp - vlf)) * 100
                nhf = (hf / (tp - vlf)) * 100
                lfhf_ratio_log = np.log(lf / hf)
                tp_log, hf_log, lf_log, vlf_log = np.log(tp), np.log(hf), np.log(lf), np.log(vlf)

                # Calculate EMG frequency domain parameters
                mnf = np.sum(y_f_EMG * x_f_EMG) / np.sum(y_f_EMG)
                y_f_EMG_integral = np.cumsum(y_f_EMG)
                mdf_median_index = (np.where(y_f_EMG_integral > np.max(y_f_EMG_integral) / 2))[0][0]  # Array is bigger than (under area)/2
                mdf = x_f_EMG[mdf_median_index]

                df_now_fqdomain = pd.DataFrame({'N': n, 'Epoch': i + 1, 'Situation': situation,
                  'TP': tp_log, 'HF': hf_log, 'LF': lf_log, 'VLF': vlf_log,
                  'nLF': nlF, 'nHF': nhf, 'LF/HF': lfhf_ratio_log,
                  'MNF': mnf, 'MDF': mdf}, index=[0])

                df_onen_fqdomain = pd.concat([df_onen_fqdomain, df_now_fqdomain])

    return df_onen_fqdomain
