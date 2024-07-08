#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:23:50 2022

@author: weien
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import def_getRpeak_main as getRpeak
import def_measureSQI as measureSQI
from scipy.interpolate import interp1d  # 導入 scipy 中的一維插值工具 interp1d
import math


def get_timedomian_features(n):

    """
    input: [int] a participant number
    return: [dataframe] the participant's time domain features
    """

    # ------Setting variables--------
    df_one_n_timedomain = pd.DataFrame()
    situations = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']

    fs = 250
    checknoise_threshold = 20  # threshold of deleting noisy, epoch=2s
    lta3_baseline = 0.9
    lta3_magnification = 250

    # Parameters of detecting R peak
    situation_time = 300  # sec
    epoch_time = 30  # sec

    # Parameters of EMG
    qrs_range = int(0.32 * fs)
    tpeak_range = int(0.2 * fs)
    # ------End of setting variables--------


    # ------Load data--------
    # Run for situations
    for situation in situations:
        ecg_url = 'Data/ClipSituation_CSVfile/N{}/{}.csv'.format(n, situation)  # 讀取之ECG csv檔
        df = pd.read_csv(ecg_url)
        ecg_situation = df['ECG']

        #  Run for epochs
        for i in range(0, int(situation_time / epoch_time)):
            print('Paricipant:{} Situation: {} Epoch:{}'.format(n, situation, i))
            ecg_epoch = ecg_situation[i * fs * epoch_time:(i + 1) * epoch_time * fs]
            if len(ecg_epoch) < (epoch_time * fs):
                break

            # ------Preprocessing--------
            ecg_clean = measureSQI.replace_noisy_ecg_tozero(ecg_epoch, fs, checknoise_threshold)  # 將雜訊的Y值改為0
            ecg_nonoise_mV = (((np.array(ecg_clean)) * 1.8 / 65535 - lta3_baseline) / lta3_magnification) * 1000
            ecg_mV = (((np.array(ecg_nonoise_mV)) * 1.8 / 65535 - lta3_baseline) / lta3_magnification) * 1000

            # get R peak location
            median_ecg, rpeakindex = getRpeak.getRpeak_shannon(ecg_mV, fs)

            # ------ECG (RRI) features--------
            if len(rpeakindex) <= 2:  # 排除: 若只抓到小於等於2點的Rpeak，會無法算HRV參數，因此將參數設為0
                rri_mean, rri_sd, rri_rmssd, rri_nn50, rri_pnn50, rri_skew, rri_kurt = 0, 0, 0, 0, 0, 0, 0

            else:  # 若Rpeak有多於2個點，進行HRV參數計算
                rrinterval = np.diff(rpeakindex)
                rrinterval = rrinterval / (fs / 1000)  # RRI index點數要換算回ms (fs/1000是因為要換算成毫秒)
                re_rrinterval = getRpeak.interpolate_rri(rrinterval, fs)  # 對因雜訊刪除的RRI進行補點

                # RRI 相關參數
                outlier_upper = np.mean(re_rrinterval) + (3 * np.std(re_rrinterval))
                outlier_lower = np.mean(re_rrinterval) - (3 * np.std(re_rrinterval))
                re_rrinterval = re_rrinterval[re_rrinterval < outlier_upper]
                re_rrinterval = re_rrinterval[re_rrinterval > outlier_lower]  # 刪除outlier的rrinterval

                # 因有刪除outlier，所以重新計算平均與SD
                rri_mean = np.mean(re_rrinterval)
                rri_sd = np.std(re_rrinterval)
                [niu, sigma, rri_skew, rri_kurt] = getRpeak.calc_stat(re_rrinterval)  # 峰值與偏度
                rri_rmssd = math.sqrt(np.mean((np.diff(re_rrinterval) ** 2)))  # RMSSD
                rri_nn50 = len(np.where(np.abs(np.diff(re_rrinterval)) > 50)[0])  # NN50 心跳間距超過50ms的個數，藉此評估交感
                rri_pnn50 = rri_nn50 / len(re_rrinterval)

            # ------EMG features--------
            emg_mv_linearwithzero, _ = getRpeak.delete_rtpeak(median_ecg, rpeakindex, qrs_range, tpeak_range)  # 刪除rtpeak並補0
            emg_mv_deletezero = getRpeak.delete_zero(emg_mv_linearwithzero)
            emg_rms = np.sqrt(np.mean(emg_mv_deletezero ** 2))
            emg_var = np.var(emg_mv_deletezero)
            emg_mav = np.sqrt(np.mean(np.abs(emg_mv_deletezero)))
            emg_energy = np.sum((np.abs(emg_mv_deletezero)) ** 2)
            emg_zc = 0  # 交0的次數 公式：{xi >0andxi+1 <0}or{xi <0andxi+1 >0}
            emg_mv_deletezero = emg_mv_deletezero.reset_index(drop=True)
            for x in range(len(emg_mv_deletezero) - 1):
                if (emg_mv_deletezero[x] > 0 and emg_mv_deletezero[x + 1] < 0) or (
                        emg_mv_deletezero[x] < 0 and emg_mv_deletezero[x + 1] > 0):
                    emg_zc += 1

            # Save data
            df_oneepoch_timedomain = pd.DataFrame({'N': n, 'Epoch': i+1, 'Situation': situation, 'Mean': rri_mean, 'SD': rri_sd, 'RMSSD': rri_rmssd,
                                                  'NN50': rri_nn50, 'pNN50': rri_pnn50, 'Skewness': rri_skew,
                                                  'Kurtosis': rri_kurt,
                                                  'EMG_RMS': emg_rms, 'EMG_VAR': emg_var, 'EMG_MAV': emg_mav,
                                                  'EMG_ENERGY': emg_energy,
                                                  'EMG_ZC': emg_zc}, index=[0])

            df_one_n_timedomain = pd.concat([df_one_n_timedomain, df_oneepoch_timedomain])


    return df_one_n_timedomain

