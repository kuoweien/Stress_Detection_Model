#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:53:14 2022

@author: weien
"""

import os
import wfdb
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import Library.def_getRpeak_main as getRpeak
# from wfdb.processing.peaks import find_local_peaks
from wfdb.processing import ann2rr


if __name__ == '__main__':

    fs = 360
    lowpass_fq = 20
    highpass_fq = 10
    sig_len = 30

    n_list = []
    total_count_list = []
    true_positive_list = []
    false_positive_list = []
    false_negative_list = []
    sensitivity_list = []
    precision_list = []
    accuracy_list = []


    # n = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    #       111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    #       122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    #       209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    #       222, 223, 228, 230, 231, 232, 233, 234]
    n = [219]

    for i in n:

        real_peaks = []
        true_positive = []
        false_positive = []

        print(i)
        # Read data from MIT-BIH database
        record = wfdb.rdsamp('Data/MITBIH/'+str(i))
        annotation = wfdb.rdann('Data/MITBIH/'+str(i), 'atr')  # 2024/07/08 Error to read file

        # R peak algorithm
        ecg_data = record[0][:, 0]
        median_ecg, detect_rpeakindex = getRpeak.getRpeak_shannon(ecg_data, fs, relocated_rpeak_method="onebeat_choose_highorlow")

        # Evaluate performance
        peak_samples = annotation.sample
        peak_symbols = annotation.symbol

        for index, sym in enumerate(peak_symbols):
            # if sym == 'N' or sym == 'V' or sym=='A':
            if sym == '/' or sym == 'N' or sym == 'L' or sym == 'R' or sym == 'A' or sym == 'a' or sym == 'J' or sym == 'S' or sym == 'V' or sym == 'F' or sym == 'O' or sym == 'N' or sym == 'E' or sym == 'P' or sym == 'F' or sym == 'Q':
                real_peaks = np.append(real_peaks, index)

        real_peaks = real_peaks.astype(int)
        real_peaks_index = peak_samples[real_peaks]

        HitR = np.ones(len(real_peaks_index), dtype=bool)
        for indP, ValP in np.ndenumerate(detect_rpeakindex):
            Hit = 0
            for indR, ValR in np.ndenumerate(real_peaks_index):
                if np.absolute(ValP-ValR) < 50:  # 50個點前後都算準確
                    Hit = 1
                    HitR[indR[0]] = False
                    true_positive = np.append(true_positive, indP[0])
                    real_peaks_index = real_peaks_index[HitR]
                    HitR = HitR[HitR]
                    break
            if Hit == 0:
                false_positive = np.append(false_positive, indP[0])

        false_negative = HitR
        accuracy = len(true_positive)/(len(true_positive)+len(false_positive)+len(false_negative))
        sensitivity = len(true_positive)/(len(true_positive)+len(false_negative))
        precision = len(true_positive)/(len(true_positive)+len(false_positive))

        n_list.append(i)
        total_count_list.append(len(detect_rpeakindex))
        true_positive_list.append(len(true_positive))
        false_positive_list.append(len(false_positive))
        false_negative_list.append(len(HitR))
        accuracy_list.append(round(accuracy*100, 2))
        sensitivity_list.append(round(sensitivity*100, 2))
        precision_list.append(round(precision*100, 2))
        print('Total Count: {}'.format(len(detect_rpeakindex)))
        print('True Positive Count: {0:5d}'.format(len(true_positive)))
        print('False Positive Count: {0:d}'.format(len(false_positive)))
        print('False Negative Count: {0:d}'.format(len(false_negative)))
        print('ACC: {}%'.format(round(accuracy*100, 2)))
        print('SEN: {}%'.format(round(sensitivity*100, 2)))
        print('PRE: {}%'.format(round(precision*100, 2)))
        plt.figure(figsize=(12, 4))
        plt.plot(median_ecg, 'black')
        plt.scatter(detect_rpeakindex, median_ecg[detect_rpeakindex], c='blue', alpha=0.5)
        plt.scatter(peak_samples, median_ecg[peak_samples], c='red', alpha=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(range(-4, 5, 1), fontsize=14)
        plt.xlim(0, len(median_ecg))
        plt.ylim(-4, 4)
        plt.title('Record 104', fontsize=16)


    today = date.today()
    todaydate = today.strftime("%y%m%d")
    df = pd.DataFrame({'ECG record': n_list, 'Total (beats)': total_count_list, 'TP (beats)': true_positive_list, 'FP (beats)': false_positive_list , 'FN (beats)': false_negative_list, 'ACC (%)': accuracy_list, 'SEN ()%': sensitivity_list, 'PRE (%)': precision_list})
    df.to_excel('Data/Performance/{}_MITBIH_validation.xlsx'.format(todaydate))

