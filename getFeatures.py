#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:13:56 2022

@author: weien
"""

import pandas as pd

ecg_url = '/Users/weien/Desktop/ECG穿戴/實驗二_人體壓力/DataSet/ClipSituation_eachN/N27/'
filename_baseline = 'Baseline.csv'
filename_stroop = 'Stroop.csv'
filename_b2 = 'Baseline_after_stroop.csv'
filename_arithmetic = 'Arithmetic.csv'
filename_b3 =  'Baseline_after_Arithmetic.csv'
filename_speech = 'Speech_3m.csv'
filename_b4 = 'Baseline_after_speech.csv'


df_baseline1 = pd.read_csv(ecg_url+filename_baseline)
df_stroop = pd.read_csv(ecg_url+filename_stroop)
df_baseline2 = pd.read_csv(ecg_url+filename_b2)
df_arithmetic = pd.read_csv(ecg_url+filename_arithmetic)
df_baseline3= pd.read_csv(ecg_url+filename_b3)
df_speech = pd.read_csv(ecg_url+filename_speech)
df_baseline4 = pd.read_csv(ecg_url+filename_b4)


ecg_baseline1 = df_baseline1['ECG']
ecg_stroop = df_stroop['ECG']
ecg_baseline2 = df_baseline2['ECG']
ecg_arithmetic = df_arithmetic['ECG']
ecg_baseline3 = df_baseline3['ECG']
ecg_speech = df_speech['ECG']
ecg_baseline4 = df_baseline4['ECG']

