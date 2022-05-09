#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:23:50 2022

@author: weien
"""

import ECG_morphology_analysis
import pandas as pd
import matplotlib.pyplot as plt

getRpeak = ECG_morphology_analysis.ECG_morphology_analysis()


df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/petted.csv').iloc[10000:12500]
data = df['petted']
# data=data.reset_index(drop=True)
data = data.tolist()

peak_num, wave_range, peak_point_x, peak_point_y,peak_all,peak_all_value = getRpeak.R_peak(0,2500,data)

plt.plot(data)
plt.scatter(peak_point_x, peak_point_y,c='red')