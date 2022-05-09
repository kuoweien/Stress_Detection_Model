#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:59:38 2022

@author: weien
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_ECG=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/isolation.csv').iloc[12000:22000]
rawdata=df_ECG['isolation']
rawdata=rawdata.reset_index(drop=True)

df=pd.read_excel('filterEMG.xlsx')
isolation=df['isolation']


epoch=1
sample=epoch*250 #2500個點算一個rms值
one_epoch=len(isolation)/sample
isolation_dim=np.array(isolation).reshape(int(one_epoch),int(len(isolation)/one_epoch))

rms=[]
for i in range(len(isolation_dim)):
    temp_rms=np.sqrt(np.mean(isolation_dim[i]**2))
    rms.append(round(temp_rms,3))

x=range(0,10000)

plt.figure(figsize=(16,6))
plt.subplot(3,1,1)
plt.plot(rawdata)
plt.xlim(0,10000)
plt.title('RawData')

plt.subplot(3,1,2)
plt.plot(isolation)
plt.xlim(0,10000)
plt.title('EMG')

plt.subplot(3,1,3)
plt.plot(rms,'-o')
plt.xlim(0,40)
plt.title('RMS')

plt.tight_layout()

