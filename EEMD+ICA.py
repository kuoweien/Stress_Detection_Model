#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:38:53 2021

@author: weien
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EEMD
import time
# from scipy import signal
from sklearn.decomposition import FastICA, PCA
import defgetRpeak as getRpeak



df=pd.read_csv('/Users/weien/Desktop/狗狗穿戴/HRV實驗/Dataset/2110Nimo/isolation.csv')
rawdata=df['isolation']
rawdata=rawdata[12000:22000]

rawdata_V=(rawdata*(1.8/65535)-0.9)/500

t = np.linspace(0, len(rawdata)/250,len(rawdata))

start=time.time()
S=np.array(rawdata)

print(time.time()-start)

start = time.time()
eemd = EEMD()
# 设置探测极值的方法
emd = eemd.EMD
emd.extrema_detection="parabol"
# 執行eemd
eIMFs = eemd.eemd(S,t)
nIMFs = eIMFs.shape[0]

print(time.time()-start)

plt.figure(figsize=(12,9))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(t,S, 'black')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(eIMFs[n], 'gray')
    plt.ylabel("eIMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)
    
#製作input ICA的值-> IMF留前3個(IMF1,2,3) 最後一個刪除 其他相加承一個
for i in range(3,nIMFs):
    temp=0
    highIMFs=temp+eIMFs[i]
    
forICA_IMFs=np.array([eIMFs[0],eIMFs[1],eIMFs[2],highIMFs])#建立成一個矩陣
forICA_IMFs=forICA_IMFs.T

ica = FastICA(n_components=4)
outputICA = ica.fit_transform(forICA_IMFs)  # Reconstruct signals



plt.figure(figsize=(12,9))
# n_outputICA=len(outputICA)
outputICA=outputICA.T
for i in range(len(outputICA)):
    plt.subplot(len(outputICA),1,i+1)
    plt.plot(outputICA[i])


#ICA還原
A=ica.mixing_
A_np=np.asarray(A)

row=1

A_end=A_np[:,row-1]
outputICA_np=np.asarray(outputICA)
outputICA_end = outputICA_np[row-1,:]
c=outputICA_end*np.array([A_end]).T
plt.figure(figsize=(12,9))
plt.plot(sum(c))


#抓Rpeak
df = pd.DataFrame({'ECG':sum(c)})
rawdatalist=df.ECG
ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,1.4)


plt.figure(figsize=(12,9))
plt.subplot(3,1,1)
plt.plot(t,rawdata)
plt.title('Rawdata')


plt.subplot(3,1,2)
plt.plot(t,sum(c))



plt.subplot(3,1,3)
plt.plot(x_time,rawdatalist)
plt.xlabel('Time')
plt.title('Isolation')
plt.scatter(peaklist_time, ybeat, c='red')

plt.tight_layout()


'''
#原始圖及乾淨圖取rpeak
df = pd.DataFrame({'Raw':S,'ECG':Xtrans_SUM})
rawdatalist=df.Raw
ybeat,x_time,peaklist_time=getRpeak.getYValueofRPeak(df,rawdatalist,0.5)

rawdatalist2=df.ECG
ybeat2,x_time2,peaklist_time2=getRpeak.getYValueofRPeak(df,rawdatalist2,20)

plt.figure()
plt.subplot(2,1,1)
plt.plot(df['Raw'])
plt.xlabel('Time')
plt.ylabel('Raw')
# plt.ylim(-2000,3000)
plt.scatter(peaklist_time, ybeat, c='red') #Plot detected peaks

plt.subplot(2,1,2)
plt.plot(df['ECG'])
plt.xlabel('Time')
plt.ylabel('ECG')
# plt.ylim(-2000,3000)
plt.scatter(peaklist_time2, ybeat2, c='red') #Plot detected peaks
plt.tight_layout()
  
'''
