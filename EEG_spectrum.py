# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:29:48 2022

@author: kylab
"""


import numpy as np
import dataDecode_CH
import time
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime, timedelta 


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



def get_data_complement(input_signal):
    np_signal=np.array(input_signal)
    for i in range(len(np_signal)):
        if np_signal[i]<32768:
            np_signal[i]+=65535
    np_signal-=65535      
    return np_signal


fname='220512b.raw'    # filename
Raw_Data=open(fname, "rb").read()
Data,sampling_rate,timestr=dataDecode_CH.dataDecode.rawdataDecode(Raw_Data)

format='%Y-%m-%d %H:%M:%S'
data_duration=len(Data[0])/sampling_rate[0]
end_record_time=datetime.strptime(timestr, format)+timedelta(seconds = int(data_duration)) 


print(fname+" data was loaded")
print('start record time: ',timestr)
print('end record time: ',datetime.strftime(end_record_time,format))
print('Record time: ', str(timedelta(seconds = int(data_duration)) ))


dic_Config={"idx_EEG_1":0,"idx_EEG_2":1,"EEG_amp":2000,"resolution":65536}

len_epoch=2 # in second
EEG_cali=1.8/(dic_Config['resolution']*dic_Config['EEG_amp'])*1000000;

Signal_EEG_1=get_data_complement(Data[dic_Config['idx_EEG_1']])*EEG_cali
Signal_EEG_2=get_data_complement(Data[dic_Config['idx_EEG_2']])*EEG_cali
    
Signal_EEG_1=Signal_EEG_1[0:(len(Signal_EEG_1)//(len_epoch*int(sampling_rate[dic_Config['idx_EEG_1']]))*len_epoch*(len_epoch*int(sampling_rate[dic_Config['idx_EEG_1']])))]
Signal_EEG_2=Signal_EEG_2[0:(len(Signal_EEG_2)//(len_epoch*int(sampling_rate[dic_Config['idx_EEG_1']]))*len_epoch*(len_epoch*int(sampling_rate[dic_Config['idx_EEG_1']])))]


# df = pd.read_csv('Data/Baseline.csv')
# Signal_ECG = df['ECG']

EPSD1=[]
EPSD2=[]

power_40hz_EEG1=[]
power_alpha_EEG1=[]
power_low_alpha_EEG1=[]
power_middle_alpha_EEG1=[]
power_high_alpha_EEG1=[]
power_beta_EEG1=[]
power_low_beta_EEG1=[]
power_middle_beta_EEG1=[]
power_high_beta_EEG1=[]
power_theta_EEG1=[]
power_delta_EEG1=[]

power_40hz_EEG2=[]
power_alpha_EEG2=[]
power_low_alpha_EEG2=[]
power_middle_alpha_EEG2=[]
power_high_alpha_EEG2=[]
power_beta_EEG2=[]
power_low_beta_EEG2=[]
power_middle_beta_EEG2=[]
power_high_beta_EEG2=[]
power_theta_EEG2=[]
power_delta_EEG2=[]

EEG_40hz_index=[]
EEG_alpha_index=[]
EEG_low_alpha_index=[]
EEG_middle_alpha_index=[]
EEG_high_alpha_index=[]
EEG_beta_index=[]
EEG_low_beta_index=[]
EEG_middle_beta_index=[]
EEG_high_beta_index=[]
EEG_theta_index=[]
EEG_delta_index=[]




for i in range(int(len(Signal_EEG_1)//len_epoch)*2-1):
    if i ==0:
        slice_Signal_EEG_1=Signal_EEG_1[0:len_epoch*int(sampling_rate[dic_Config['idx_EEG_1']])]
        slice_Signal_EEG_2=Signal_EEG_2[0:len_epoch*int(sampling_rate[dic_Config['idx_EEG_2']])]
    else:

        slice_Signal_EEG_1=Signal_EEG_1[int(i/2*len_epoch*sampling_rate[dic_Config['idx_EEG_1']]):int((i/2+1)*len_epoch*sampling_rate[dic_Config['idx_EEG_1']])]
        slice_Signal_EEG_2=Signal_EEG_2[int(i/2*len_epoch*sampling_rate[dic_Config['idx_EEG_2']]):int((i/2+1)*len_epoch*sampling_rate[dic_Config['idx_EEG_2']])]
        if len(slice_Signal_EEG_1)<len_epoch*sampling_rate[dic_Config['idx_EEG_1']]:
            break
    Slice_EEG1_PSD,EPSD_x_f=fft_power(slice_Signal_EEG_1,sampling_rate[dic_Config['idx_EEG_1']],"hamming")    
    Slice_EEG2_PSD,MPSD_x_f=fft_power(slice_Signal_EEG_2,sampling_rate[dic_Config['idx_EEG_2']],"hamming")   

    if i ==0:
        EEG_40hz_index.append(np.where( (EPSD_x_f>39) & (EPSD_x_f<=41)))
        EEG_alpha_index.append(np.where( (EPSD_x_f>8) & (EPSD_x_f<=14)))
        EEG_low_alpha_index.append(np.where( (EPSD_x_f>8) & (EPSD_x_f<=9)))
        EEG_middle_alpha_index.append(np.where( (EPSD_x_f>9) & (EPSD_x_f<=12)))
        EEG_high_alpha_index.append(np.where( (EPSD_x_f>12) & (EPSD_x_f<=14)))
        EEG_beta_index.append(np.where( (EPSD_x_f>14) & (EPSD_x_f<=30)))
        EEG_low_beta_index.append(np.where( (EPSD_x_f>12.5) & (EPSD_x_f<=16)))
        EEG_middle_beta_index.append(np.where( (EPSD_x_f>16.5) & (EPSD_x_f<=20)))
        EEG_high_beta_index.append(np.where( (EPSD_x_f>20.5) & (EPSD_x_f<=28)))
        EEG_theta_index.append(np.where( (EPSD_x_f>4) & (EPSD_x_f<=8)))
        EEG_delta_index.append(np.where( (EPSD_x_f>0.5) & (EPSD_x_f<=4)))
        EEG_40hz_index=EEG_40hz_index[0][0].tolist()
        EEG_alpha_index=EEG_alpha_index[0][0].tolist()
        EEG_low_alpha_index=EEG_low_alpha_index[0][0].tolist()
        EEG_middle_alpha_index=EEG_middle_alpha_index[0][0].tolist()
        EEG_high_alpha_index=EEG_high_alpha_index[0][0].tolist()
        EEG_beta_index=EEG_beta_index[0][0].tolist()
        EEG_low_beta_index=EEG_low_beta_index[0][0].tolist()
        EEG_middle_beta_index=EEG_middle_beta_index[0][0].tolist()
        EEG_high_beta_index=EEG_high_beta_index[0][0].tolist()
        EEG_theta_index=EEG_theta_index[0][0].tolist()
        EEG_delta_index=EEG_delta_index[0][0].tolist()
        
    power_40hz_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_40hz_index]))
    power_alpha_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_alpha_index]))
    power_low_alpha_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_low_alpha_index]))
    power_middle_alpha_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_middle_alpha_index]))
    power_high_alpha_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_high_alpha_index]))
    power_beta_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_beta_index]))
    power_low_beta_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_low_beta_index]))
    power_middle_beta_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_middle_beta_index]))
    power_high_beta_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_high_beta_index]))
    power_theta_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_theta_index]))
    power_delta_EEG1.append(np.sum(Slice_EEG1_PSD[EEG_delta_index]))
    
    power_40hz_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_40hz_index]))
    power_alpha_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_alpha_index]))
    power_low_alpha_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_low_alpha_index]))
    power_middle_alpha_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_middle_alpha_index]))
    power_high_alpha_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_high_alpha_index]))
    power_beta_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_beta_index]))
    power_low_beta_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_low_beta_index]))
    power_middle_beta_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_middle_beta_index]))
    power_high_beta_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_high_beta_index]))
    power_theta_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_theta_index]))
    power_delta_EEG2.append(np.sum(Slice_EEG2_PSD[EEG_delta_index]))
    
    

    EPSD1.append(Slice_EEG1_PSD)
    EPSD2.append(Slice_EEG2_PSD)
    
x_PSD=np.linspace(1, len(EPSD1), len(EPSD1)) 
np_EPSD1=np.array(EPSD1)
np_EPSD1=np.log10(np_EPSD1.transpose())

np_EPSD2=np.array(EPSD2)
np_EPSD2=np.log10(np_EPSD2.transpose())
    #
fig1, axs1 = plt.subplots(nrows=4, sharex=True)
fig1.suptitle('EEG channel 1')
  

x_EEG=np.linspace(1/sampling_rate[dic_Config['idx_EEG_1']], len(Signal_EEG_1)/sampling_rate[dic_Config['idx_EEG_1']], len(Signal_EEG_1))
axs1[0].plot(x_EEG,Signal_EEG_1)
axs1[1].pcolormesh(x_PSD,EPSD_x_f,np_EPSD1,shading='auto',cmap='Greys')
axs1[2].plot(x_PSD,power_40hz_EEG1)
axs1[2].set_xlim([x_PSD[0], x_PSD[-1]])
axs1[3].plot(x_PSD,power_alpha_EEG1,'y')
axs1[3].plot(x_PSD,power_beta_EEG1,'r')
axs1[3].plot(x_PSD,power_theta_EEG1,'b')
axs1[3].plot(x_PSD,power_delta_EEG1,'g')

axs1[0].set_title('EEG Signal')
axs1[1].set_title('EEG PSD')
axs1[2].set_title('Sum of power of 40 Hz')
axs1[2].set_xlabel('time(s)')

fig2, axs2= plt.subplots(nrows=4, sharex=True)
fig2.suptitle('EEG channel 2')

x_EEG=np.linspace(1/sampling_rate[dic_Config['idx_EEG_2']], len(Signal_EEG_2)/sampling_rate[dic_Config['idx_EEG_2']], len(Signal_EEG_2))
axs2[0].plot(x_EEG,Signal_EEG_2)
axs2[1].pcolormesh(x_PSD,EPSD_x_f,np_EPSD2,shading='auto',cmap='Greys')
axs2[2].plot(x_PSD,power_40hz_EEG2)
axs2[2].set_xlim([x_PSD[0], x_PSD[-1]])
axs2[3].plot(x_PSD,power_alpha_EEG2,'y')
axs2[3].plot(x_PSD,power_beta_EEG2,'r')
axs2[3].plot(x_PSD,power_theta_EEG2,'b')
axs2[3].plot(x_PSD,power_delta_EEG2,'g')
axs2[0].set_title('EEG Signal')
axs2[1].set_title('EEG PSD')
axs2[2].set_title('Sum of power of 40 Hz')
axs2[2].set_xlabel('time(s)')

