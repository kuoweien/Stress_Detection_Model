#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:30:48 2022

@author: weien
"""
from scipy import signal
import matplotlib.pyplot as plt


def lowPassFilter(fq,data):
    b, a = signal.butter(8, (2*fq)/250, 'lowpass')   #濾除 fq HZ以上的頻率
    data_lowfilter = signal.filtfilt(b, a, data) 
    return data_lowfilter
    

def highPassFilter(fq,data):
    b, a = signal.butter(8, (2*fq)/250, 'highpass') #濾除 fq HZ以下的頻率
    data_highfilter = signal.filtfilt(b, a, data)
    return data_highfilter