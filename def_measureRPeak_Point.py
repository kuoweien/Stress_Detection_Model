#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:42:56 2022

@author: weien
"""
#給每個R點分數

import numpy as np



def measureRPoint(rr_index,ecg_raw):
    output = []
    for i in range(len(rr_index)):        
        
        if i == len(rr_index)-1:  
            
            y = ecg_raw[rr_index[i]:]
            
        else:
            y = ecg_raw[rr_index[i]:rr_index[i+1]]
            
        rms = np.sqrt(np.mean(np.array(y)**2))
        output.append(round(rms,3)) #新增R點的index值 以及計算到下一個R點前的RMS
          
    return output #印出每個點的RMS
    

# rr_index = [2,5,8,10]
# ecg_raw = [1,2,3,4,5,6,7,8,9,10,11,12,13]

# rms = measureRPoint(rr_index, ecg_raw)

# zip_rrscore = list(zip(rr_index,rms))
