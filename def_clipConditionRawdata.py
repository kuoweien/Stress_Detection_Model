#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:55:49 2022

@author: weien
"""
import datetime

#輸入時間，取出時間斷之rawdata
def getConditionRawdata(start_time,end_time, ecgrawlist,frequency,updatetime):
    
    start_datetime = datetime.datetime.strptime(start_time, '%H:%M:%S')
    end_datetime = datetime.datetime.strptime(end_time, '%H:%M:%S')
    

    start_time_index = start_datetime-updatetime
    end_time_index = end_datetime-updatetime
    
    start_index = start_time_index.total_seconds()
    end_index = end_time_index.total_seconds()
        
    rawdata_list = ecgrawlist[int(start_index*frequency):int(end_index*frequency)]
    
    return rawdata_list