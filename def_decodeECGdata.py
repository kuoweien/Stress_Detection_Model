#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:43:19 2022

@author: weien
"""
import dataDecode
import datetime

def openRawFile(filename):
    
    with open(filename,'rb') as f:
        rawtxt=f.read()
    rawlist=dataDecode.dataDecode.rawdataDecode(rawtxt)
    
    rawdata=rawlist[0]
    ecg_rawdata=rawdata[0]#取原始資料channel1 (234是加速度)
    
    frquency=rawlist[1]
    ecg_fq=frquency[0]#頻率 狗狗的是250Hz
    
    updatetime_str=rawlist[2].split(' ')[1]#抓上傳時間
    update_datetime = datetime.datetime.strptime(updatetime_str, '%H:%M:%S')#上傳時間字串轉datetime type

    return ecg_rawdata,ecg_fq,update_datetime