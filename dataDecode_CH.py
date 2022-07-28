#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:15:13 2019
@author: chenghan
"""

import math
from datetime import datetime,timedelta

class dataDecode:
    def HeaderDataDecode(rawtxt):
        Raw_Data = rawtxt
        header = Raw_Data[0:512]             #RAW files header
        #RAW = RAW[512:len(RAW)]         #RAW data
        
        header2 = []                    #RAW files header to String
        for s in header:
            header2.append(chr(s))
        
        
        time=Raw_Data[320:324]

        timeformat='%Y-%m-%d %H:%M:%S'
        timestr=(datetime.strptime("2000-01-01 00:00:00", timeformat)+timedelta(seconds=int.from_bytes(time, byteorder='little'))).strftime(timeformat)
        #%% Acquisition sampling rate ratio
            
        splr = header2[39:54]   
        splr2 = ''.join(splr)
        splr2 = float(splr2)        #Sampling Rate
        
        start = 55
        SRn = []
        for i in range(header[36]):     #Acquisition sampling rate ratio SRn
            SRtemp = header2[start+i*15+i:start+(i+1)*15+i]
            splrtemp = ''.join(SRtemp)
            splr_float = float(splrtemp)
            SRn.append(splr2/splr_float)
        
        maxi = int(max(SRn))
        
        SR_array=[]
        for i in range(len(SRn)):
            SR_array.append(splr2/SRn[i])
            
        return SR_array, timestr
                    
        
    def rawdataDecode(rawtxt):
        '''
        input:
            -rawtxt: a list in binary read from kylab raw file format
        output:
            -Data: a list with channel recorded by kylab sensor
            -SR_array: a list with sampling rate of each channel
            -timestr: file established time, not available for TD1/3, rat & mice sensor
            if use TD1/3, rat & mice sensor, you can read file established time by datetime.fromtimestamp(os.path.getmtime(filename+'.RAW')) command
        '''
        Raw_Data = rawtxt

        
        header = Raw_Data[0:512]             #RAW files header
        #RAW = RAW[512:len(RAW)]         #RAW data
        
        header2 = []                    #RAW files header to String
        for s in header:
            header2.append(chr(s))
        
        
        time=Raw_Data[320:324]

        timeformat='%Y-%m-%d %H:%M:%S'
        timestr=(datetime.strptime("2000-01-01 00:00:00", timeformat)+timedelta(seconds=int.from_bytes(time, byteorder='little'))).strftime(timeformat)
        
        if header[12]==33:
            res=2 
        elif header[12]==35:
            res=1
        elif header[12]==40:
            res=3
        #%% Acquisition sampling rate ratio
            
        splr = header2[39:54]   
        splr2 = ''.join(splr)
        splr2 = float(splr2)        #Sampling Rate
        
        start = 55
        SRn = []
        for i in range(header[36]):     #Acquisition sampling rate ratio SRn
            SRtemp = header2[start+i*15+i:start+(i+1)*15+i]
            splrtemp = ''.join(SRtemp)
            splr_float = float(splrtemp)
            SRn.append(splr2/splr_float)
        
        maxi = int(max(SRn))
        
        SR_array=[]
        for i in range(len(SRn)):
            SR_array.append(splr2/SRn[i])
        
        #%%  Find the Channel 
        
        
        #matrix = np.zeros([maxi,header[36]])
        channel=[]
        for i in range(maxi) :
            for j in range(header[36]):
                #matrix[i][j] = i
                if i % SRn[j] == 0:
                    channel.append(j)
        
        #%% Data segmentation
        #Data = np.zeros([header[36],])
        Data = []
        cont = []
        for i in range(header[36]):
            Data.append([])
            cont.append(1)
                   
        Raw_Data = Raw_Data[512:math.floor((len(Raw_Data)-512)/len(channel))*len(channel)+512]
        
        if res==2:
            for i in range(0, len(Raw_Data), len(channel)*2):
                for j in range(len(channel)):
                    try:
                        Data[channel[j]].append(Raw_Data[i + 2*j+1]*256+Raw_Data[i + 2*j])
                    except IndexError:
                        pass
                    except:
                        print('unknown error')
        elif res==1:
            for i in range(0, len(Raw_Data), len(channel)):
                for j in range(len(channel)):
                        Data[channel[j]].append(Raw_Data[i+j])              
        elif res==3:
            try:
                for i in range(0, len(Raw_Data), len(SR_array)*2):
                    for j in range(len(SR_array)):
                        Data[j].append(Raw_Data[i + 2*j+1]*256+Raw_Data[i + 2*j])
            except:
                pass
                    
        return Data, SR_array, timestr