#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:10:34 2021

@author: weien
"""

'''紹芬的getRpeak程式'''
# import statistics
# import dogHRV
# import tkinter as tk  
# from tkinter import filedialog  
import math
import numpy as np


#-----detect R-peak-----#

def getYValueofRPeak(df,rawdatalist,parameter):
    HR_rollingmean=[]
    
    hrw = 0.3 #One-sided window size, as proportion of the sampling frequency
    fs = 250 #The example dataset was recorded at 100Hz
    mov_avg = rawdatalist.rolling(int(hrw*fs)).mean() #Calculate moving average
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    
    avg_hr = (np.mean(rawdatalist))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*parameter for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    df['HR_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of inerest
    window = []
    peaklist = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in rawdatalist:
        rollingmean = df.HR_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            #maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    ybeat = [rawdatalist[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
    
    x_time=range(len(rawdatalist))
    
    peaklist_time = [x_time[j] for j in peaklist]
    return ybeat,x_time,peaklist_time

