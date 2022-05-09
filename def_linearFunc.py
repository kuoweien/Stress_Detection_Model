#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:28:37 2022

@author: weien
"""

#線性函式 且x刻度為1
def linearFunc(start,end):
    output=[]
    
    a=(end[1]-start[1])/(end[0]-start[0])
    b=start[1]-a*start[0]
    
    for i in range(start[0]+1,end[0]):
        y=a*i+b
        tempoutput=[i,y]
        output.append(tempoutput)
    
    return output

a=linearFunc([3,2],[7,4])
    