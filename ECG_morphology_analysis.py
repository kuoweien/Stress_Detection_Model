#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:38:03 2020

@author: chenghan
"""
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
import numpy as np
from io import BytesIO
import base64


class ECG_morphology_analysis:
    # def R_peak(self,start,inverse_end, f_ECG):
    def R_peak(self,start, end, f_ECG):
        peak=0
        peak_num=0
        peak_point_x=[]; peak_point_y=[]
        wave_range=[]
        peak_all = []
        peak_all_value=[]
        
        # end= len(f_ECG)+inverse_end				#計算到倒數10秒
        
        bs= statistics.mean(f_ECG[start:end])	#全段平均
        sd= statistics.stdev(f_ECG[start:end])	#標準差
        R_thr= bs+ 2*sd							#R點需超過的值
        
        for i in range(start,end):			   #10~ -10秒
            if (int(f_ECG[i])-int(f_ECG[i-20]))>10:                   	#20單位時間內,電壓上升超過10單位,視為進入陡坡, 陡坡最大者稱為peak #學長使用2560 根據data調整
                if (int(f_ECG[i])>=peak) and (f_ECG[i] > f_ECG[i-1]):
                    peak = f_ECG[i]										#之後進下個迴圈，i自動+1
                    peak_all.append(i)
                    peak_all_value.append(peak)
                    
                elif (int(f_ECG[i])< peak) and (f_ECG[i-1]>=f_ECG[i-2]) and (f_ECG[i-1]==peak) and peak >= R_thr: #會是至高點
                    peak_num+=1
                    peak_point_x.append(i-1)
                    peak_point_y.append(peak)						#R點座標=(i-1, peak)
                    wave_range.append([i-1-60,i-1+94])
                                                                    #以R_peak(i-1)為零分格，前120,後180 訂為wave_range #(前後值待討論修改!)
                    peak=0                    						#peak歸零,方便找下一個陡波
            else:
                peak=0                        						#雜訊太多時,不在條件內, 也要歸零
    
        return (peak_num, wave_range,  peak_point_x,peak_point_y ,peak_all, peak_all_value)

    def split_wave(self,wave_range, peak_num, f_ECG):							#依R點為基準,做分割
        wave_y_list=[]
        for j in range (0,peak_num):
           wave_y_list.append(f_ECG[ wave_range[j][0]:wave_range[j][1] ]) #每個wave的初值,終值為: (wave_range[j][0],wave_range[j][1])
            
        return np.array(wave_y_list)      								   #將list轉成array(numpy)

    def avg_wave(self,wave_y):
    
        wavesum = np.zeros((154,))    #創造一個'形狀'為300的 array,一開始裡面每個都是0
        for i in wave_y:          #此時的wave_y是def split wav回傳結果(np.array(wave_y_list) )
            wavesum = wavesum + i         #將wave_y中的每個項目加起來[[1],[2],[3]...} -> [1+2+3...]
        avg = wavesum/(len(wave_y))
        #plt.plot(avg,lw=3)       #畫修正偏差前的平均
        #plt.title('Average ECG')
        #plt.show()
        
    #    print(wave_y.shape); print(avg.shape); print(len(wave_y))
    
        num=0
        new_wave=[]
        for i in wave_y:
            total_dev= 0			#total_dev要在wave_y迴圈內歸零,不然會一直遞增上去
            for j in range(0,154):
                dev=(i[j]-avg[j])**2 
                total_dev+=dev		#算wave各點與avg差值的平方和
            num+=1
            if total_dev > 60000*256:   #過大偏差的之後印出,此處不納入
                continue
    			
            elif total_dev < 60000*256: #偏差不大的,列入new_wave  #(數值待討論)
                new_wave.append(i)
    
        
        include_wave=len(np.array(new_wave))
        
        new_wave_y = np.array(new_wave)
        wavesum= np.zeros((154,))			#設形狀為300的array,計算去除偏差後的平均波型
        for i in new_wave_y:
            wavesum = wavesum +i
        m_avg = wavesum/len(new_wave_y)
        return m_avg, include_wave



    def avg_sd_plot(self,m_avg, wave_y):
        	x_major_ticks=np.arange(0,301,100)			#x軸100單位=心電圖上一大格= 200ms
        	y_major_ticks=np.arange(-122.3,150,61.156)  #y軸上61.15單位=心電圖上一大格= 0.5mV
            
        	x_minor_ticks=np.arange(0,301,20)
        	y_minor_ticks=np.arange(-122.3,150,12.231111)
           
        	sd= np.std(wave_y, axis=0)   #算標準差
            
        	m_plus=np.zeros((300,))		 #設定300*1的array,方便運算
        	m_minus=np.zeros((300,))
        	m_plus=  m_avg+(sd)
        	m_minus= m_avg-(sd)
            
        	rc['figure.figsize'] = (10.0, 5.0)   #圖片大小
            
        	rc['grid.linestyle']= '--'
        	rc['grid.color']='red'
        	rc['grid.linewidth']= '0.3'
        	big= plt.gca()
        	big.set_xticks(x_major_ticks)  #設定圖中格線
        	big.set_yticks(y_major_ticks)
            
        	plt.plot(m_avg, lw=3)          #畫平均&標準差
        	plt.plot(m_plus, 'g:')
        	plt.plot(m_minus, 'g:')
        	plt.title('avg ECG (del dev) +/- 1sd', fontsize=18)
        	plt.ylim(-50,150)
        	big.grid()					   #畫出圖中格線
        	#plt.savefig('1_avg ECG.jpg')   #圖片存檔
        	#plt.clf()					   #清除資訊,免得下個圖畫進去(clean fig)
            
    
    def normal_R(self,peak_point_x, wave_y):
        wavesum = np.zeros((154,))    
        for i in wave_y:          
            wavesum = wavesum + i       
        avg = wavesum/(len(wave_y))
        num=0
        del_list=[]
        for i in wave_y:
            total_dev= 0
            for j in range(0,154):
                dev=(i[j]-avg[j])**2
                
               # print(dev)
                total_dev+=dev
            num+=1
            if total_dev > 60000*256*256*256:   
                del_list.append(num-1)
        normal_R= [x for x in peak_point_x if peak_point_x.index(x) not in del_list]  ##新招 學起來!!
        #print(normal_R)
        #del_list是要刪除的序列[2,5,8...], 不在del_list中的index, 就保留為normal_R
        
    	#print(len(peak_point_x)); print(len(normal_R)); print(normal_R)
        
        #for i,x in enumerate(normal_R):
            #print(i,x)
        
        HR=[]
        for i in range(1,len(normal_R)):
            RR= (normal_R[i]-normal_R[i-1]) / 256
    #        print(RR)
            if int(60/RR)> 20 and int(60/RR) <200:
                    HR.append(60/ RR)
        
        HR_avg= statistics.mean(HR)
        RR= 60 / HR_avg
        
        return   RR

    def PQRST_anl(self,m_avg):
    # R    
        Ry= (m_avg[60]) 
    
    # P                        #選0~100最大值為p點y值
        list_p= m_avg[0:50]   #將array轉成list,才能算max
        py= max(list_p) 
        px=[]
        for i in range(0,50):
            if (py - m_avg[i]) == 0:
                px.append(i)   #找px
        px= px[0]
    
    # T                        #選140~199最大值為t點y值
        list_t= m_avg[70:154] #將array轉成list,才能算max
        ty= max(list_t)
        tx=[]
        for i in range(70,154):
            if (ty- m_avg[i] ) == 0:
                tx.append(i)
        tx=tx[0]
     
    #Q_new
        slope=[]									 #Q,S點不一定是minimum,因此想到以下方法來抓Q,S點(概念: m[i+8]-m[i]/m[i]-m[i-3] ),再加條件修正
        q=[]
        for i in range(px+5,60):
            if abs(m_avg[i] - m_avg[i-2]) <= 1:      #m[i]-m[i-3]升幅太小or為負的,直接視為1,相除才不會出問題
                a= 1
                s= ( abs(m_avg[i+4]-m_avg[i]) / a )
                slope.append(s)
            elif abs(m_avg[i] - m_avg[i-2]) > 1:     #紀錄前後升幅,8單位時間內上升(m[i+8]-m[i])最大的起點訂為Q點(同時除以前3單位時間變化,才能找到轉折點)
                b= abs(m_avg[i] - m_avg[i-2])
                s = ( abs(m_avg[i+4]-m_avg[i]) / b )
                slope.append(s)
        maxi= max(slope)                             #找到最大斜率
    #    print(slope, 'max_Q', maxi)
        for i in range(px+5,60):
            if abs(m_avg[i] - m_avg[i-2]) <= 1:
                a= 1
                s= ( abs(m_avg[i+4]-m_avg[i]) / a )
                slope.append(s)
            elif abs(m_avg[i] - m_avg[i-2]) > 1:
                b= abs(m_avg[i] - m_avg[i-2])
                s = ( abs(m_avg[i+4]-m_avg[i]) / b )
                slope.append(s)
            if s == maxi:                           #找與最大斜率相對應的s點
                q.append(i)
        qx= q[0]; qy= m_avg[qx]
        #print(qx)
    
        
    #S_new
        slope=[]                                    #與Q點概念相仿,只是正負顛倒
        ss=[]
        for i in range(60,100):
            if abs(m_avg[i+2] - m_avg[i]) <= 1:
                a= 1
                s= ( abs(m_avg[i]-m_avg[i-4]) / a )
                slope.append(s)
            elif abs(m_avg[i+2] - m_avg[i]):
                b= abs(m_avg[i+2] - m_avg[i])
                s = ( abs(m_avg[i]-m_avg[i-2]) / b )
                slope.append(s)
        maxi= max(slope)
        
        ss.append(slope.index(maxi)+60)
    #    print(slope, 'max_S', maxi)
    #    for i in range(60,100):
    #        if abs(m_avg[i+2] - m_avg[i]) <= 1:
    #            a= 1
    #            s= ( abs(m_avg[i]-m_avg[i-4]) / a )
    #            slope.append(s)
    #        elif abs(m_avg[i+2] - m_avg[i]) :
    #            b= abs(m_avg[i+2] - m_avg[i])
    #            s = ( abs(m_avg[i]-m_avg[i-4]) / b )
    #            slope.append(s)
    ##        print (str(s)+' '+str(maxi))
    #        if s == maxi:
    #            ss.append(i)
                
        #print(ss)
        sx= ss[0]; sy= m_avg[sx]
    
        
    #波型細節
     
        int_m=[]
        for i in range(0,154):
            int_m.append(round(m_avg[i]))     #將波型數值轉為整數,方便看觀察數值分布
        bs= statistics.median(int_m)		  #找中位數(就相當於眾數)作為baseline,不找眾數原因:若有兩個眾數則無法判斷
        
        #訂P波起點
        int_m_0p=[]
        for i in range(0,px):
            int_m_0p.append(round(m_avg[i]))
        bs_0p= statistics.median(int_m_0p)    #bs_0p為起點至P點的baseline
        ps=[]
        for i in range(0,px):
            if abs(round(m_avg[i]) - bs_0p ) <= 256:      
                ps.append(i)
        psx= ps[-1]; psy= (m_avg[psx])        #屬於baseline的最後一個點(離開baseline的第一個點),為P波起點
        
        #訂P波終點,Q波起點
        int_m_pq=[]
        for i in range(px,qx):
            int_m_pq.append(round(m_avg[i]))
        bs_pq= statistics.median(int_m_pq)
        
        pe=[]
        for i in range (px,qx):
            #print(abs(round(m_avg[i] - bs_pq)))
            if  abs(round(m_avg[i] - bs_pq)) <=256:
                pe.append(i)
                
        #print(str(px)+' '+str(ps)+' '+str(qx)+' '+str(pe))
        pex=pe[0]; pey= (m_avg[pex])        #從P波進入baseline的第一個點,訂為P波終點
        qsx=pe[-1]; qsy=(m_avg[qsx])		#屬於baseline的最後一點, 訂為Q波起點
    
        
        #訂T波起點終點
        ts=[]; te=[]
        
        if (qy- bs_pq) <= 0:                        #典型狀況:Q點低於baseline
    												#為何用P至Q的baseline:定義,ST segment有無elevation or depression,是以PQ段的baseline為準
            for i in range(sx,tx):
                if abs(round (m_avg[i]-bs_pq) ) <=256:
                    ts.append(i)
            tsx = ts[-1]; tsy = m_avg[tsx]          #S,T段中屬於baseline的最後一個點(脫離basleine的第一個點),訂為T波起點
         #   print(tx)
            for i in range(tx,154):
              #  print(str(tx)+' '+str(m_avg[i])+' '+str(bs_pq))
                if abs(round (m_avg[i]-bs_pq) ) <=1024:
                    te.append(i)
            tex= te[0]; tey= m_avg[tex]             #T至結束中,第一個進入baseline的點,訂為T波終點
    
        
    
        mean_st = statistics.mean(m_avg[sx:tx])
    
        if (qy- bs_pq) > 0:                        #非典型狀況:Q點高於baseline
            for i in range(sx,tx):
                if abs(round (m_avg[i]- mean_st)) <=256:
                    ts.append(i)
            tsx = ts[-1]; tsy = m_avg[tsx]
            print(tx)
            for i in range(tx,154):
                #print(str(tx)+' '+str(m_avg[i])+' '+str(mean_st))
                if abs(round (m_avg[i]- mean_st)) <=256:
                    te.append(i)
            tex= te[0]; tey= m_avg[tex]
        
        return Ry, px,py, tx,ty, qx,qy, sx,sy,   psx,psy, pex,pey, qsx,qsy,  tsx,tsy, tex,tey #, qex,qey

    def avg_point_plot(self,m_avg, Ry, px,py, tx,ty, qx,qy, sx,sy, psx,psy, pex,pey, qsx,qsy, tsx,tsy, tex,tey): #, qex,qey
    
    
        ecgnp=np.array(m_avg)
        ecgnp=ecgnp*(1000* (1.8/(65536*860)))
        
        xsecg = np.linspace(0, len(m_avg)/256, len(m_avg))
    
        x_major_ticks=np.arange(0,0.8,0.2)							#同AVG,設定格線大小
        y_major_ticks=np.arange(-0.5,1,0.5)                       #y軸上61.15單位=心電圖上一大格= 0.5mV
    
        
        rc['grid.color']='red'; rc['grid.linestyle']='--'; rc['grid.linewidth']='0.3'
        big= plt.gca()
        big.set_xticks(x_major_ticks)
        big.set_yticks(y_major_ticks)
        
        rc['figure.figsize'] = (10.0, 5.0)
        plt.plot(xsecg,ecgnp, lw=2, color='blue')
        
        plt.scatter(xsecg[60], ecgnp[60], color='red')			#點出PQRST(紅色)
        plt.scatter(xsecg[px],py*(1000* (1.8/(65536*860))), color='red')
        plt.scatter(xsecg[tx],ty*(1000* (1.8/(65536*860))), color='red')
        plt.scatter(xsecg[qx],qy*(1000* (1.8/(65536*860))), color='red')
        plt.scatter(xsecg[sx],sy*(1000* (1.8/(65536*860))), color='red')
        
        plt.scatter(xsecg[psx],psy*(1000* (1.8/(65536*860))), color='cyan'); plt.scatter(xsecg[pex],pey*(1000* (1.8/(65536*860))), color='cyan')      #點出P波,T波起點終點(cyan,magenta色)
        plt.scatter(xsecg[tsx],tsy*(1000* (1.8/(65536*860))), color='magenta');plt.scatter(xsecg[tex],tey*(1000* (1.8/(65536*860))), color='magenta')
        
        plt.title('avg ECG (del dev)', fontsize=18)
        plt.xlim(0,0.6)
        plt.ylim(-0.5,0.5)
        big.grid()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)  # rewind to beginning of file
        figdata_png = base64.b64encode(buffer.getvalue())
        return figdata_png
    
    
    def Ana(self,m,qx,psx,sx,Ry,py,qy,sy,ty,tex,RR,qsx):
        	int_m=[]
        	for i in range(0,154):
        		int_m.append(int(m[i]))
        	bs= statistics.median(int_m)				#設定baseline,voltage以此為標準
        
        
        	PR_i= int( (( qx - psx )/256 ) *1000 )      #PR interval= P波起點~Q波
        	QRS = int( ((sx -qx)/256) *1000      )		#QRS duration=Q波~S波	
        	R   = ( (Ry- bs) * (1000* (1.8/(65536*860)))  )
        	P   = ( (py- bs) * (1000* (1.8/(65536*860)) ) )
        	Q   = ( (qy- bs) * (1000* (1.8/(65536*860)) ) )
        	S   = ( (sy- bs) * (1000* (1.8/(65536*860)) ) )
        	T   = ( (ty- bs) * (1000* (1.8/(65536*860)) ) )
        	QT  = int( ((tex-qx)/256)* 1000 )           #QT interval= Q波~T波終點
        	QTc = int( ((tex-qx)/256)* 1000 / (RR**(1/2)) ) #標準化QT
        	RR  =  int(RR*1000)
        	Qwave_dura= ((qx-qsx)/256)*1000
        	'''
        	print('\n','PR_interval (ms)',PR_i,
              '\n','QRS duration(ms)',QRS,
              '\n','RR interval (ms)',RR,
              '\n','QT          (ms)',QT,
              '\n','QTc         (ms)',QTc,
              '\n','R_voltage(mV): ',"%.2f" %R, 
              '\n','P wave (mV):   ',"%.2f" %P,
              '\n','T wave (mV):   ',"%.2f" %T,
              '\n', 'Q wave (mV):   ',"%.2f" %Q,
              '\n', 'S wave (mV):   ',"%.2f" %S)
            '''	
        	text3=""
        	text3= '\n'+ '[Analysis]'+ '\n'+'PR_interval (ms)'+str(PR_i)+'\n'+  'QRS duration(ms)'+str(QRS) +'\n'+  'RR interval (ms)'+str(RR)  +'\n'+  'QT          (ms)'+str(QT)  +'\n'+  'QTc         (ms)'+str(QTc)+'\n'+ 'R_voltage(mV): '  +str("%.2f" %R)+ '\n'+'P wave (mV):   '+str("%.2f" %P)+'\n'+'T wave (mV):   '+str("%.2f" %T)+'\n'+ 'Q wave (mV):   '+str("%.2f" %Q)+'\n'+ 'S wave (mV):   '+str("%.2f" %S)
        	#就是把print的那一串寫入text
        	
        	return PR_i, QRS, R, Q, QTc, Qwave_dura, S, text3,P,T,RR,QT


    def Ana_problem(self,PR_i, QRS, R, Q, QTc, Qwave_dura, S, qx,sx, tsx, m):    
        #check LVH
        	text3=''
        	if R >= 1.4:
        		a= 'high R voltage, suspected LVH'
        	else:
        		a= 'LVH(-)'
        
        	text4=""
        	text4= a
        
        #check RVH(+/-)
        	if abs(Q)>= R:
        		b= 'suspected RVH or post.LBBB'
        	else:
        		b='RVH(-)'
        
        	text4+= '\n'+ b
        	
        #check ST段
        	mean_st = statistics.mean(m[sx:tsx])
        	ss=[]
        	for i in range(0,qx):
        		ss.append(round(m[i]))
        	bs_pq   = statistics.median(ss)
        #print('mean_ST:',mean_st,'bs:', bs_pq)
        
        	ST= (mean_st - bs_pq)
        	if ST >= 12:
        		c='probable ST_elevation, suspected STEMI or pericarditis'
        	elif ST <= (-6):
        		c='probable ST_depression, suspected NSTEMI or measured error'
        	else:
        		c='normal ST segment'
        	
        	text4+= '\n'+ c
            
        
        #check 深Q波    
        	if ( abs(Q) >= 1/3*R ) and  (  Qwave_dura > 40 ):
        		d='deep Q wave, suspected AMI'
        	else:
        		d='no deep Q wave'
        		
        	text4+= '\n'+d
        	
        #check 1st AV block
        	if PR_i > 200:
        		e='long PR interval, suspected 1st AV block'
        	else:
        		e='1st AV block(-)'
        
        	text4+= '\n'+e
        	
        #check VA, BBB
        	if QRS >= 120:
        		f='long QRS, suspected VA or BBB'
        	else:
        		f='normal QRS duration'
        	text4+= '\n'+f
        	
        #check PE
        	if abs(S) >0.5:
        		g='deep S wave, suspected pulmonary embolism, check Q wave on lead3 and d-dimer'
        	else:
        		g='no deep S wave, pulmonary embolism(-)'
        		
        	text4+= '\n'+g
        	
        	if QTc >=440:
        		h='prolong QTc'
        	else:
        		h='normal QTc'
        
        	text4+= '\n'+ h
        	text= text4
        	return text