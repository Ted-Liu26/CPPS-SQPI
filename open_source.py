import time
import os
import gc
from time import sleep
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from CSPIFuntions import WHRecoverPMDN
from CSPIFuntions import WHRecoverAngRnoise

import csv

import multiprocessing as mp
from multiprocessing import Queue,Event,Process

from scipy.linalg import hadamard
from ctypes import cast, POINTER, c_double

import sys
        
channel_checkboxs = [
    '-channel0-',
    '-channel1-',
    '-channel2-',
    '-channel3-',
    '-channel4-',
    '-channel5-',
    '-channel6-',
    '-channel7-',
]
# 列表按钮状态

# mat图表
mat_data_start = 0  # 绘图起始行，图表相关
mat_data_end = None  # 绘图终止行，图表相关

# mat图表_通道图例
mat_CHs = [
    'CH0',
    'CH1',
    'CH2',
    'CH3',
    'CH4',
    'CH5',
    'CH6',
    'CH7',
]

# mat图表_采样数值
mat_CH_values = [
    'value0',
    'value1',
    'value2',
    'value3',
    'value4',
    'value5',
    'value6',
    'value7',
]

# mat图表 通道颜色
mat_CH_colours = [
    'steelblue',
    'orange',
    'orchid',
    'slateblue',
    'slategrey',
    'c',
    'aquamarine',
    'lightgreen',
]

read_file_code = "out_put_production_2026_04_20_0.5_40_xue_20fps_fs_2"
N = 32  # 设定重构图像大小为N*N
N2 = N-1
tnum = N*N
image_code ='2026_04_20_04'
save_address = "SPI-images-3step/"
im_P =256 # 设定投影图像大小，应为N的整数倍
rho = 0  # 设定欠采样率，全采样取值为0
group_lenth =128

fb = 5  # 图像重建去除噪声截止频率
frame_shift = 2 #帧率 = 60/(frame_shift + 1)    
show_figure = False
# 计算像素binning数目
bin_num = im_P // N
parameter_1 = 1

#pattern_queue = mp.Queue()
#group_index_queue = mp.Queue()
#data_queue = mp.Queue()
#result_queue = mp.Queue()
event_queue = mp.Queue()
#replay_queue = mp.Queue()
#data_sync_queue = mp.Queue()
imaging_queue = mp.Queue()


###############################################################################
# 1. Walsh Hadamard matrix
###############################################################################

def WHadamard(n):
    if n & (n - 1) != 0:
        print(f"The order must be a power of 2!")
        return []

    ind = [0] * n
    N = len(bin(n)) - 3

    for i in range(n):
        g = i ^ (i >> 1)
        g = bin(g)[2:]
        g = (N - len(g)) * "0" + g
        gr = g[-1::-1]
        ind[i] = int(gr, 2)

    H = hadamard(n)
    return H[ind, :]

#主进程4-1 数据整理及成像
class SPI_data_pool():
    def __init__(self):
        self.RGB_list_total=[]
        self.green_list_total=[]
        self.red_list_total=[]
        self.blue_list_total_1=[]
        self.blue_list_total_2=[]
        self.RGB_list_replay=[]
        self.green_list_replay=[]
        self.red_list_replay=[]
        self.blue_list_replay_1=[]
        self.blue_list_replay_2=[]
        self.rec_image_g =[]
        self.rec_image_r =[]
        self.rec_image_b =[]

        self.imaging_list_total_0 =[]
        self.imaging_list_total_1 =[] ###
        self.imaging_list_total_2 =[]

        self.imaging_list_replay_0 =[]
        self.imaging_list_replay_1 =[]
        self.imaging_list_replay_2 =[]

    def SPI_3PM(self):
        N2 = N-1

        delta_phi = 0.9  # 设定坐标系旋转角度为 delta_phi*pi
        manification = 29  # 设定光学放大率
        
        imaging_list_array_0 = np.concatenate(self.imaging_list_total_0)
        imaging_list_array_1 = np.concatenate(self.imaging_list_total_1)
        imaging_list_array_2 = np.concatenate(self.imaging_list_total_2)

        df_imaging_list_array_0 = pd.DataFrame(imaging_list_array_0,columns=['value_0'])
        df_imaging_list_array_1 = pd.DataFrame(imaging_list_array_1,columns=['value_0'])
        df_imaging_list_array_2 = pd.DataFrame(imaging_list_array_2,columns=['value_0'])    

        df_imaging_list_array_0.to_csv('df_df_3PM_imaging_list_array_0_{}_{}.csv'.format(N,image_code), index_label='order') 
        df_imaging_list_array_1.to_csv('df_df_3PM_imaging_list_array_1_{}_{}.csv'.format(N,image_code), index_label='order') 
        df_imaging_list_array_2.to_csv('df_df_3PM_imaging_list_array_2_{}_{}.csv'.format(N,image_code), index_label='order') 

        # real part of Spectrum
        mdata_re = ((2*imaging_list_array_0[0:]) - 1*imaging_list_array_1[0:] - 1*imaging_list_array_2[0:])/(3*parameter_1)
        mdata_re = mdata_re.reshape(N, N)
        # imag part of Spectrum
        mdata_im = (1*imaging_list_array_1[0:] - 1*imaging_list_array_2[0:])/((3**0.5)*parameter_1)
        mdata_im = mdata_im.reshape(N, N)

        print('!!!!!!!!!!!!!!!!!!!mdata_re.shape[0]',mdata_re.shape[0])
        print('!!!!!!!!!!!!!!!!!!!mdata_re.shape[1]',mdata_re.shape[1])

        rec_image_re = WHRecoverPMDN(mdata_re, N, fb)
        rec_image_im = WHRecoverPMDN(mdata_im, N, fb)

        #########################################################
        ## rotation reconstruction coordinate system
        #########################################################
        delta_phase = np.exp(1j * np.pi * delta_phi)
        rec_image_complex = (rec_image_re + 1j * rec_image_im) * delta_phase

        rec_image_re_rot = np.real(rec_image_complex)
        rec_image_im_rot = np.imag(rec_image_complex)

        rec_image_ABS = np.abs(rec_image_complex)
        rec_image_Angle = np.angle(rec_image_complex)

        rec_image_Angle = WHRecoverAngRnoise(rec_image_Angle, N, fb)

        # display recover image
        fig_red = plt.figure()
        plt.imshow(rec_image_re_rot)
        plt.axis('off')
        plt.title('60fps_green')

        fig_red.savefig("%sSPI_%dx%d_%s_re.png" %(save_address,N,N,image_code), facecolor='black',bbox_inches='tight',dpi=100)
        sleep(0.1)
        plt.show()
        plt.close()

        #print('!!!!!!!!!!!!!test_3')    
        # display recover image
        fig_green = plt.figure()
        plt.imshow(rec_image_im_rot)
        plt.axis('off')
        plt.title('60fps_green')

        fig_green.savefig("%sSPI_%dx%d_%s_im.png" %(save_address,N,N,image_code), facecolor='black',bbox_inches='tight',dpi=100)
        sleep(0.1)
        plt.show()
        plt.close()

        #print('!!!!!!!!!!!!!test_4')   
        # display recover image
        fig_green = plt.figure()
        plt.imshow(rec_image_ABS)
        plt.axis('off')
        plt.title('60fps_green')

        fig_green.savefig("%sSPI_%dx%d_%s_ABS.png" %(save_address,N,N,image_code), facecolor='black',bbox_inches='tight',dpi=100)
        sleep(0.1)
        plt.show()
        plt.close()

        #print('!!!!!!!!!!!!!test_5')   
        # display recover image
        fig_red = plt.figure()
        plt.imshow(rec_image_Angle)
        plt.axis('off')
        plt.title('60fps_green')

        fig_red.savefig("%sSPI_%dx%d_%s_Angle.png" %(save_address,N,N,image_code), facecolor='black',bbox_inches='tight',dpi=100)
        sleep(0.1)
        plt.show()
        plt.close()

        sys.exit()


#子进程3-1 数据分析
def data_analysis_RGB(imaging_queue,event_queue):
    while True:
        start_flag_and_phase = event_queue.get()
        start_flag = start_flag_and_phase[0]
        current_display_phase = start_flag_and_phase[1]
        print('start_flag',start_flag)
        imaging_list_total_0 = []
        imaging_list_total_1 = []
        imaging_list_total_2 = []

        replay = False        
        false_list = []
        if current_display_phase in [1,2]:
            for i in start_flag: 
                
                read_file_name = "%s_%s.npy" %(read_file_code,i)
                print("read_file_name",read_file_name)
                cdata = pd.DataFrame(np.load(read_file_name),columns=["value0", "value1", "value2", "value3"])

                print('结果已接收：',len(cdata))

                cdata['value3_rolling'] = cdata['value3'].rolling(100).mean().shift(-45)
                signal_max = cdata['value3_rolling'].max()
                signal_min = cdata['value3_rolling'].min()
                print('signal_max',signal_max)
                print('signal_min',signal_min)
                
                signal_diff_1 = signal_min + (signal_max - signal_min)*0.40
                signal_diff_2 = signal_min + (signal_max - signal_min)*0.40
                cdata['signal_diff_1'] = signal_diff_1
                cdata['signal_diff_2'] = signal_diff_2

                cdata['is_stable']=False
                cdata['signal_shift']=False

                cdata.loc[((cdata['value3_rolling']>=signal_diff_1)&(cdata['value3_rolling'].shift(-1)<signal_diff_1)),'signal_shift'] = True
                cdata.loc[((cdata['value3_rolling'].shift(-50)<=signal_diff_2)&(cdata['value3_rolling'].shift(-51)>signal_diff_2)),'signal_shift'] = True

                cdata.loc[cdata['signal_shift'] == True,'signal_shift'] = 1
                cdata.loc[cdata['signal_shift'] == False,'signal_shift'] = 0
                cdata['signal_shift_count'] = cdata['signal_shift'].cumsum()
                cdata['signal_shift_count'] = cdata['signal_shift_count'].shift(-1).ffill()

                shift_revise = 0    
                for k in range(750,1050):
                    cdata.loc[(cdata['signal_shift'] == 1).shift(k+shift_revise,fill_value=False),'is_stable'] = True

                diff_count = cdata['signal_shift'].dropna().sum()

                print('>>>>>>>>>>>>>>>>>>>>>>>>diff_count',diff_count)

                print('signal_max',signal_max)
                print('signal_min',signal_min)
    
                cdata.to_csv('out_put_production_{}_fs_{}_{}.csv'.format(image_code,frame_shift,i), index_label='order')

                if (i==0) & (show_figure == True):
                    plt.figure(figsize=(10, 5))
                    plt.title('Scan result')
                    plt.xlabel('sequence')
                    plt.ylabel('volt (V)')
                    
                    plt.plot(cdata.index, cdata['value0'],
                            label=['ch0'], c=mat_CH_colours[0])
                    plt.plot(cdata.index, cdata['value1'],
                            label=['ch1'], c=mat_CH_colours[1])
                    plt.plot(cdata.index, cdata['value2'],
                            label=['ch2'], c=mat_CH_colours[2])
                    plt.plot(cdata.index, cdata['value3'],
                            label=['ch3'], c=mat_CH_colours[3])
                    plt.plot(cdata.index, cdata['value3_rolling'],
                            label=['value3_rolling'], c=mat_CH_colours[5])
                    plt.plot(cdata.index, cdata['signal_diff_1'],
                            label=['signal_diff_1'], c=mat_CH_colours[1])
                    plt.plot(cdata.index, cdata['signal_diff_2'],
                            label=['signal_diff_2'], c=mat_CH_colours[0])
                    plt.plot(cdata.index, cdata['signal_shift_count'],label=['signal_shift_count'], c=mat_CH_colours[6])
                    plt.plot(cdata.index, cdata['is_stable'],label=['is_stable'], c=mat_CH_colours[7])
                    plt.legend(loc=0)
                    plt.show()

                    print(len(cdata))
                
                c_index=cdata.loc[(cdata['signal_shift_count'] == (group_lenth+2))&(cdata['signal_shift'] == 1)].index[0]
                cdata.loc[c_index+1001:,'signal_shift_count']=None
                
                imaging_group_0 = cdata[(cdata['is_stable'] == True)].groupby('signal_shift_count')['value0'].mean().reset_index()
                imaging_list_0 = imaging_group_0['value0'].values
                imaging_list_0 = imaging_list_0[2:]
                imaging_list_total_0.append(imaging_list_0)

                imaging_group_1 = cdata[(cdata['is_stable'] == True)].groupby('signal_shift_count')['value1'].mean().reset_index()
                imaging_list_1 = imaging_group_1['value1'].values
                imaging_list_1 = imaging_list_1[2:]
                imaging_list_total_1.append(imaging_list_1)

                imaging_group_2 = cdata[(cdata['is_stable'] == True)].groupby('signal_shift_count')['value2'].mean().reset_index()
                imaging_list_2 = imaging_group_2['value2'].values
                imaging_list_2 = imaging_list_2[2:]
                imaging_list_total_2.append(imaging_list_2)
                #print('imaging_list',imaging_list_0)
                print('imaging_group_0',len(imaging_group_0))
                #print('imaging_group_0',imaging_group_0)
                print('imaging_group_1',len(imaging_group_1))
                #print('imaging_group_1',imaging_group_1)
                print('imaging_group_2',len(imaging_group_2))
                #print('imaging_group_2',imaging_group_2)

                print('imaging_list_total',len(imaging_list_total_0))
                print('==============================================')
                print("%d/%d" %(i+1,N*N//128))
                print('==============================================')
                RGB_list_total = [imaging_list_total_0,imaging_list_total_1,imaging_list_total_2]

            if len(false_list)!=0:
                replay = True
                # event_queue.put(replay)
                # replay_queue.put(false_list)
                # imaging_queue.put(RGB_list_total)
            
            else:
                # replay_queue.put(false_list)
                imaging_queue.put(RGB_list_total)
                print('准备生成图像 false_list',false_list)

def main():

    #process_1 = mp.Process(target=pattern_preload_p1_g1, args=(group_lenth,pattern_queue,group_index_queue))
    #process_2 = mp.Process(target=data_collection_RGB, args=(data_queue,data_sync_queue,result_queue,event_queue))
    process_3 = mp.Process(target=data_analysis_RGB, args=(imaging_queue,event_queue))

    #phase_queue.put(1)
    #process_1.start()
    #process_2.start()
    process_3.start()

    finish_flag = int((N*N)/group_lenth) #2st
    display_list=[]
    for i in range(finish_flag):
        display_list.append(i)
    print('display_list',display_list)
    display_list_and_phase=[display_list,1,N,im_P]
    print('display_list_and_phase',display_list_and_phase)
    event_queue.put(display_list_and_phase)

    spi_data_pool_1.RGB_list_total=imaging_queue.get()
    spi_data_pool_1.imaging_list_total_0=spi_data_pool_1.RGB_list_total[0]
    spi_data_pool_1.imaging_list_total_1=spi_data_pool_1.RGB_list_total[1]
    spi_data_pool_1.imaging_list_total_2=spi_data_pool_1.RGB_list_total[2]
    spi_data_pool_1.SPI_3PM()

if __name__ == '__main__':

    spi_data_pool_1 = SPI_data_pool()
    next_group_ready = False
    pattern_group = 0
    pattern_group_bool = False
    imaging_phase = 0
    data_sync_flag=0
    main()
