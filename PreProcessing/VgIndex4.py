'''
该代码用于将得到的三波段反射率计算各植被指数特征值
'''
import os
import cmath
import itertools
import numpy as np
import pandas as pd
import pickle
import time
import os # os分装了常见的系统调用
from openpyxl import load_workbook
from multiprocessing import Process, Lock, Queue,Pool,Manager
import threading



# 多进程计算各列表的均值并写入到queue
def sub(input_xlsx_name, sheet_name):
    print("getin:", os.getpid()) # 子进程调用getppid()得到父进程的ID getpid()得到当前进程的ID
    start_time = time.time()
    _df = pd.read_excel(input_xlsx_name, sheet_name=sheet_name) # 变量前加_表示private
    print("读取文件时间：", time.time()-start_time)
    # 组合，从10个数值中筛选8个数值,将tuple型转化为array，a为组合的结果
    # cmb_number = np.array(list(itertools.combinations([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],5)))
    cmb_number = None
    with open("cmb_number", 'rb') as f:
        cmb_number = pickle.load(f) # 被持久化后的对象还原
    List = [] #保存一个列表的植被指数
    print("排列组合时间：", time.time()-start_time)
    print("开始循环")
    # print(len(cmb_number))
    start_time = time.time()
    sub_originarl = []#记录原始反射率
    for number in cmb_number:
        sample_list=[] #保存一个样本的植被指数，每个循环清空
        a = time.time()
        # 选取的5个值作平均
        meandata=(_df.iloc[:,number[0]-1]+_df.iloc[:,number[1]-1]+_df.iloc[:,number[2]-1]+_df.iloc[:,number[3]-1]+\
                  _df.iloc[:,number[4]-1])/5

        # 读取三个波段的数值
        nir_value = meandata[2]
        red_value = meandata[0]
        green_value = meandata[1]

        sub_originarl.append(red_value)
        sub_originarl.append(green_value)
        sub_originarl.append(nir_value)

        # 计算植被指数
        normalizeddiff_veg_index = (nir_value - red_value) / (nir_value + red_value)
        sample_list.append(normalizeddiff_veg_index)

        #new
        sipi = (nir_value - green_value)/(nir_value + red_value)
        sample_list.append(sipi)

        #new
        tvi = 0.5*(120*(nir_value-green_value)/(nir_value+red_value))
        sample_list.append(tvi)

        diff_veg_index = nir_value - red_value
        sample_list.append(diff_veg_index)

        #new
        gdvi = nir_value -green_value
        sample_list.append(gdvi)

        soil_reg_veg_index = (1 + 0.16) * (nir_value - red_value) / (nir_value + red_value + 0.16)
        sample_list.append(soil_reg_veg_index)

        rvi = nir_value / red_value
        sample_list.append(rvi)

        #new
        sr = nir_value/green_value
        sample_list.append(sr)

        #new
        g = green_value/red_value
        sample_list.append(g)

        ndgi = (green_value - red_value) / (green_value + red_value)
        sample_list.append(ndgi)

        ipvi = nir_value / (nir_value + red_value)
        sample_list.append(ipvi)

        cvi = (nir_value * red_value) / (green_value ** 2)
        sample_list.append(cvi)

        #new
        mcari1 = 1.2*(2.5*(nir_value-red_value)-1.3*(red_value-green_value))
        sample_list.append(mcari1)

        #new
        mtvi1 = 1.2*(1.2*(nir_value-green_value)-2.5*(red_value-green_value))
        sample_list.append(mtvi1)

        #new
        mtvi2 = (1.5*(1.2*(nir_value-green_value)-2.5*(red_value-green_value)))/(cmath.sqrt(((2*nir_value+1)**2)-(6*nir_value-5*(cmath.sqrt(red_value)))-0.5))
        sample_list.append(mtvi2.real)

        #new
        rdvi = (nir_value-red_value)/(cmath.sqrt(nir_value+red_value))
        sample_list.append(rdvi.real)

        green_red_ndvi = (nir_value - red_value - green_value) / (nir_value + red_value + green_value)
        sample_list.append(green_red_ndvi)

        norm_red = red_value / (nir_value + red_value + green_value)
        sample_list.append(norm_red)

        norm_nir = nir_value / (nir_value + red_value + green_value)
        sample_list.append(norm_nir)

        norm_green = green_value / (nir_value + red_value + green_value)
        sample_list.append(norm_green)

        List.append(sample_list)
        print("内部循环一次时间：", time.time() - a)
    # 保存一个列表的植被指数
    sumindex = pd.DataFrame(List).rename(columns={0: 'NDVI',1: 'SIPI',2:'TVI',3: 'DVI',4: 'GDVI',
                                             5: 'OSAVI', 6: 'RVI',7: 'SR',8:'G',9:'NDGI',10:'IPVI',
                                             11: 'CVI',12: 'MCARI1',13:'MTVI1',14:'MTVI2',15: 'RDVI',
                                             16: 'GRNDVI',17: 'Norm R',18:'Norm NIR',19:'Norm G'})
    sub_originarl=np.array(sub_originarl)
    sub_originarl=sub_originarl.reshape(-1,3)
    sheetsum = pd.DataFrame(sub_originarl).rename(columns={0:'Red', 1:'Green',2: 'NIR'})

    with open("f2_original", 'wb') as f:
        pickle.dump(sheetsum, f, True)

    with open("f2_predict_index", 'wb') as f:
        pickle.dump(sumindex, f, True)
    print("循环一次时间：", time.time()-start_time)


if __name__ == '__main__':
    sheet_names=["Sheet1"]
    sub('E:\\论文\\培敏\\预测ROI\\sequoia_predict\\new\\f2\\predictsave.xlsx',"Sheet1")


