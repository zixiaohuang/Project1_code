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
def sub(input_xlsx_name, sheet_name, list_queue, lock):
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
    for number in cmb_number:
        sample_list=[] #保存一个样本的植被指数，每个循环清空
        a = time.time()
        # 选取的5个值作平均
        meandata=(_df.iloc[:,number[0]-1]+_df.iloc[:,number[1]-1]+_df.iloc[:,number[2]-1]+_df.iloc[:,number[3]-1]+\
                  _df.iloc[:,number[4]-1])/5

        # 读取三个波段的数值
        nir_value = meandata[0]
        red_value = meandata[1]
        green_value = meandata[2]

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

    lock.acquire() # 锁住，其他进程和线程附加值就不会给予干扰
    list_queue.put(sumindex) # 计算完的对象List放到queue中
    lock.release() # 释放
    print("循环一次时间：", time.time()-start_time)


if __name__ == '__main__':
    cmb_number = np.array(list(itertools.combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 5)))
    #  wb以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
    with open("cmb_number", 'wb') as f:
        pickle.dump(cmb_number, f, True)  # pickle 模块将对象转化为文件保存在磁盘上，在需要的时候再读取并还原 pickle.dump(obj, file[, protocol])
    #illsheet_names = ["A89", "C56", "C85"]
    #illsheet_names=["A89","C56","C85","D85","D88","F65","F66","F74","F82","G67","G69","G79","G81","G82","G83","G84",\
                   #"H65","H80","H81","H82","I44","I46","I50","I51","I52","I54","I60"]
    healthsheet_names = ["A54","A55","A57","A58","A61","A62","A63","A65","A66","A68","A70","A74","A76","A77","A79"
                         ,"A81","A82","A83","A84","A85","B51","B54","B56","B80","B83","B84","B85"]
    # healthsheet_names =["A54","A55","A57"]


    # 多进程读取数据计算存储于queue
    manager = Manager()
    queue =  manager.Queue() # 父进程创建Queue，并传给各个子进程
    lock = manager.Lock() # Lock（）避免进程间抢夺资源，不加锁会同步进行
    pool = Pool() # pool = Pool(processes = 4) 进程池，放到进程池中，python会自行解决多进程的问题，Pool默认大小为CPU核数，默认调用所有的核
    for i, name in enumerate(healthsheet_names):
        pool.apply_async(sub, args=('C:\\Users\\Administrator\\Desktop\\数据处理2\\健康\\HealthOriginal.xlsx',\
            name,queue,lock,))  # apply_async只接受一个值，放进一个核、一个进程运算得出结果
    pool.close() # 关闭进程池，其他进程无法加入
    pool.join() # join()方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步
    print("完成计算，合并数据")
    assert queue.qsize() == len(healthsheet_names) # 确保计算已完成并都存于queue中

    # normal 写入excel
    #while queue.qsize() > 0:
    # 创建空的dataframe所有列表植被指数的结果
    write_start = time.time()
    print("开始合并数据，写入")
    sheetsum=pd.DataFrame(columns=['NDVI','SIPI','TVI', 'DVI','GDVI','OSAVI','RVI','SR','G','NDGI','IPVI',
                                    'CVI','MCARI1','MTVI1','MTVI2', 'RDVI', 'GRNDVI','Norm R','Norm NIR','Norm G'])
    for sheet_name in healthsheet_names:
        sumindex = queue.get()
        sheetsum = sheetsum.append(sumindex)# 将各列表汇总
        print("合并列表：{}".format(sheet_name))
    # 存入txt
    np.savetxt('E:\\combinate_newdata\\vegindex_health_new.txt',sheetsum)
    print("normal写入时间：", time.time() - write_start)
    # 从txt中读取数据(路径不能有中文)
    # data =pd.read_table("路径\\_.txt",sep=" ",names=['NDVI','SIPI','TVI', 'DVI','GDVI','OSAVI','RVI','SR','G','NDGI','IPVI',
    #                                     'CVI','MCARI1','MTVI1','MTVI2', 'RDVI', 'GRNDVI','Norm R','Norm NIR','Norm G'])

