'''
该代码采用于多进程处理将导出的ROI三个波段的反射率进行排列组合取平均值
'''

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
    List = []
    print("排列组合时间：", time.time()-start_time)
    print("开始循环")
    # print(len(cmb_number))
    start_time = time.time()
    for number in cmb_number:
        a = time.time()
        # 选取的6个值作平均
        meandata=(_df.iloc[:,number[0]-1]+_df.iloc[:,number[1]-1]+_df.iloc[:,number[2]-1]+_df.iloc[:,number[3]-1]+\
                  _df.iloc[:,number[4]-1])/5
        List.append(meandata)
        print("内部循环一次时间：", time.time()-a)
    lock.acquire() # 锁住，其他进程和线程附加值就不会给予干扰
    list_queue.put(List) # 计算完的对象List放到queue中
    lock.release() # 释放
    print("循环一次时间：", time.time()-start_time)

# 多进程从queue取出计算结果并写入excel
def writer2excel(sheet_name,queue,lock):
    book = load_workbook('C:\\Users\Administrator\\Desktop\\数据处理2\\健康\\healthoutput.xlsx')
    writer = pd.ExcelWriter('C:\\Users\Administrator\\Desktop\\数据处理2\\健康\\healthoutput.xlsx',engine='openpyxl')
    writer.book = book
    writer.sheets = sheet_name
    #lock.acquire()
    List = queue.get()  # 从queue中得到计算出的结果，每次只能取一次
    sheetsum = pd.DataFrame(List)
    sheetsum.to_excel(excel_writer=writer, sheet_name=sheet_name)
    writer.save()
    writer.close()
    # lock.release()

def thread2excel(output_xlsx_name,sheet_name,queue):
    writer = pd.ExcelWriter(output_xlsx_name)
    List=queue.get()
    sheetsum = pd.DataFrame(List)
    sheetsum.to_excel(writer, sheet_name=sheet_name)
    writer.save()


if __name__ == '__main__':
    cmb_number = np.array(list(itertools.combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 5)))
    #  wb以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
    with open("cmb_number", 'wb') as f:
        pickle.dump(cmb_number, f, True)  # pickle 模块将对象转化为文件保存在磁盘上，在需要的时候再读取并还原 pickle.dump(obj, file[, protocol])
    # illsheet_names = ["A89", "C56", "C85"]
    # illsheet_names=["A89","C56","C85","D85","D88","F65","F66","F74","F82","G67","G69","G79","G81","G82","G83","G84",\
                   # "H65","H80","H81","H82","I44","I46","I50","I51","I52","I54","I60"]
    healthsheet_names = ["A54","A55","A57","A58","A61","A62","A63","A65","A66","A68","A70","A74","A76","A77","A79"
                        ,"A81","A82","A83","A84","A85","B51","B54","B56","B80","B83","B84","B85"]


    # 多进程读取数据计算存储于queue
    manager = Manager()
    queue =  manager.Queue() # 父进程创建Queue，并传给各个子进程
    lock = manager.Lock() # Lock（）避免进程间抢夺资源，不加锁会同步进行
    pool = Pool() # pool = Pool(processes = 4) 进程池，放到进程池中，python会自行解决多进程的问题，Pool默认大小为CPU核数，默认调用所有的核
    for i, name in enumerate(healthsheet_names):
        pool.apply_async(sub, args=('C:\\Users\Administrator\\Desktop\\数据处理2\\健康\\HealthOriginal.xlsx',\
            name,queue,lock,))  # apply_async只接受一个值，放进一个核、一个进程运算得出结果
    pool.close() # 关闭进程池，其他进程无法加入
    pool.join() # join()方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步
    print("完成计算，合并数据")
    assert queue.qsize() == len(healthsheet_names) # 确保计算已完成并都存于queue中

    '''
    # 多进程写入excel
    manager2 = Manager()
    lock2 =manager2.Lock()
    pool2 = Pool()
    print("开始写入")
    write_start = time.time()
    # while queue.qsize()>0:
    for i,sheet_name in enumerate(illsheet_names):
        pool2.apply_async(writer2excel,args=(sheet_name,queue,lock2,))
    pool2.close()
    pool2.join()
    print("多进程写入时间：", time.time() - write_start)
    '''
    # normal 写入excel
    output_xlsx_name = 'C:\\Users\Administrator\\Desktop\\数据处理2\\健康\\healthoutput.xlsx'
    writer = pd.ExcelWriter(output_xlsx_name)
    write_start = time.time()
    print("开始写入")
    while queue.qsize()>0:
        for sheet_name in healthsheet_names:
            List = queue.get()
            sheetsum=pd.DataFrame(List)
            sheetsum.to_excel(writer,sheet_name=sheet_name)
            print("写入列表：{}".format(sheet_name))
    writer.save()
    print("normal写入时间：", time.time() - write_start)

    '''
    # 多线程写入excel
    output_xlsx_name = 'C:\\Users\\Administrator\\Desktop\\数据处理2\\有病\\illoutput.xlsx'
    start = time.time()
    print('这是主线程:{}'.format(threading.current_thread().name))
    thread_list = []
    for sheet_name in illsheet_names:
        t = threading.Thread(target=thread2excel, args=(
        'C:\\Users\\Administrator\\Desktop\\数据处理2\\有病\\illoutput.xlsx', sheet_name, queue,))
        thread_list.append(t)

    # 开始运行
    for t in thread_list:
        t.start()

    # 主线程等子线程运行完才结束
    for t in thread_list:
        t.join()

    end =time.time()
    print('multithreading写入时间：',end-start)
    '''