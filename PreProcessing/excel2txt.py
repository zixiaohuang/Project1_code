import  pandas as pd
import numpy as np
from multiprocessing import Pool,Manager
import time

def sub(input_xlsx_name, sheet_names, list_queue, lock):
    start_time = time.time()
    # sheetsum = pd.DataFrame(columns=['NIR','Red','Green'])
    # for name in enumerate(sheet_names):
    _df = pd.read_excel(input_xlsx_name, sheet_name=sheet_names)
    # sheetsum=sheetsum.append(_df)
    lock.acquire()  # 锁住，其他进程和线程附加值就不会给予干扰
    list_queue.put(_df)  # 计算完的对象List放到queue中
    lock.release()  # 释放
    print("循环一次时间：", time.time() - start_time)

if __name__ == '__main__':
    # 多进程读取数据计算存储于queue
    manager = Manager()
    queue = manager.Queue()  # 父进程创建Queue，并传给各个子进程
    lock = manager.Lock()  # Lock（）避免进程间抢夺资源，不加锁会同步进行
    pool = Pool()  # pool = Pool(processes = 4) 进程池，放到进程池中，python会自行解决多进程的问题，Pool默认大小为CPU核数，默认调用所有的核
    input_xlsx_name='C:\\Users\\Administrator\\Desktop\\数据处理2\\健康\\healthoutput.xlsx'
    input_xlsx_illname = 'C:\\Users\\Administrator\\Desktop\\数据处理2\\有病\\illoutput.xlsx'
    output_illtxtname = 'E:\\combinate_newdata\\illorginaloutput.txt'
    output_txtname='E:\\combinate_newdata\\healthoutput.txt'
    illsheet_names = ["A89", "C56", "C85", "D85", "D88", "F65", "F66", "F74", "F82", "G67", "G69", "G79", "G81", "G82",
                      "G83", "G84", "H65", "H80", "H81", "H82", "I44", "I46", "I50", "I51", "I52", "I54", "I60"]
    healthsheet_names = ["A54", "A55", "A57", "A58", "A61", "A62", "A63", "A65", "A66", "A68", "A70", "A74", "A76",
                         "A77", "A79"
        , "A81", "A82", "A83", "A84", "A85", "B51", "B54", "B56", "B80", "B83", "B84", "B85"]
    for name in illsheet_names:
        pool.apply_async(sub, args=(input_xlsx_illname,name,queue, lock,))
    pool.close()  # 关闭进程池，其他进程无法加入
    pool.join()  # join()方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步
    sheetsum = pd.DataFrame(columns=['NIR', 'Red', 'Green'])
    assert queue.qsize() == len(healthsheet_names)
    # 存入txt
    for sheet_name in illsheet_names:
        data = queue.get()
        sheetsum=sheetsum.append(data)
    np.savetxt(output_illtxtname,sheetsum)