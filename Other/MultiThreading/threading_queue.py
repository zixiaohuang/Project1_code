import threading
import time
from queue import Queue

# 多线程的函数是不能有返回值，为了使用返回值需要用到queue
def job(l,q): # 把计算过的列表放进queue
    for i in range(len(l)):
        l[i]=l[i]**2
    q.put(l) # 多线程不能用retunr，return l

def multithreading():
    q=Queue() # 将计算的值放进Queue
    threads = []
    data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]
    for i in range(4): # 分批输入运算
        t = threading.Thread(target=job,args=(data[i],q))
        t.start()
        threads.append(t) #将运算结果添加到thread中
    for thread in threads: # 全部运算完才执行下面的程序
        thread.join()
    results =[]
    for _ in range(4):
        results.append(q.get()) # 从运算结果储存在queue中的值逐一取出
    print(results)


if __name__ == '__main__':
    multithreading()