import multiprocessing as mp
import threading as td
import time #测量效率,看运行时间

def job(q):
    res = 0
    for i in range(1000000):
        res+=i+i**2+i**3
    q.put(res) # Process无返回值，要放进queue里面

# 使用多核运算
def multicore():
    q=mp.Queue() # 把每个核的运算结果放进Queue，运算完再分别取出
    p1=mp.Process(target=job,args=(q,))
    p2=mp.Process(target=job,args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1=q.get()
    res2=q.get()
    print("MultiCore:",res1+res2)

# 普通
def normal():
    res = 0
    for _ in range(2):
        for i in range(1000000):
            res+=i+i**2+i**3
    print("Normal:",res)

# 运用多线程
def multithread():
    q = mp.Queue()  # 把每个核的运算结果放进Queue，运算完再分别取出
    p1 = td.Thread(target=job, args=(q,))
    p2 = td.Thread(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print("MultiThreading:",res1+res2)


if __name__ == '__main__':
    st=time.time()
    normal()
    st1=time.time()
    print("normal time:",st1-st)
    multicore()
    st2=time.time()
    print("multicore time:",st2-st1)
    multithread()
    print("multithreading time:",time.time()-st2)