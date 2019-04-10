import multiprocessing as mp
import time

def job(v,num,l):
    l.acquire() # 锁住，其他进程和线程附加值就不会给予干扰,基于p1相加完后加p2
    for _ in range(10):
        time.sleep(0.1)
        v.value += num
        print(v.value)
    l.release() # 释放

# 共享内存
def multicore():
    l = mp.Lock() # 避免进程间抢夺资源
    v = mp.Value('i',1) # 形式为i整数型，初始值为1
    p1 = mp.Process(target=job,args=(v,1,l)) # p1功能：每次给共享内存加1
    p2 = mp.Process(target=job,args=(v,3,l)) # p2功能：每次给共享内存加3
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == '__main__':
    multicore()