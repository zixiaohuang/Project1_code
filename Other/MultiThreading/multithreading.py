'''
多线程有一全局的控制，未必平均把任务分配给每个线程运算
为了实现多线程，让程序把线程锁住
GIL会让唯一一个线程在这一时间作运算，等线程运算完这一步，进行释放，跳到另一线程，
同时用GIL把其他线程锁上，不断在线程间进行切换
把读写的时间在整个程序中扣除掉 I/O密集型
多线程不一定很有效率
如果是运算型的数据，用到多进程，每个核有单独的逻辑空间，不受GIL的影响
'''


import threading
import time

def thread_job():
    print('This is an added Thread, number is %s' %threading.current_thread())
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish\n")

def T2_job():
    print('T2 start\n')
    print('T2 finish\n')


def main():
    added_thread = threading.Thread(target= thread_job,name='T1') # 添加线程,给线程工作为thread_job
    thread2 = threading.Thread(target=T2_job,name='T2')
    added_thread.start() # 开始执行线程
    thread2.start()
    thread2.join()
    added_thread.join() # 若没有join，多线程是同时进行的任务，加上join需要等待线程完成才执行下面的命令
    print('all done\n')
    # print(threading.active_count()) # 查看有多少激活的线程
    # print(threading.enumerate()) # 查看激活的线程名称
    # print(threading.current_thread()) # 运行时的线程名

if __name__ == '__main__':
    main()