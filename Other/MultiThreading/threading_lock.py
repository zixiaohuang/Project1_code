import threading

def job1():
    global A #全局变量
    for i in range(10):
        A += 1
        print('job1',A)

def job2():
    global A
    for i in range(10):
        A += 10
        print('job2',A)

# 不加锁lock，会同时进行
if __name__ == '__main__':
    A=0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()