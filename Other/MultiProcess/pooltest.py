import multiprocessing as mp

def job(x):
    return x*x

def multicore():
    pool = mp.Pool() # 默认是选用全部的核数 要设定多少个核：Pool（processes=2）
    res = pool.map(job,range(100)) # 往pool放进要运算的方程和运算的值，map可以接受多个值迭代，自动分配给进程
    print(res)
    res = pool.apply_async(job,(2,)) # apply_async只接受一个值，放进一个核、一个进程运算得出结果
    print(res.get())
    # 若要用apply_async接受多个值
    multi_res = [pool.apply_async(job,(i,))for i in range(10)]
    print([res.get() for res in multi_res])

if __name__ == '__main__':
    multicore()