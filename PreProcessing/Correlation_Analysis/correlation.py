'''
计算各植被指数间的相似度
'''

import numpy as np
import numpy.linalg as la
import pandas as pd


def eulidSim(inA, inB):
    return 1.0/(1.0+la.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    dataMat = []
    for line in fr.readlines(): # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，如果碰到结束符 EOF 则返回空字符串
        stringArr = line.strip().split(delim) # strip()删除s字符串中开头、结尾处，位于 rm删除序列的字符
        datArr = []
        for line in stringArr:
            datArr.append(float(line))
        dataMat.append(datArr)
    return np.mat(dataMat)

def normalization(dataMat):
    meanVals = np.mean(dataMat, axis=1)  # axis =1 压缩列，对各行求均值 axis=0 压缩行
    meanRemoved = dataMat - meanVals
    rowMax = dataMat.max(axis=1)  # 获取每行的最大值
    rowMin = dataMat.min(axis=1)  # 获取每行的最小值
    rowDiff = rowMax - rowMin
    normalVals = meanRemoved / rowDiff  # 归一化处理
    return normalVals

if __name__ == "__main__":
    myMat = np.mat(loadDataSet('E:\论文\培敏\数据处理2\sum2.txt'))
    newMat = normalization(myMat)
    eulid = []
    pears = []
    cos = []
    for i in range(11):
        for j in range(i,11):
            eulid.append(eulidSim(newMat[i,:],newMat[j,:]))
            pears.append(pearsSim(newMat[i,:],newMat[j,:]))
            cos.append(cosSim(newMat[i,:].T,newMat[j,:].T))
    data_df1 = pd.DataFrame(eulid)
    data_df2 = pd.DataFrame(pears)
    data_df3 = pd.DataFrame(cos)

    # 分析结果保存的路径
    writer = pd.ExcelWriter('C:\\Users\\Administrator\\Desktop\\新建文件夹\\save.xlsx')
    data_df1.to_excel(writer,'eulidSim',float_format='%.20f')
    data_df2.to_excel(writer, 'pearsSim', float_format='%.20f')
    data_df3.to_excel(writer, 'cosSim', float_format='%.20f')

    writer.save()
