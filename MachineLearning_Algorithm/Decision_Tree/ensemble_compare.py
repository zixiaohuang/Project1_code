import pickle
import time
import math
import numpy as np
from sklearn.model_selection import train_test_split

# def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
#     """
#       Function：   通过阈值比较对数据进行分类
#
#       Input：      dataMatrix：数据集
#                   dimen：数据集列数
#                   threshVal：阈值
#                   threshIneq：比较方式：lt，gt
#
#       Output： retArray：分类结果
#       """
#     # 新建一个数组用于存放分类结果，初始化都为1
#     retArray = np.ones((np.shape(dataMatrix)[0],1))
#     # lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray
#     if threshIneq == 'lt':
#         retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
#     else:
#         retArray[dataMatrix[:,dimen]>threshVal]=-1.0
#     # 返回分类结果
#     return retArray
#
# def buildStump(dataArr,classLabels,D):
#     """
#        Function：   找到最低错误率的单层决策树
#
#        Input：      dataArr：数据集
#                    classLabels：数据标签
#                    D：权重向量
#
#        Output： bestStump：分类结果
#                    minError：最小错误率
#                    bestClasEst：最佳单层决策树
#        """
#     # 初始化数据集和数据标签
#     dataMatrix=np.mat(dataArr);labelMat=np.mat(classLabels).T
#     # 获取行列值
#     m,n=np.shape(dataMatrix)
#     # 初始化步数，用于在特征的所有可能值上进行遍历
#     numSteps=10.0
#     # 初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
#     bestStump={}
#     # 初始化类别估计值
#     bestClassEst=np.mat(np.zeros((m,1)))
#     # 将最小错误率设无穷大，之后用于寻找可能的最小错误率
#     minError=np.inf;
#     # 遍历数据集中每一个特征
#     for i in range(n):
#         # 获取数据集的最大最小值
#         rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max();
#         # 根据步数求得步长
#         stepSize=(rangeMax-rangeMin)/numSteps
#         # 遍历每个步长
#         for j in range(-1,int(numSteps)+1):
#             # 遍历每个不等号
#             for inequal in ['lt','gt']:
#                 # 设定阈值
#                 threshVal = (rangeMin+float(j)*stepSize)
#                 # 通过阈值比较对数据进行分类
#                 predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
#                 # 初始化错误计数向量
#                 errArr = np.mat(np.ones((m, 1)))
#                 # 如果预测结果和标签相同，则相应位置0
#                 errArr[predictedVals==labelMat]=0
#                 # 计算权值误差，这就是AdaBoost和分类器交互的地方
#                 weightedError=D.T*errArr
#                 # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
#                 # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
#                 if weightedError< minError:
#                     minError=weightedError
#                     bestClassEst=predictedVals.copy()
#                     bestStump['dim']=i
#                     bestStump['thresh']=threshVal
#                     bestStump['ineq']=inequal
#         # 返回最佳单层决策树，最小错误率，类别估计值
#         return bestStump,minError,bestClassEst
#
# def adaBoostTrainDS(dataArr,classLabels,numIt=40):
#     """
#        Function：   找到最低错误率的单层决策树
#
#        Input：      dataArr：数据集
#                    classLabels：数据标签
#                    numIt：迭代次数
#
#        Output： weakClassArr：单层决策树列表
#                    aggClassEst：类别估计值
#        """
#     # 初始化列表，用来存放单层决策树的信息
#     weakClassArr=[]
#     # 获取数据集行数
#     m = np.shape(dataArr)[0]
#     # 初始化向量D每个值均为1/m，D包含每个数据点的权重
#     D = np.mat(np.ones((m, 1)) / m)
#     # 初始化列向量，记录每个数据点的类别估计累计值
#     aggClassEst = np.mat(np.zeros((m, 1)))
#     # 开始迭代
#     for i in range(numIt):
#         # 利用buildStump()函数找到最佳的单层决策树
#         bestStump,error,classEst=buildStump(dataArr,classLabels,D)
#         print("D:",D.T)
#         # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
#         alpha = float(0.5*math.log((1.0-error)/max(error,1e-16)))
#         # 保存alpha的值
#         bestStump['alpha']=alpha
#         # 填入数据到列表
#         weakClassArr.append(bestStump)
#         print("classEst:",classEst.T)
#         # 为下一次迭代计算D
#         expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
#         D = np.multiply(D, np.exp(expon))
#         D = D / D.sum()
#         # 累加类别估计值
#         aggClassEst += alpha * classEst
#         print("aggClassEst: ", aggClassEst.T)
#         # 计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
#         aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
#         errorRate=aggErrors.sum()/m
#         print("total error",errorRate)
#         # 如果总错误率为0则跳出循环
#         if errorRate ==0.0:break
#     # 返回单层决策树列表和累计错误率
#     # return weakClassArr
#     return weakClassArr,aggClassEst
#
# def adaClassify(datToClass,classifierArr):
#     """
#     Function：   AdaBoost分类函数
#
#     Input：      datToClass：待分类样例
#                 classifierArr：多个弱分类器组成的数组
#
#     Output： sign(aggClassEst)：分类结果
#      """
#     # 初始化数据集
#     dataMatrix = np.mat(datToClass)
#     # 获得待分类样例个数
#     m = np.shape(dataMatrix)[0]
#     # 构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
#     aggClassEst = np.mat(np.zeros((m,1)))
#     # 遍历每个弱分类器
#     for i in range(len(classifierArr)):
#         # 基于stumpClassify得到类别估计值
#         classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], classifierArr[i]['thresh'],classifierArr[i]['ineq'])
#         # 累加类别估计值
#         aggClassEst +=classifierArr[i]['alpha']*classEst
#         # 打印aggClassEst，以便我们了解其变化情况
#         print(aggClassEst)
#         # 返回分类结果，aggClassEst大于0则返回+1，否则返回-1
#     return np.sign(aggClassEst)


if __name__ == '__main__':
    with open('E:\\VgIndex2.py\\数据\\pca_data', 'rb') as f1:
        pca_data = pickle.load(f1)
    pca_data=pca_data.sample(n=50000,axis=0)

    with open('E:\\VgIndex2.py\\数据\\auto_code', 'rb') as f2:
        code_data = pickle.load(f2)
    code_data = code_data.sample(n=50000,axis=0)

    with open('E:\\VgIndex2.py\\数据\\code_original', 'rb') as f3:
        codeori_data = pickle.load(f3)
    codeori_data = codeori_data.sample(n=50000,axis=0)

    with open('E:\\VgIndex2.py\\数据\\pca_original', 'rb') as f4:
        pcaori_data = pickle.load(f4)
    pcaori_data = pcaori_data.sample(n=50000,axis=0)

    datas = [pca_data,code_data,codeori_data,pcaori_data]
    data_title = ['pca_data','code_data','code_originaldata','pca_originaldata']

    for data,i in zip(datas,range(len(datas))):
        print("{} datas start predict".format(data_title[i]))
        X = data.drop('label', axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # 定义函数
        labels = ['AdaBoost','XgBoost', 'RandomForest']