import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.patches as mpatches
import time

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
      Function：   通过阈值比较对数据进行分类

      Input：      dataMatrix：数据集
                  dimen：数据集列数
                  threshVal：阈值
                  threshIneq：比较方式：lt，gt

      Output： retArray：分类结果
      """
    # 新建一个数组用于存放分类结果，初始化都为1
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    # lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    # 返回分类结果
    return retArray

def buildStump(dataArr,classLabels,D):
    """
       Function：   找到最低错误率的单层决策树

       Input：      dataArr：数据集
                   classLabels：数据标签
                   D：权重向量

       Output： bestStump：分类结果
                   minError：最小错误率
                   bestClasEst：最佳单层决策树
       """
    # 初始化数据集和数据标签
    dataMatrix=np.mat(dataArr);labelMat=np.mat(classLabels).T
    # 获取行列值
    m,n=np.shape(dataMatrix)
    # 初始化步数，用于在特征的所有可能值上进行遍历
    numSteps=10.0
    # 初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump={}
    # 初始化类别估计值
    bestClassEst=np.mat(np.zeros((m,1)))
    # 将最小错误率设无穷大，之后用于寻找可能的最小错误率
    minError=np.inf;
    # 遍历数据集中每一个特征
    for i in range(n):
        # 获取数据集的最大最小值
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max();
        # 根据步数求得步长
        stepSize=(rangeMax-rangeMin)/numSteps
        # 遍历每个步长
        for j in range(-1,int(numSteps)+1):
            # 遍历每个不等号
            for inequal in ['lt','gt']:
                # 设定阈值
                threshVal = (rangeMin+float(j)*stepSize)
                # 通过阈值比较对数据进行分类
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                # 初始化错误计数向量
                errArr = np.mat(np.ones((m, 1)))
                # 如果预测结果和标签相同，则相应位置0
                errArr[predictedVals==labelMat]=0
                # 计算权值误差，这就是AdaBoost和分类器交互的地方
                weightedError=D.T*errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
                if weightedError< minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
        # 返回最佳单层决策树，最小错误率，类别估计值
        return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """
       Function：   找到最低错误率的单层决策树

       Input：      dataArr：数据集
                   classLabels：数据标签
                   numIt：迭代次数

       Output： weakClassArr：单层决策树列表
                   aggClassEst：类别估计值
       """
    # 初始化列表，用来存放单层决策树的信息
    weakClassArr=[]
    # 获取数据集行数
    m = np.shape(dataArr)[0]
    # 初始化向量D每个值均为1/m，D包含每个数据点的权重
    D = np.mat(np.ones((m, 1)) / m)
    # 初始化列向量，记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 开始迭代
    for i in range(numIt):
        # 利用buildStump()函数找到最佳的单层决策树
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
        alpha = float(0.5*math.log((1.0-error)/max(error,1e-16)))
        # 保存alpha的值
        bestStump['alpha']=alpha
        # 填入数据到列表
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T)
        # 为下一次迭代计算D
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 累加类别估计值
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        # 计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate=aggErrors.sum()/m
        print("total error",errorRate)
        # 如果总错误率为0则跳出循环
        if errorRate ==0.0:break
    # 返回单层决策树列表和累计错误率
    # return weakClassArr
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    """
    Function：   AdaBoost分类函数

    Input：      datToClass：待分类样例
                classifierArr：多个弱分类器组成的数组

    Output： sign(aggClassEst)：分类结果
     """
    # 初始化数据集
    dataMatrix = np.mat(datToClass)
    # 获得待分类样例个数
    m = np.shape(dataMatrix)[0]
    # 构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m,1)))
    # 遍历每个弱分类器
    for i in range(len(classifierArr)):
        # 基于stumpClassify得到类别估计值
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        # 累加类别估计值
        aggClassEst +=classifierArr[i]['alpha']*classEst
        # 打印aggClassEst，以便我们了解其变化情况
        print(aggClassEst)
        # 返回分类结果，aggClassEst大于0则返回+1，否则返回-1
    return np.sign(aggClassEst)

def plotROC(predStrengths, classLabels,roc_auc):
    """
       Function：   绘制ROC曲线

       Input：      predStrengths：一个numpy数组或者一个行向量组成的矩阵，代表分类器的预测强度
                   classLabels：数据标签

        """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.figure(figsize=(10, 10))
    lw = 2
    color = ['aqua', 'g', 'y', 'darkorange']
    label = ['DecisionTree ROC(area = %0.2f)', 'AdaBoost ROC(area = %0.2f)',
             'XgBoost ROC(area = %0.2f)', 'RandomForest ROC(area = %0.2f)']
    for i in range(len(predStrengths)):
        # 保留绘制光标的位置
        cur =(1.0,1.0)
        # 用于计算AUC的值
        ySum = 0.0

        # 计算正例的数目
        numPosClas = sum(np.array(classLabels)==1.0)
        # 计算在y，x坐标上的步进数目
        yStep = 1/float(numPosClas)
        xStep = 1/float(len(classLabels)-numPosClas)
        # 获取排行序的索引
        # sortedIndicies = np.array([predStrengths[i].argsort()])# argsort函数返回的是数组值从小到大的索引值
        sortedIndicies = np.mat(predStrengths[i].argsort())
        ySumList=[]
        for index in sortedIndicies.tolist()[0]:
            if classLabels[index]==1.0:
                delX=0
                delY=yStep
            else:
                delX=xStep
                delY=0
                ySum+=cur[1]
            plt.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c=color[i],
                     lw=lw)
            cur = (cur[0]-delX,cur[1]-delY)
            ySumList.append(ySum)
    patches = [mpatches.Patch(color=color[j], label=label[j]%(roc_auc[j]))for j in range(len(color))]
    plt.legend(handles=patches,loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(PCA:3components)')
    # plt.axis([0, 1, 0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()


if __name__ == "__main__":
    all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                             names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                    'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                    'norm_green', 'cvi', 'green_red_ndvi', 'label'])

    all_dataselect = all_data.drop(
        labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(n=20000,axis=0)
    pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X = all_dataselect.drop('label', axis=1)
    X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

    fpr=dict()
    tpr=dict()
    threshold=dict()
    roc_auc=dict()
    aggClassEstList=dict()
    labels = ['DecisionTree ', 'AdaBoost',
             'XgBoost', 'RandomForest']
    #decision tree
    start_tree=time.time()
    decisionTree=DecisionTreeClassifier(criterion= 'gini',max_depth=9,min_impurity_decrease= 0.05,min_samples_split=4)
    decisionTree.fit(X_train,y_train)
    y_score_dt =decisionTree.predict(X_test)
    aggClassEstList[0]=y_score_dt.T
    labellist=y_test.T
    # y_score_dt = decisionTree.predict_proba(X_test)
    fpr[0], tpr[0], threshold[0] = roc_curve(y_test, y_score_dt)
    roc_auc[0] = auc(fpr[0], tpr[0])
    print("{} cost time:{}s".format(labels[0],time.time()-start_tree))

    # adaboost
    start_ada = time.time()
    classifierArray, aggClassEst = adaBoostTrainDS(X_train, y_train, 100000)
    prediction = adaClassify(X_test, classifierArray)
    aggClassEstList[1]=prediction.T
    fpr[1], tpr[1], threshold[1] = roc_curve(y_test, prediction)
    roc_auc[1] = auc(fpr[1], tpr[1])
    print("{} cost time:{}s".format(labels[1],time.time()-start_ada))

    #xgboost
    start_xg= time.time()
    xgalg=XGBClassifier(booster='gbtree', nthread=-1,n_jobs=-1,objective='binary:logistic',
        learning_rate=0.05, gamma=0.1,  max_depth=8,reg_lambda= 1,subsample=0.8,colsample_bytree=1.0,
        min_child_weight=3,seed=1000)
    y_score_xg=xgalg.fit(X_train, y_train).predict(X_test)
    aggClassEstList[2]=y_score_xg.T
    # y_score_xg = xgalg.fit(X_train, y_train).predict_proba(X_test)
    fpr[2], tpr[2], threshold[2] = roc_curve(y_test, y_score_xg)
    roc_auc[2] = auc(fpr[2], tpr[2])
    print("{} cost time:{}s".format(labels[2], time.time() - start_xg))

    #randomforest
    start_rf =time.time()
    forest_model = RandomForestClassifier(random_state=1)
    forest_model.fit(X_train, y_train)
    y_score_rf = forest_model.predict(X_test)
    aggClassEstList[3]=y_score_rf.T
    # y_score_rf = forest_model.predict_proba(X_test)
    fpr[3], tpr[3], threshold[3] = roc_curve(y_test, y_score_rf)
    roc_auc[3] = auc(fpr[3], tpr[3])
    print("{} cost time:{}s".format(labels[2], time.time() - start_rf))

    plotROC(aggClassEstList,labellist,roc_auc)
'''
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    color = ['aqua',  'g',  'y', 'darkorange']
    label = ['DecisionTree ROC(area = %0.2f)', 'AdaBoost ROC(area = %0.2f)',
             'XgBoost ROC(area = %0.2f)','RandomForest ROC(area = %0.2f)']
    for i in range(len(roc_auc)):
        plt.plot(fpr[i], tpr[i], color=color[i],
             lw=lw, label=label[i] % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(PCA:3components)')
    plt.legend(loc="lower right")
    plt.show()
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(PCA:3components)')
    plt.axis([0, 1, 0, 1])

'''