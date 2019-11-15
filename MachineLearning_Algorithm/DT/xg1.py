#import xgboost as xgb
import pickle
from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.metrics import confusion_matrix, roc_curve, auc,classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

def receiver_operating_characteristic1(Y_true, prob):# Y_true为真实值，prob为预测值
    print("Ploting ROC graph ...")
    cv = StratifiedKFold(Y_true,n_folds=10)
    fpr,tpr,threshold = roc_curve(Y_true, prob)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve of class ill (area = {1:0.2f})'
                   ''.format(1, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])  # 设置x坐标的范围
    plt.ylim([0.0, 1.05])  # 设置y坐标的范围
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def receiver_operating_characteristic(Y_true, prob):  # Y_true为真实值，prob为预测值
    print("Ploting ROC graph ...")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(Y_true[0])):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], prob[:, i])  # fpr和tpr就是混淆矩阵中的FP和TP的值；thresholds就是y_score逆序排列后的结果
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(),
                                              prob.ravel())  # sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw,
             label='ROC curve of class ill (area = {1:0.2f})'
                   ''.format(1, roc_auc[1]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])  # 设置x坐标的范围
    plt.ylim([0.0, 1.05])  # 设置y坐标的范围
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def main():
    with open("E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_tune",'rb') as f:
        all_data = pickle.load(f)
    all_data = all_data.sample(n=60000, axis=0)
    X = all_data.drop('label', axis=1)
    y = all_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # subsample = np.linspace(0.5,1,10)
    colsample = np.linspace(0.5,1,10)
    learning_rate = [0.01,0.05,0.1,0.15,0.2,0.3]
    params = [{
        'booster':[ 'gbtree'], # 有两中模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。缺省值为gbtree
        'silent':[0], # 输出中间过程为1，不输出为0
        'nthread': [-1],  # 使用cpu 全部进程
        'n_jobs':[-1],
        'objective': ['binary:logistic'],  # 二分类问题
        'learning_rate':learning_rate, #含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。
        'gamma': [0.1,0.2],  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': range(3,10),  # 构建树的深度，越大越容易过拟合
        'reg_lambda': [1],  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': [0.5,0.8,1],  # 随机采样训练样本,使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
        'colsample_bytree': colsample,  # 生成树时进行的列采样,使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
        'min_child_weight': [3],
        'seed': [1000]
        #n_jobs':[4] #并行线程数
    }]
    xgb_model = GridSearchCV(XGBClassifier(), params, cv=5, n_jobs=8)
    # xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    print("best param:{0}\nbest score:{1}".format(xgb_model.best_params_,xgb_model.best_score_))
    probas_ = xgb_model.predict_proba(X_test)
    pred = xgb_model.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    receiver_operating_characteristic(y_test, probas_)

'''
    kf = KFold(n_splits=2,shuffle=True,random_state=1234)
    for train_index, test_index in kf.split(X):
        xgboost_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
        pred = xgboost_model.predict(X[test_index])
        ground_truth = y[test_index]
        print(confusion_matrix(ground_truth,pred))
        print(classification_report(ground_truth,pred ))
        receiver_operating_characteristic(ground_truth, pred)  # Y_val为真实值，prob为预测值
'''



if __name__ == "__main__":
    main()
