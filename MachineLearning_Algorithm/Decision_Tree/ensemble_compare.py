'''
比较优化后集成学习不同算法的准确率
'''
import pickle
import time
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

if __name__ == '__main__':
    with open('E:\\Project1_code\\Datasets\\pca_data', 'rb') as f1:
        pca_data = pickle.load(f1)
    pca_data=pca_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\auto_code', 'rb') as f2:
        code_data = pickle.load(f2)
    code_data = code_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\code_original', 'rb') as f3:
        codeori_data = pickle.load(f3)
    codeori_data = codeori_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\pca_original', 'rb') as f4:
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
        clf_adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',max_depth=11,min_impurity_decrease=0.0,min_samples_leaf=1,min_samples_split=20),
                                          algorithm='SAMME.R',random_state=42,learning_rate=0.8,n_estimators=170)
        clf_xgboost = XGBClassifier(booster='gbtree', nthread=-1,n_jobs=-1,objective='binary:logistic',learning_rate=0.05, gamma=0.1,
                                    max_depth=8,reg_lambda= 1,subsample=0.8,colsample_bytree=1.0,min_child_weight=3,seed=1000)
        clf_rf = RandomForestClassifier(n_estimators=80,max_depth=17,min_samples_split=50,min_samples_leaf=10,max_features=None,
                              random_state=42,oob_score=True,n_jobs=-1)
        clfs=[clf_adaboost,clf_xgboost,clf_rf]
        titles = ['AdaBoost','XgBoost', 'RandomForest']

        for clf,j in zip(clfs,range(len(clfs))):
            begin =time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            print("{0} 循环一次时间:{1}".format(titles[j],time.time()-begin))