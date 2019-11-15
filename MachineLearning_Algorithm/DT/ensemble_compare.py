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
from sklearn.metrics import  cohen_kappa_score
from sklearn.externals import joblib

if __name__ == '__main__':
    # training data
    with open('E:\\modifiedversion\\Datasets\\original_data\\original_shuffle_train', 'rb') as f1:
        original_data = pickle.load(f1)
    original_data = original_data.sample(n=20000, axis=0)

    with open('E:\\modifiedversion\\Datasets\\pcadata\\pca_shuffle_train', 'rb') as f2:
        pca_data = pickle.load(f2)
    pca_data = pca_data.sample(n=20000, axis=0)

    with open('E:\\modifiedversion\\Datasets\\autodata\\auto_shuffle_train', 'rb') as f3:
        code_data = pickle.load(f3)
    code_data = code_data.sample(n=20000, axis=0)

    with open('E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_train', 'rb') as f4:
        all_data = pickle.load(f4)
    pcaori_data = all_data.drop(labels=['auto1', 'auto2', 'auto3'], axis=1)
    pcaori_data = pcaori_data.sample(n=20000, axis=0)

    codeori_data = all_data.drop(labels=['pca1', 'pca2', 'pca3'], axis=1)
    codeori_data = codeori_data.sample(n=20000, axis=0)

    datas = [original_data, pca_data, code_data, pcaori_data, codeori_data]
    data_title = ['original_data', 'pca_data', 'code_data', 'pca_originaldata', 'code_originaldata']

    # testing data
    with open('E:\\modifiedversion\\Datasets\\original_data\\original_shuffle_save', 'rb') as f1:
        original_data_test = pickle.load(f1)
    original_data_test = original_data_test.sample(n=20000, axis=0)

    with open('E:\\modifiedversion\\Datasets\\pcadata\\pca_shuffle_save', 'rb') as f2:
        pca_data_test = pickle.load(f2)
    pca_data_test = pca_data_test.sample(n=20000, axis=0)

    with open('E:\\modifiedversion\\Datasets\\autodata\\auto_shuffle_save', 'rb') as f3:
        code_data_test = pickle.load(f3)
    code_data_test = code_data_test.sample(n=20000, axis=0)

    with open('E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_save', 'rb') as f4:
        all_data_test = pickle.load(f4)
    pcaori_data_test = all_data_test.drop(labels=['auto1', 'auto2', 'auto3'], axis=1)
    pcaori_data_test = pcaori_data_test.sample(n=20000, axis=0)

    codeori_data_test = all_data_test.drop(labels=['pca1', 'pca2', 'pca3'], axis=1)
    codeori_data_test = codeori_data_test.sample(n=20000, axis=0)

    test_datas = [original_data_test, pca_data_test, code_data_test, pcaori_data_test, codeori_data_test]

    for data,test_data,i in zip(datas,test_datas,range(len(datas))):
        print("{} datas start predict".format(data_title[i]))
        X_train = data.drop('label', axis=1)
        y_train = data['label']
        X_test = test_data.drop('label', axis=1)
        y_test = test_data['label']

        # 定义函数
        clf_adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=19,min_impurity_decrease=0.0,min_samples_leaf=3,min_samples_split=6),
                                          algorithm='SAMME.R',random_state=42,learning_rate=1,n_estimators=440)
        clf_xgboost = XGBClassifier(booster='gbtree', nthread=-1,n_jobs=-1,objective='binary:logistic',learning_rate=0.3, gamma=0.2,
                                    max_depth=8,reg_lambda= 1,subsample=1,colsample_bytree= 0.7777777777777778,min_child_weight=3,seed=1000)
        clf_rf = RandomForestClassifier(n_estimators=220,max_depth=21,min_samples_split=30,min_samples_leaf=10,max_features=3,
                              random_state=42,oob_score=True,n_jobs=-1)
        clfs=[clf_adaboost,clf_xgboost,clf_rf]
        titles = ['AdaBoost','XgBoost', 'RandomForest']

        for clf,j in zip(clfs,range(len(clfs))):
            begin =time.time()
            clf.fit(X_train, y_train)
            print("{0} 循环一次时间:{1}".format(titles[j], time.time() - begin))
            joblib.dump(clf, "{0}_{1}".format(data_title[i], titles[j]))
            y_pred = clf.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred,digits=4))
            print("Cohen_kappa系数:", cohen_kappa_score(y_test, y_pred))