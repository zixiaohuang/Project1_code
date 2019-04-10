import pickle
import time
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

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

        models = []
        models.append(("KNN with uniform weights", KNeighborsClassifier(algorithm='kd_tree',n_neighbors=18,leaf_size=30,p=1,weights='uniform',n_jobs=-1)))
        models.append(("KNN with distance weights", KNeighborsClassifier(algorithm='kd_tree',leaf_size=30,n_neighbors=22, p=2,weights="distance",n_jobs=-1)))
        # models.append(("Radius Neighbors", RadiusNeighborsClassifier(n_neighbors=2, radius=1.0,n_jobs=-1)))

        # 分别训练3个模型，并计算评分
        results = []
        for name, model in models:
            kfold = KFold(n_splits=10)
            cv_result = cross_val_score(model, X_train, y_train, cv=kfold)
            results.append((name, cv_result))
            start = time.time()
            # 做预测
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("{}模型训练结果,花费时间{}".format(name, time.time() - start))
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc = metrics.auc(fpr, tpr)
        for i in range(len(results)):
            print("name:{};cross val score:{}".format(results[i][0], results[i][1].mean()))

    #基于半径的knn只适用于三维或以下的特征
    # # 不同的数据以不同半径作为判定标准的knn，需要单独计算
    # data_code = [code_data,codeori_data]
    # data_title = ['pca_data', 'pca_originaldata']
    # for data, i in zip(datas, range(len(data_code))):
    #     print("{} datas start predict".format(data_title[i]))
    #     X = data.drop('label', axis=1)
    #     y = data['label']
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    #     neigh_radius1 = RadiusNeighborsClassifier(n_neighbors=2, radius=2.0,n_jobs=-1)
    #     results = []
    #     kfold = KFold(n_splits=10)
    #     cv_result = cross_val_score(neigh_radius1, X_train, y_train, cv=kfold)
    #     results.append((cv_result))
    #     start = time.time()
    #     # 做预测
    #     neigh_radius1.fit(X_train, y_train)
    #     y_pred = neigh_radius1.predict(X_test)
    #     print("{}模型训练结果,花费时间{}".format("neigh_radius", time.time() - start))
    #     print(confusion_matrix(y_test, y_pred))
    #     print(classification_report(y_test, y_pred))
    #     fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    #     auc = metrics.auc(fpr, tpr)
    #     for i in range(len(results)):
    #         print("name:{};cross val score:{}".format(results[i][0], results[i][1].mean()))
    #
    # data_pca = [pca_data, pcaori_data]
    # data_title = ['code_data', 'code_originaldata']
    # for data, i in zip(datas, range(len(data_pca))):
    #     print("{} datas start predict".format(data_title[i]))
    #     X = data.drop('label', axis=1)
    #     y = data['label']
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    #     neigh_radius2 = RadiusNeighborsClassifier(n_neighbors=2, radius=5.0, n_jobs=-1)
    #     results = []
    #     kfold = KFold(n_splits=10)
    #     cv_result = cross_val_score(neigh_radius2, X_train, y_train, cv=kfold)
    #     results.append((cv_result))
    #     start = time.time()
    #     # 做预测
    #     neigh_radius2.fit(X_train, y_train)
    #     y_pred = neigh_radius2.predict(X_test)
    #     print("{}模型训练结果,花费时间{}".format("neigh_radius", time.time() - start))
    #     print(confusion_matrix(y_test, y_pred))
    #     print(classification_report(y_test, y_pred))
    #     fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    #     auc = metrics.auc(fpr, tpr)
    #     for i in range(len(results)):
    #         print("name:{};cross val score:{}".format(results[i][0], results[i][1].mean()))