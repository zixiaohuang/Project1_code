import pickle
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix

if __name__ == '__main__':
    with open('E:\\Project1_code\\Datasets\\pca_data', 'rb') as f1:
        pca_data = pickle.load(f1)
    pca_data=pca_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\auto_code', 'rb') as f2:
        code_data = pickle.load(f2)
    code_data = code_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\pca_original', 'rb') as f4:
        pcaori_data = pickle.load(f4)
    pcaori_data = pcaori_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\code_original', 'rb') as f3:
        codeori_data = pickle.load(f3)
    codeori_data = codeori_data.sample(n=50000,axis=0)

    datas = [pca_data,code_data,pcaori_data,codeori_data]
    data_title = ['pca_data','code_data','pca_originaldata','code_originaldata']

    for data,i in zip(datas,range(len(datas))):
        print("{} datas start predict".format(data_title[i]))
        X = data.drop('label', axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        models = []
        # models.append(("Linear Regression", LinearRegression(n_jobs=-1)))
        models.append(("Logistic Regression", LogisticRegression(penalty='l2',solver='saga',max_iter=10000,warm_start=True,n_jobs=-1,random_state=42)))
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
            print(classification_report(y_test, y_pred,digits=4))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc = metrics.auc(fpr, tpr)
        for i in range(len(results)):
            print("name:{};cross val score:{}".format(results[i][0], results[i][1].mean()))
