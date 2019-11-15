import pickle
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

if __name__ == '__main__':
    with open('E:\\Project1_code\\Datasets\\all_originaldata', 'rb') as f1:
        original_data = pickle.load(f1)
    original_data=original_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\pca_data', 'rb') as f2:
        pca_data = pickle.load(f2)
    pca_data=pca_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\auto_code', 'rb') as f3:
        code_data = pickle.load(f3)
    code_data = code_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\pca_original', 'rb') as f4:
        pcaori_data = pickle.load(f4)
    pcaori_data = pcaori_data.sample(n=50000,axis=0)

    with open('E:\\Project1_code\\Datasets\\code_original', 'rb') as f5:
        codeori_data = pickle.load(f5)
    codeori_data = codeori_data.sample(n=50000,axis=0)

    datas = [original_data,pca_data,code_data,pcaori_data, codeori_data]
    data_title = ['original_data','pca_data','code_data','pca_originaldata','code_originaldata']

    for data,i in zip(datas,range(len(datas))):
        print("{} datas start predict".format(data_title[i]))
        X = data.drop('label', axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # 定义函数
        clf_linear = SVC(C=1, kernel='linear')
        clf_poly = SVC(C=1, kernel='poly', degree=3)
        clf_rbf = SVC(C=6.32, kernel='rbf', gamma=0.0947)
        clfs = [clf_linear, clf_poly, clf_rbf]
        titles = ['Linear Kernel',
                  'Polynomial Kernel with Degree=3',
                  'Gaussian Kernel with gamma=0.0947']

        for clf,j in zip(clfs,range(len(clfs))):
            begin =time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred,digits=4))
            print("{0} 循环一次时间:{1}".format(titles[j],time.time()-begin))