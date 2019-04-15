import pickle
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

def multi_bayes():
    min_max_scaler = MinMaxScaler()
    clf = MultinomialNB()
    pipline = Pipeline([("polynomial_features",min_max_scaler),
                        ("logistic_regression",clf)])
    return pipline

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

        # 定义函数
        clf_gnb = GaussianNB()
        clf_mulit = multi_bayes()
        clf_nb = BernoulliNB()
        clfs = [clf_gnb, clf_mulit, clf_nb]
        titles = ['GaussianNB',
                  'MultinomialNB',
                  'BernoulliNB']

        for clf,j in zip(clfs,range(len(clfs))):
            begin =time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred,digits=4))
            print("{0} 循环一次时间:{1}".format(titles[j],time.time()-begin))