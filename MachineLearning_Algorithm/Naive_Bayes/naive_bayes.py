
import pickle
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == "__main__":
    # with open("all_data", 'rb') as f:
    #     all_data = pickle.load(f)
    #
    # all_dataselect = all_data.drop(
    #     labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
    #     n=40000, axis=0)
    # pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    # y = np.array(all_dataselect['label']).ravel()
    # X = all_dataselect.drop('label', axis=1)
    # X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    # X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)
    # with open("X_train", 'wb') as f:
    #     pickle.dump(X_train, f)
    # with open("X_test", 'wb') as f:
    #     pickle.dump(X_test, f)
    # with open("y_train", 'wb') as f:
    #     pickle.dump(y_train, f)
    # with open("y_test", 'wb') as f:
    #     pickle.dump(y_test, f)

    with open("X_train", 'rb') as f:
        X_train = pickle.load(f)
    with open("X_test", 'rb') as f:
        X_test = pickle.load(f)
    with open("y_train", 'rb') as f:
        y_train = pickle.load(f)
    with open("y_test", 'rb') as f:
        y_test = pickle.load(f)

    # Gaussian
    # from sklearn.naive_bayes import GaussianNB
    # clf=GaussianNB()
    # start=time.time()
    # clf.fit(X_train,y_train)
    # y_pred=clf.predict(X_test)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test,y_pred))
    # print("cost time:{}".format(time.time()-start))

    #MultinomialNB 不支持X为负值，需要预处理
    # from sklearn.naive_bayes import MultinomialNB
    # from sklearn.preprocessing import MinMaxScaler
    # min_max_scaler = MinMaxScaler()
    # X_p=min_max_scaler.fit_transform(X=X_train)
    # X_t=min_max_scaler.fit_transform(X=X_test)
    # clf=MultinomialNB()
    # start = time.time()
    # clf.fit(X_p,y_train)
    # y_pred = clf.predict(X_t)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test,y_pred))
    # print("cost time:{}".format(time.time()-start))

    # BernoulliNB
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(X_train,y_train)
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("cost time:{}".format(time.time() - start))
