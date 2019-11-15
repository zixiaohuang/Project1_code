'''
PCA压缩特征，并绘制三维图像
'''

import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest


def std_PCA():
    scaler = StandardScaler(copy=False)
    pca = PCA(n_components=3,copy=False) #n_components为整数时为压缩的维度，为float值时表示保存的占原有数据信息量
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == "__main__":
    time1 = time.time()
    # with open('E:\\modifiedversion\\Datasets\\new_alldata', 'rb') as f:
    #     data = pickle.load(f)
    with open('E:\\modifiedversion\\PredictData\\F2\\f2_predict_index', 'rb') as f:
        data = pickle.load(f)
    print("0,time:{}".format(time.time()-time1))
    pca = std_PCA()  # PCA降维为两个特征值
    X = data.drop(labels=['TVI', 'MCARI1','DVI', 'OSAVI', 'IPVI', 'GRNDVI'], axis=1)
    X_pca = pca.fit_transform(X)

    #保存pca压缩的数据
    X_pd=pd.DataFrame(X_pca)
    with open('f2_predict_pca', 'wb') as f:
        pickle.dump(X_pd, f)
