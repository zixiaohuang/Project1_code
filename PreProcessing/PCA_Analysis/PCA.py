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
    with open('E:\\modifiedversion\\predict', 'rb') as f:
        data = pickle.load(f)
    print("0,time:{}".format(time.time()-time1))
    pca = std_PCA()  # PCA降维为两个特征值
    X = data.drop(labels=['TVI', 'MCARI1','DVI', 'OSAVI', 'IPVI', 'GRNDVI', 'label'], axis=1)
    X_pca = pca.fit_transform(X)
    y = data['label']
    # 取10%展示
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=10000, random_state=42)
    print("1,time:{}".format(time.time()-time1))
    #保存pca压缩的数据
    X_pd=pd.DataFrame(X_pca)
    y_reset = y.reset_index(drop=True)
    df = pd.concat([X_pd, y_reset], axis=1)
    # with open('pca_data2', 'wb') as f:
    #     pickle.dump(df, f)
    with open('pca_data2', 'wb') as f:
        pickle.dump(df, f)
    print("2,time:{}".format(time.time()-time1))
    selector = SelectKBest(k=3)
    X_new = selector.fit_transform(X_test, y_test)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_title('Datasets after PCA')

    print("3,time:{}".format(time.time()-time1))
    ax.scatter(X_new[y_test==1][:,0],X_new[y_test==1][:,1],X_new[y_test==1][:,2],c='g',s=10,marker='o')#绿色代表健康
    ax.scatter(X_new[y_test==-1][:,0], X_new[y_test==-1][:,1], X_new[y_test==-1][:,2], c='r', s=10,marker='^')#红色代表有病

    ax.set_zlabel('feature3')  # 坐标轴
    ax.set_ylabel('feature2')
    ax.set_xlabel('feature1')
    plt.show()