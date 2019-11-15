'''PCA选择压缩多少个特征k'''
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

if __name__ =="__main__":
    print("Exploring explained variance ratio for datasets ...")
    with open('E:\\modifiedversion\\Datasets\\new_alldata', 'rb') as f:
        data = pickle.load(f)
    X = data.drop(labels=['TVI', 'MCARI1', 'OSAVI', 'IPVI', 'GRNDVI', 'label'], axis=1)
    scaler = preprocessing.StandardScaler().fit(X)
    X_standard=scaler.transform(X)
    candidate_components=range(1,15,1)
    explained_ratios=[]
    start=time.clock()
    for c in candidate_components:
        pca=PCA(n_components=c)
        X_pca=pca.fit_transform(X_standard)
        explained_ratios.append(np.sum(pca.explained_variance_ratio_))
    print('Done in {0:.2f}s'.format(time.clock()-start))

    # plt.figure(figsize=(10,6),dpi=144)
    plt.figure()
    plt.grid()
    plt.plot(candidate_components,explained_ratios)
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Explained Variance Ratio')
    # plt.title('Explained variance ratio for PCA')
    plt.yticks(np.arange(0.5,1.10,.05))
    plt.xticks(np.arange(0,17,1))
    plt.show()