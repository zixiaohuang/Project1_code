import matplotlib.pyplot as plt
import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
def std_PCA():
    scaler = StandardScaler(copy=False)
    pca = PCA(n_components=0.999,copy=False) #n_components为整数时为压缩的维度，为float值时表示保存的占原有数据信息量
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == "__main__":
    with open('E:\\Project1_code\\Datasets\\new_alldata', 'rb') as f:
        data = pickle.load(f)
    pca = std_PCA()  # PCA降维为两个特征值
    X = data.drop(labels=['TVI', 'MCARI1', 'OSAVI', 'IPVI', 'GRNDVI', 'label'], axis=1)
    X_pca = pca.fit_transform(X)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=10000, random_state=42)
    selector = SelectKBest(k=3)
    X_new = selector.fit_transform(X_test, y_test)
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Datasets after PCA')
    ax.scatter(X_new[y_test == 1][:, 0], X_new[y_test == 1][:, 1], X_new[y_test == 1][:, 2], c='g', s=10,
               marker='o')  # 绿色代表健康
    ax.scatter(X_new[y_test == -1][:, 0], X_new[y_test == -1][:, 1], X_new[y_test == -1][:, 2], c='r', s=10,
               marker='^')  # 红色代表有病

    ax.set_zlabel('feature3')  # 坐标轴
    ax.set_ylabel('feature2')
    ax.set_xlabel('feature1')
    for angle in range(95,180,3):
        ax.view_init(30,angle)
        filename="./"+str(angle)+".png"
        plt.savefig(filename)
