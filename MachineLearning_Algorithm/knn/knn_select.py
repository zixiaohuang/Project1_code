import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == "__main__":
    # all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
    #                          names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
    #                                 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
    #                                 'norm_green', 'cvi', 'green_red_ndvi', 'label'])
    #
    # all_dataselect = all_data.drop(
    #     labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
    #     n=20000, axis=0)
    # pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    # y = np.array(all_dataselect['label']).ravel()
    # with open('E:\\VgIndex2.py\\数据\\pca_data', 'rb') as f1:
    #     #     pca_data = pickle.load(f1)
    with open('E:\\VgIndex2.py\\数据\\auto_code', 'rb') as f1:
        code_data = pickle.load(f1)
    code_data = code_data.sample(20000,axis=0)
    X = code_data.drop('label', axis=1)
    y = code_data['label']
    # X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    radius=np.linspace(5,500,20)
    # number = np.linspace(2,2000,10)
    number = [17,18,19,20,21,22,23,24,25]
    tree_size = np.linspace(30,1000,50)
    p = [1,2]

    #param_grid = {'algorithm':['kd_tree'],'radius':radius,'n_jobs':[6]}
    #clf = GridSearchCV(RadiusNeighborsClassifier(), param_grid, cv=5, n_jobs=6)
    param_grid = {'algorithm':['kd_tree'],'weights':['distance'],'n_neighbors':number,'leaf_size':tree_size,'p':[2],'n_jobs':[-1]}
    clf = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5,n_jobs=-1)
    clf.fit(X_train, y_train)
    print("best param: {0}\n best score: {1}".format(clf.best_params_, clf.best_score_))