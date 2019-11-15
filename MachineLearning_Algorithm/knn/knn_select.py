import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    with open("E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_tune",'rb') as f:
        all_data = pickle.load(f)
    code_data = all_data.sample(50000,axis=0)
    X = code_data.drop('label', axis=1)
    y = code_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    radius=np.linspace(5,500,20)
    # number = np.linspace(2,2000,10)
    # number = [17,18,19,20,21,22,23,24,25]
    number = [7,9,11,13,15,17,21]
    tree_size = np.linspace(30,1000,50)
    p = [1,2]

    # param_grid = {'algorithm':['kd_tree'],'radius':radius,'n_jobs':[6]}
    param_grid = {'algorithm': ['kd_tree'],'weights':['uniform'], 'n_neighbors': number, 'leaf_size': tree_size, 'p': p, 'n_jobs': [-1]}
    #clf = GridSearchCV(RadiusNeighborsClassifier(), param_grid, cv=5, n_jobs=6)
    # param_grid = {'algorithm':['kd_tree'],'weights':['distance'],'n_neighbors':number,'leaf_size':tree_size,'p':[2],'n_jobs':[-1]}
    clf = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5,n_jobs=-1)
    clf.fit(X_train, y_train)
    print("best param: {0}\n best score: {1}".format(clf.best_params_, clf.best_score_))