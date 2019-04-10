import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import GridSearchCV

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == "__main__":
    all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                             names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                    'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                    'norm_green', 'cvi', 'green_red_ndvi', 'label'])

    all_dataselect = all_data.drop(
        labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
        n=20000, axis=0)
    pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X = all_dataselect.drop('label', axis=1)
    X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

    models=[]
    models.append(("KNN",KNeighborsClassifier(n_neighbors=2)))
    models.append(("KNN with weights",KNeighborsClassifier(n_neighbors=2,weights="distance")))
    models.append(("Radius Neighbors",RadiusNeighborsClassifier(n_neighbors=2,radius=1.0)))

    # 分别训练3个模型，并计算评分
    results=[]
    for name,model in models:
        start=time.time()
        kfold = KFold(n_splits=10)
        cv_result = cross_val_score(model,X_train,y_train,cv=kfold)
        results.append((name,cv_result))
        # 做预测
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print("{}模型训练结果,花费时间{}".format(name,time.time()-start))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr, tpr)
    for i in range(len(results)):
        print("name:{};cross val score:{}".format(results[i][0],results[i][1].mean()))