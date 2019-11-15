'''
比较各svm核函数并绘制roc曲线
'''
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy import interp
import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == '__main__':
    # all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
    #                          names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
    #                                 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
    #                                 'norm_green', 'cvi', 'green_red_ndvi', 'label'])
    #
    # all_dataselect = all_data.drop(labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(n=10000,axis=0)
    # pca = std_PCA(n_components=5) #PCA降维为两个特征值
    # y = np.array(all_dataselect['label']).ravel()
    # X = all_dataselect.drop('label', axis=1)
    # X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    # # cv = list(StratifiedKFold(n_splits=3,random_state=1).split(X,y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_trainpca, X_testpca, y_trainpca, y_testpca = train_test_split(X_pca, y, test_size=0.1, random_state=42)

    #定义函数
    clf_linear = SVC(C=1, kernel='linear')
    clf_poly = SVC(C=1, kernel='poly', degree=3)
    clf_rbf = SVC(C=6.32, kernel='rbf', gamma=0.0947)
    clfs = [clf_linear, clf_poly, clf_rbf]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold=dict()
    # linear
    y_score = clf_linear.fit(X_train,y_train).decision_function(X_test)
    fpr[0],tpr[0],threshold[0] = roc_curve(y_test,y_score)
    roc_auc[0] = auc(fpr[0],tpr[0])
    #linear_pca
    y_linearpca =  clf_linear.fit(X_trainpca,y_trainpca).decision_function(X_testpca)
    fpr[1],tpr[1],threshold[1]=roc_curve(y_testpca,y_linearpca)
    roc_auc[1] = auc(fpr[1],tpr[1])
    #poly
    y_score21 = clf_poly.fit(X_train, y_train).decision_function(X_test)
    fpr[2], tpr[2],threshold[2] = roc_curve(y_test, y_score21)
    roc_auc[2] = auc(fpr[2], tpr[2])
    #poly_pca
    y_polypca = clf_poly.fit(X_trainpca, y_trainpca).decision_function(X_testpca)
    fpr[3], tpr[3],threshold[3] = roc_curve(y_testpca, y_polypca)
    roc_auc[3] = auc(fpr[3], tpr[3])
    #rbf
    y_rbf = clf_rbf.fit(X_train, y_train).decision_function(X_test)
    fpr[4], tpr[4],threshold[4]= roc_curve(y_test, y_rbf)
    roc_auc[4] = auc(fpr[4], tpr[4])
    #rbf_pca
    y_rbfpca = clf_rbf.fit(X_trainpca, y_trainpca).decision_function(X_testpca)
    fpr[5], tpr[5],threshold[5]= roc_curve(y_testpca, y_rbfpca)
    roc_auc[5] = auc(fpr[5], tpr[5])

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    color = ['aqua', 'r','g', 'pink', 'y', 'darkorange']
    label = ['Linear ROC(area = %0.2f)', 'Linear PCA ROC(area = %0.2f)',
             'Polynomial-3Degree ROC(area = %0.2f)',
             'Polynomial-3D_PCA ROC(area = %0.2f)', 'Gaussian ROC(area = %0.2f)',
             'Gaussian PCA ROC(area = %0.2f)']
    for i in range(len(roc_auc)):
        plt.plot(fpr[i], tpr[i], color=color[i],
             lw=lw, label=label[i] % roc_auc[i])  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(PCA:5components)')
    plt.legend(loc="lower right")
    plt.show()



'''
    fig = plt.figure(figsize=(7,5))

    mean_tpr =0.0
    mean_fpr = np.linspace(0,1,100)
    all_tpr =[]


    for i, (train, test) in enumerate(cv):
        # clf_linear
        probas__linear = clf_linear.fit(X[train],y[train]).predict_proba(X[test])
        fpr_linear, tpr_linear, thresholds_linear = roc_curve(y[test],probas__linear[:,1],pos_label=1)
        mean_tpr += interp(mean_fpr,fpr_linear,tpr_linear)
        mean_tpr[0]=0.0
        roc_auc = auc(fpr_linear,tpr_linear)
        plt.plot(fpr_linear,tpr_linear,lw=1,label='Linear Kernel ROC(area = %0.3f)'%(roc_auc))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('Receiver Operator Characteristic')
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.show()
'''


