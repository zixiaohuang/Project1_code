import random
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.patches as mpatches

def main():
    print("function start")
    percentage = 0.2
    with open("all_data", 'rb') as f:
        all_data = pickle.load(f)


    all_dataselect = all_data.drop(
        labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
        n=20000, axis=0)
    pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X =all_dataselect.drop('label', axis=1)


    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=percentage, shuffle=True, random_state=42)
    X_trans = pca.fit_transform(X_train)
    # params = {'solver':['lbfgs'],'hidden_layer_sizes': [(i,i,j)for i in range(2,100)for j in range(2,50)],'random_state':[1],'alpha':[1e-4]}
    # clf = GridSearchCV(MLPClassifier(), params, cv=5, n_jobs=8)
    print("train start")
    start=time.time()
    #clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(64,64,8),random_state=42,alpha=1e-4,verbose=True)
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(64, 64, 8),learning_rate='adaptive',random_state=42, alpha=1e-4, verbose=True)
    print("1")
    clf.fit(X_trans, Y_train)
    print("2")
    Y_pred = clf.predict(pca.fit_transform(X_val))

    print("report")
    # cm = confusion_matrix(Y_val, Y_pred)
    # auc = 1.0 * (cm[0][0] + cm[1][1]) / sum(sum(cm))
    print(confusion_matrix(Y_val, Y_pred))
    print(classification_report(Y_val,Y_pred))

    # Y_score = clf.predict_proba(X_val)
    # enc = preprocessing.OneHotEncoder()
    # Y_val1 = enc.fit_transform(Y_val).toarray()
    fpr, tpr, threshold = roc_curve(Y_val, Y_pred)
    roc_auc = auc(fpr, tpr)
    plotROC(Y_val,Y_pred,roc_auc)
    print("cost time{}".format(time.time()-start))


    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='MultiPerceptron PCA ROC(area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(PCA:5components)')
    plt.legend(loc="lower right")
    plt.show()


def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

'''
def confusion_matrix_analysis(y_true, y_pred, labels):
    temp_labels = []
    for l in labels:
        temp_labels.append(str(l))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + temp_labels)
    ax.set_yticklabels([''] + temp_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
'''

def plotROC(predStrengths, classLabels,roc_auc):
    """
       Function：   绘制ROC曲线

       Input：      predStrengths：一个numpy数组或者一个行向量组成的矩阵，代表分类器的预测强度
                   classLabels：数据标签

        """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.figure(figsize=(10, 10))
    lw = 2
    # 保留绘制光标的位置
    cur =(1.0,1.0)
    # 用于计算AUC的值
    ySum = 0.0

    # 计算正例的数目
    numPosClas = sum(np.array(classLabels)==1.0)
    # 计算在y，x坐标上的步进数目
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    # 获取排行序的索引
    # sortedIndicies = np.array([predStrengths[i].argsort()])# argsort函数返回的是数组值从小到大的索引值
    sortedIndicies = np.mat(predStrengths.argsort())
    ySumList=[]
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        plt.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='darkorange',
                     lw=lw)
        cur = (cur[0]-delX,cur[1]-delY)
        ySumList.append(ySum)
    patches = [mpatches.Patch(color='darkorange', label='MultiPerceptron ROC(area= %0.2f)'%(roc_auc))]
    plt.legend(handles=patches,loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(PCA:3components)')
    # plt.axis([0, 1, 0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()


if __name__ == "__main__":
    '''
    all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                             names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                    'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                    'norm_green', 'cvi', 'green_red_ndvi', 'label'])
    with open("all_data", 'wb') as f:
        pickle.dump(all_data, f)
    '''
    main()



