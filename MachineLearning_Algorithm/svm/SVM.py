'''
建立svm模型
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn import metrics

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    dataMat = []
    for line in fr.readlines(): # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，如果碰到结束符 EOF 则返回空字符串
        stringArr = line.strip().split(delim) # strip()删除s字符串中开头、结尾处，位于 rm删除序列的字符
        datArr = []
        for line in stringArr:
            datArr.append(float(line))
        dataMat.append(datArr)
    return np.mat(dataMat)

def load_data(filename):
    df = pd.read_excel(filename).drop(columns = ['sample'])
    X = df[df.columns[0:-1]].values
    y = df.label.values.reshape(-1,1)
    enc = preprocessing.OneHotEncoder()
    Y = enc.fit_transform(y).toarray()
    return X, Y, enc

if __name__ == '__main__':
    all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                             names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                    'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                    'norm_green', 'cvi', 'green_red_ndvi', 'label'])

    all_dataselect = all_data.drop(
        labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1)
    y = np.array(all_data['label']).ravel()
    X = all_data.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.997, random_state=42)  # 选部分样本进行计算

    # X, Y, enc = load_data("C:\\Users\\Administrator\\Desktop\\数据处理2\\VegIndexData2.xlsx")
    # training_features, testing_features, training_target, testing_target = train_test_split(X, Y, random_state=None)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 0,shuffle=True)

    clf_linear = SVC(C=6.32, kernel='linear')
    clf_ploy = SVC(C=6.32, kernel='ploy',degree=2)
    clf_rbf = SVC(C=6.32,kernel='rbf', gamma=0.0947)

    svclassifier = SVC(kernel='poly',degree=3)
    svclassifier.fit(X_train,y_train)

    # 做预测
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
    auc = metrics.auc(fpr,tpr)

    plt.plot(fpr,tpr,label='ROC curve (area=%.2f)'%auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()