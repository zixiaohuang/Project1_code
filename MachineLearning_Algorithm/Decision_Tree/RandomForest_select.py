import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


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
    df = pd.read_excel(filename)
    X = df[df.columns[0:-1]].values
    y = df.label.values.reshape(-1,1)
    enc = preprocessing.OneHotEncoder()
    Y = enc.fit_transform(y).toarray()
    return X, Y, enc

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

def plot_curve(train_sizes,cv_results,xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(6,4),dpi=144)
    plt.figure('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,
                     alpha=0.1,color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,
                     alpha=0.1,color="g")
    plt.plot(train_sizes,train_scores_mean,'.--',color="r",
             label="Training score")
    plt.plot(train_sizes,test_scores_mean,'.-',color="g",
             label="Cross-validation score")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    #y = np.array(loadDataSet('E:\\论文\\培敏\\数据处理2\\index.txt')).ravel()
    #X = np.array(loadDataSet('E:\\论文\\培敏\\数据处理2\\sumdata.txt'))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
    # all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
    #                          names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
    #                                 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
    #                                 'norm_green', 'cvi', 'green_red_ndvi', 'label'])
    # with open("all_data", 'wb') as f:
    #     pickle.dump(all_data,f)
    with open("all_data", 'rb') as f:
        all_data = pickle.load(f)
    all_dataselect = all_data.drop(
        labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
        n=20000, axis=0)
    pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X = all_dataselect.drop('label', axis=1)
    X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)


    # forest_model = RandomForestClassifier(random_state=1)
    # n_estimators=range(10,201,10)
    # param_test1 = {'n_estimators':n_estimators}

    # max_depth=range(10,101,5)
    # min_samples_split=range(200,2000,50)
    # max_depth = range(3, 30, 2)
    # min_samples_split = range(50, 301, 20)
    # param_test2 = {'max_depth': max_depth, 'min_samples_split': min_samples_split}

    # min_samples_split = range(50, 301, 10)
    # min_samples_leaf = range(10,40,5)
    # param_test3 = {'min_samples_split': min_samples_split,'min_samples_leaf':min_samples_leaf}


    # param_test4={'max_features':[3,4,5,6]}
    forest_model = GridSearchCV(estimator=RandomForestClassifier(n_estimators=80,max_depth=17,min_samples_split=50,
                                min_samples_leaf=10,random_state=42,oob_score=True,n_jobs=-1),
                                param_grid=param_test4,scoring='roc_auc',cv=5,n_jobs=-1)
    forest_model.fit(X_train, y_train)
    print(forest_model.cv_results_)
    print("best param:{0}\nbest score:{1}".format(forest_model.best_params_, forest_model.best_score_))
    # plot_curve(max_depth,forest_model.cv_results_,xlabel='number of max_depth')
    # plot_curve(min_samples_split, forest_model.cv_results_, xlabel='number of min_samples_split')

    # def cv_score(val):
    #     clf = RandomForestClassifier(n_estimators=val, min_samples_split=100, min_samples_leaf=20,
    #                                  max_depth=8, max_features='sqrt', random_state=42, n_jobs=-1)
    #     clf.fit(X_train,y_train)
    #     cv_score = clf.score(X_train, y_train)
    #     tr_score = clf.score(X_test, y_test)
    #     return(tr_score,cv_score)
    #
    # x = list(range(10, 101, 10))
    # scores = [cv_score(x[v]) for v in range(len(x))]
    # tr_scores = [s[0] for s in scores]
    # cv_scores=[s[1] for s in scores]
    # # 画出模型参数与模型评分的关系
    # plt.figure(figsize=(6,4),dpi=144)
    # plt.grid()
    # plt.xlabel('number of n_estimators')
    # plt.ylabel('score')
    # plt.xlim((0, 110))
    # plt.ylim((0.8,1.01))
    # plt.plot(x,cv_scores,'.g-',label='cross-validation score')
    # plt.plot(x,tr_scores,'.r--',label='training score')
    # plt.legend()
    # plt.show()


    # melb_preds = forest_model.predict(X_test)
    # print(mean_absolute_error(y_test, melb_preds))
    #
    # # 做预测
    # print(confusion_matrix(y_test,melb_preds))
    # print(classification_report(y_test,melb_preds))
    #
    # fpr, tpr, thresholds = metrics.roc_curve(y_test,melb_preds)
    # auc = metrics.auc(fpr,tpr)
    #
    # plt.plot(fpr,tpr,label='ROC curve (area=%.2f)'%auc)
    # plt.legend()
    # plt.title('ROC curve')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.grid(True)
    # plt.show()