import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


def main():
    percentage = 0.2
    singular_values_threshold = 0.005
    alpha = 1e-6

    X, Y, enc = load_data()
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=percentage, random_state= 20)  # shuffle，洗牌，是否在拆分前重组数据

    scaler = preprocessing.MinMaxScaler().fit(X_train)  # MinMaxScaler()将属性缩放到一个指定的最大和最小值（通常是1-0）之间, 相当于标准化？
    X_train = scaler.transform(X_train)  # 经过scaler的转换后，X_train和X_val的值在0-1
    X_val = scaler.transform(X_val)
    extractor = feature_extraction(X_train, singular_values_threshold)  # pca 压缩提取特征
    #返回  PCA(copy=True, iterated_power='auto', n_components=8, random_state=None,svd_solver='full', tol=0.0, whiten=False)

    clf = MLPClassifier(solver='lbfgs',
                        alpha=alpha,
                        hidden_layer_sizes=(32, 32, 2),
                        random_state=20)
    clf.fit(extractor.transform(X_train), Y_train)  # ？

    Y_pred = clf.predict(extractor.transform(X_val))  # ？ predict返回的是一个大小为样本数n的一维数组，一维数组中的第i个值为模型预测第i个预测样本的标签
    y_val = enc.inverse_transform(Y_val) # enc.inverse_transform是将Y_val的数值转换为对应的ill[0,1]或health[1,0]标记
    y_pred = enc.inverse_transform(Y_pred)

    # 做预测
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    confusion_matrix_analysis(y_val, y_pred, enc.categories_[0]) # enc.categories[0]为array(['health', 'ill'], dtype=object)

    prob = clf.predict_proba(extractor.transform(X_val)) #  predict_proba返回的是一个n（样本数）行k（分类数）列的数组，
                                                        # 第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。
    #                                                   此时每一行的和应该等于1
    receiver_operating_characteristic(Y_val, prob)

    return scaler, extractor, clf


def load_data():
    df = pd.read_excel("E:\论文\培敏\数据处理\嘉明\VIData_select.xlsx").drop(columns=['sample'])

    X = df[df.columns[0:-1]].values
    y = df.label.values.reshape(-1, 1)  # 重新定义了原张量的形状，-1为自动推测
    enc = preprocessing.OneHotEncoder()
    Y = enc.fit_transform(y).toarray() # enc.fit_transform将y的ill/health转换为数值表示
    return X, Y, enc


def feature_extraction(data, threshold):
    n_columns = len(data[0])
    pca = PCA(svd_solver='full')  # full则是传统意义上的SVD，使用了scipy库对应的实现
    pca.fit(data) # PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,svd_solver='full', tol=0.0, whiten=False)

    for i, val in enumerate(pca.singular_values_):  # 筛选符合特征阈值的特征值 i为0,1，2,3,4...,val为特征值
        if abs(val) < threshold:
            n_columns = i
            break

    pca = pca.set_params(n_components=n_columns)  # 设置参数，保留符合特征阈值的矩阵
    return pca


def confusion_matrix_analysis(y_true, y_pred, labels):
    temp_labels = []
    for l in labels:
        temp_labels.append(str(l)) #temp_labels为['health', 'ill']
    cm = confusion_matrix(y_true, y_pred, labels=labels) #返回一个矩阵，矩阵为[[health被判为health, health被判为ill],[ill被判为health,ill被判为ill]]
    fig = plt.figure()
    ax = fig.add_subplot(111) # 111”表示“1×1网格，第一子图”
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # 用蓝色绘制出混淆矩阵
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + temp_labels)
    ax.set_yticklabels([''] + temp_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def receiver_operating_characteristic(Y_true, prob):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(Y_true[0])):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], prob[:, i]) # fpr和tpr就是混淆矩阵中的FP和TP的值；thresholds就是y_score逆序排列后的结果
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='aqua', lw=lw,
             label='ROC curve of class health (area = {1:0.2f})'
                   ''.format(0, roc_auc[0]))
    plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw,
             label='ROC curve of class ill (area = {1:0.2f})'
                   ''.format(1, roc_auc[1]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0]) # 设置x坐标的范围
    plt.ylim([0.0, 1.05]) # 设置y坐标的范围
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right") # 绘制图例
    plt.show()


if __name__ == "__main__":
    main()

# 为什么不先pca压缩特征？
# 为什么每次测试的结果不一样？ shuffle 将数据打乱后，由于初始的数据顺序不再保持与原来一致，则即使设置了randomstate为整数，每次的结果也不再相同?