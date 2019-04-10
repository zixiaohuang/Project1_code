from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

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
'''
X_train,X_test, y_train, y_test =cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
# train_data：所要划分的样本特征集
# train_target：所要划分的样本结果
# test_size：样本占比，如果是整数的话就是样本的数量
# random_state：是随机数的种子。
# 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
'''

def feature_extraction(data, threshold):
    n_columns = len(data[0])
    pca = PCA(svd_solver='full')
    pca.fit(data)

    for i, val in enumerate(pca.singular_values_):
        if abs(val) < threshold:
            n_columns = i
            break

    pca = pca.set_params(n_components=n_columns)
    return pca

singular_values_threshold = 0.01

y = np.array(loadDataSet('E:\论文\培敏\数据处理\index.txt')).ravel()
X = np.array(loadDataSet('E:\论文\培敏\数据处理\sumdata_pca.txt'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0,shuffle=True)


'''
然后，我们将MLPClassifier类实例化。用n_hidden设置神经网络架构中隐藏层的层数。
我们将隐藏层的层数设置为两层。MLPClassifier类自动创建两个输入单元和一个输出单元。
在多元分类问题中分类器会为每一个可能的类型创建一个输出。

选择神经网络架构是很复杂的事情。确定隐藏单元和隐藏层的数量有一些首要原则，但是都没有必然的依据。
隐藏单元的数量由样本数量，训练数据的噪声，要被近似的函数复杂性，隐藏单元的激励函数，学习算法和使用的正则化方法决定。
实际上，架构的效果只能通过交叉检验得出。
'''

# 数据预处理，MLP对特征尺度敏感,可用sklearn内置的StandardScaler进行标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


clf = MLPClassifier(hidden_layer_sizes=(32,32,2), activation='relu', solver= 'sgd',learning_rate='adaptive', max_iter=2500, random_state=0)
# 我们通过fit()函数训练模型
clf.fit(X_train, y_train)
# 打印估计模型对测试集预测的准确率和一些手工输入的预测结果。预测测试集的结果表明，这个人工神经网络可以完美的近似XOR函数
print('层数:%s,输出单元数量：%s' %(clf.n_layers_,clf.n_outputs_))
predictions = clf.predict(X_test)
print('准确率：%s' %clf.score(X_test,y_test))
for i , p in enumerate(predictions[:10]):
    print('真实值：%s, 预测值：%s' %(y_test[i],p))

# sklearn自带的评估方法
print(classification_report(y_test,predictions))




