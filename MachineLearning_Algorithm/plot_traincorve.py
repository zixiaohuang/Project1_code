import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import ShuffleSplit

with open("X_train53", 'rb') as f:
    X_train = pickle.load(f)
with open("X_test53", 'rb') as f:
    X_test = pickle.load(f)
with open("y_train53", 'rb') as f:
    y_train = pickle.load(f)
with open("y_test53", 'rb') as f:
    y_test = pickle.load(f)

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()
    # 颜色填充，alpha为透明度
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")
    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="Cross-validation score")
    plt.legend(loc="best")
    return plt

n_folds=5
# 绘制roc曲线
plt.figure()
lw = 2
fpr=dict()
tpr=dict()
threshold=dict()
roc_auc=dict()
labels = ['SVM(area = %0.2f)', 'KNN(area = %0.2f)','LR(area = %0.2f)', 'Bayes(area = %0.2f)','XGboost(area = %0.2f)',
          'RF(area = %0.2f)','MLP(area = %0.2f)','Averaging(area = %0.2f)','Stacked(area = %0.2f)','FinalEnsemble(area = %0.2f)']
color = ['grey','bisque',  'tan',  'gold', 'olivedrab','teal','skyblue','darkorchid','magenta','r']
linestyle=['-.','-.','-.','-.','-.','-.','-.','-','-','-']

# 模型的选择和使用
# Gaussian Kernel SVM with gamma=0.0947
from sklearn.svm import SVC
model_svm = SVC(C=6.32,kernel='rbf',gamma=0.0947)

#KNN with weight
from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=2,weights="distance",n_jobs=-1)

# logisticregression
from sklearn.linear_model import LogisticRegression
model_lgr = LogisticRegression(penalty='l2',solver='sag',n_jobs=-1,random_state=42,max_iter=10000)

# BernoulliNB
from sklearn.naive_bayes import BernoulliNB
model_nb = BernoulliNB()

# XGboost
from xgboost.sklearn import XGBClassifier
model_xg=XGBClassifier(booster='gbtree', nthread=-1,n_jobs=-1,objective='binary:logistic',
        learning_rate=0.05, gamma=0.1,  max_depth=8,reg_lambda= 1,subsample=0.8,colsample_bytree=1.0,
        min_child_weight=3,seed=1000)

# RandomForest
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier(n_estimators=80,max_depth=17,min_samples_split=50,min_samples_leaf=10,max_features=None,
                              random_state=42,oob_score=True,n_jobs=-1)

#Multiperceptron
from sklearn.neural_network import MLPClassifier
model_mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(64, 64, 8),learning_rate='constant',tol=1e-3,warm_start=True,
                    random_state=42, alpha=1e-4,early_stopping=True, shuffle=True,verbose=True,max_iter=50,n_iter_no_change=5)

# 基础模型的分类器的平均值
class AveragingModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,models):
        self.models=models
    # 模型的fit（）函数
    def fit(self,X,y):
        self.models_=[clone(x)for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X,y)
        return self
    # 模型的预测k-fold为预测的次数，将多列的数据合并，并取众数(平均值)
    def predict(self,X):
        # column_stack可以将多个列组合[0,1][2,3]组合为[[0,2][1,3]]
        predictions = np.column_stack([model.predict(X)for model in self.models_])
        # return np.mean(predictions,axis=1)
        q=dict()
        from collections import Counter
        for i in range(len(predictions[:,0])):#样本行数
            c=Counter(predictions[i]) #c为一个列表，返回每个值出现的次数，并从大到小排序
            q[i]=c.most_common(1)[0][0] #返回出现次数最大值的数值
        y_pred=np.array(list(q.values())) #取字典的值变为array
        return y_pred
        #np.bincount()可以把数组中出现的每个数字，当做index，数字出现的次数当做value来表示
        #np.argmax()可以返回数组中最大值的index
# 评价五个模型的好坏
# 评价五个模型的好坏
averaged_models = AveragingModels(models=(model_svm,model_knn,model_xg,model_nb,model_mlp))

class StackingAveragedModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # we again fit the data on clones of the original models
    def fit(self,X,y):
        self.base_models_ = [list ()for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds,shuffle=True,random_state=42)

        # 使用K-fold的方法来进行交叉验证，将每次验证的结果作为新的特征来处理
        out_of_fold_predictions = np.zeros((X.shape[0],len(self.base_models)))
        for i,model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X,y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index,i] = y_pred

        # 将交叉验证预测出的结果和训练集中的标签值进行训练
        self.meta_model_.fit(out_of_fold_predictions,y)
        return self

    # 从得到的新的特征 采用新的模型进行预测 并输出结果
    def predict(self,X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X)for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

#用svm、knn、xgboost、BernoulliNB、Multiperceptron进行模型融合，然后将融合的模型与randomforest、LogisticRegression进行权重投票
stacked_averaged_models = StackingAveragedModels(base_models=(model_knn,model_nb,model_svm,model_xg),meta_model=(model_mlp))

# model_ensemble = VotingClassifier(estimators=[('stacked',stacked_averaged_models),('lgr',model_lgr),('rf',model_rf)],voting='hard',weights=[3,2,2],n_jobs=-1)
model_ensemble = VotingClassifier(estimators=[('stacked',stacked_averaged_models),('average',averaged_models),('lgr',model_lgr),('rf',model_rf)],voting='hard',weights=[3,2,2,2])
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=42)
plot_learning_curve(estimator=model_ensemble,X=X_train,y=y_train,ylim=(0.8,1.01),n_jobs=-1,cv=cv,title='Training examples')
plt.show()