import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import VotingClassifier

with open("X_train53", 'rb') as f:
    X_train = pickle.load(f)
with open("X_test53", 'rb') as f:
    X_test = pickle.load(f)
with open("y_train53", 'rb') as f:
    y_train = pickle.load(f)
with open("y_test53", 'rb') as f:
    y_test = pickle.load(f)

# def std_PCA(**argv):
#     scaler = StandardScaler()
#     pca = PCA(**argv)
#     pipeline = Pipeline([('scaler',scaler),
#                          ('pca',pca)])
#     return pipeline
#
# with open("all_data", 'rb') as f:
#     all_data = pickle.load(f)
# all_dataselect = all_data.drop(
#     labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
#     n=200, axis=0)
# pca = std_PCA(n_components=3)  # PCA降维为两个特征值
# y = np.array(all_dataselect['label']).ravel()
# X = all_dataselect.drop('label', axis=1)
# X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

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

def rmsle_cv(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits() # get_n_splits:Returns the number of splitting iterations in the cross-validator
    # cross_val_score: Evaluate a score by cross-validation
    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring="neg_mean_squared_error",cv=kf))
    return(rmse)

# 模型的选择和使用
# Gaussian Kernel SVM with gamma=0.0947
from sklearn.svm import SVC
svm_start=time.time()
model_svm = SVC(C=6.32,kernel='rbf',gamma=0.0947)
model_svm.fit(X_train,y_train)
svm_pred =model_svm.predict(X_test)
print("SVM cost time:{}s".format(time.time()-svm_start))
fpr[0], tpr[0], threshold[0] = roc_curve(y_test, svm_pred)
roc_auc[0] = auc(fpr[0], tpr[0])

#KNN with weight
from sklearn.neighbors import KNeighborsClassifier
knn_start=time.time()
model_knn=KNeighborsClassifier(n_neighbors=2,weights="distance",n_jobs=-1)
model_knn.fit(X_train,y_train)
knn_pred =model_knn.predict(X_test)
print("KNN cost time:{}s".format(time.time()-knn_start))
fpr[1], tpr[1], threshold[1] = roc_curve(y_test, knn_pred)
roc_auc[1] = auc(fpr[1], tpr[1])

# logisticregression
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
# def polynomial_model(degree=1,**kwargs):
#     polynomial_features = PolynomialFeatures(degree=degree,
#                                              include_bias=False)
#     logistic_regression = LogisticRegression(**kwargs)
#     pipline = Pipeline([("polynomial_features",polynomial_features),
#                         ("logistic_regression",logistic_regression)])
#     return pipline
model_lgr = LogisticRegression(penalty='l2',solver='sag',n_jobs=-1,random_state=42,max_iter=10000)


# BernoulliNB
from sklearn.naive_bayes import BernoulliNB
nb_start=time.time()
model_nb = BernoulliNB()
model_nb.fit(X_train,y_train)
nb_pred =model_nb.predict(X_test)
print("BernoulliNB cost time:{}s".format(time.time()-nb_start))
fpr[3], tpr[3], threshold[3] = roc_curve(y_test, nb_pred)
roc_auc[3] = auc(fpr[3], tpr[3])

# XGboost
from xgboost.sklearn import XGBClassifier
xg_start=time.time()
model_xg=XGBClassifier(booster='gbtree', nthread=-1,n_jobs=-1,objective='binary:logistic',
        learning_rate=0.05, gamma=0.1,  max_depth=8,reg_lambda= 1,subsample=0.8,colsample_bytree=1.0,
        min_child_weight=3,seed=1000)
model_xg.fit(X_train,y_train)
xg_pred =model_xg.predict(X_test)
print("XGBClassifier cost time:{}s".format(time.time()-xg_start))
fpr[4], tpr[4], threshold[4] = roc_curve(y_test, xg_pred)
roc_auc[4] = auc(fpr[4], tpr[4])

# RandomForest
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier(n_estimators=80,max_depth=17,min_samples_split=50,min_samples_leaf=10,max_features=None,
                              random_state=42,oob_score=True,n_jobs=-1)


#Multiperceptron
from sklearn.neural_network import MLPClassifier
mlp_start=time.time()
model_mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(64, 64, 8),learning_rate='constant',tol=1e-3,warm_start=True,
                    random_state=42, alpha=1e-4,early_stopping=True, shuffle=True,verbose=True,max_iter=50,n_iter_no_change=5)
model_mlp.fit(X_train,y_train)
mlp_pred =model_mlp.predict(X_test)
print("Multiperceptron cost time:{}s".format(time.time()-mlp_start))
fpr[6], tpr[6], threshold[6] = roc_curve(y_test, mlp_pred)
roc_auc[6] = auc(fpr[6], tpr[6])


# 基础模型的评价得分，mean计算均值，std（）计算的是标准偏差
score = rmsle_cv(model_svm)
print("\nGaussian Kernel SVM score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))
score = rmsle_cv(model_knn)
print("KNeighborsClassifier score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))
score = rmsle_cv(model_lgr)
print("LogisticRegression score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))
score = rmsle_cv(model_xg)
print("XGboost score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))
score = rmsle_cv(model_nb)
print("BernoulliNB score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))
score = rmsle_cv(model_rf)
print("RadomForest score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))
score = rmsle_cv(model_mlp)
print("Multiperceptron score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))


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
avg_time =time.time()
averaged_models = AveragingModels(models=(model_svm,model_knn,model_xg,model_nb,model_mlp))
averaged_models.fit(X_train,y_train)
avg_pred = averaged_models.predict(X_test)
score = rmsle_cv(averaged_models)
print(confusion_matrix(y_test, avg_pred))
print(classification_report(y_test, avg_pred))
print("AveragingModels cost time:{}s".format(time.time()-avg_time))
fpr[7], tpr[7], threshold[7] = roc_curve(y_test, avg_pred)
roc_auc[7] = auc(fpr[7], tpr[7])
print("Averaged base models score:{:.4f}(std:{:.4f})\n".format(score.mean(),score.std()))



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
# stacked_averaged_models = StackingAveragedModels(base_models=(model_knn,model_nb,model_svm,model_mlp),meta_model=(model_xg))
score=rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score:{:.4f}(std:{:.4f})".format(score.mean(),score.std()))

#均方根误差评价函数
def rmsle(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

#最后的training、prediction输出结果
#StackedClassifier
stacked_start=time.time()
stacked_averaged_models.fit(X_train,y_train)
stacked_train_pred = stacked_averaged_models.predict(X_train)
stacked_pred = stacked_averaged_models.predict(X_test)
print("StackedClassifier cost time:{}s".format(time.time()-stacked_start))
fpr[8], tpr[8], threshold[8] = roc_curve(y_test, stacked_pred)
roc_auc[8] = auc(fpr[8], tpr[8])
print(confusion_matrix(y_test, stacked_pred))
print(classification_report(y_test, stacked_pred))
print(rmsle(y_train,stacked_train_pred))

#LogisticRegression
lr_start=time.time()
model_lgr.fit(X_train,y_train)
lgr_train_pred = model_lgr.predict(X_train)
lgr_pred = model_lgr.predict(X_test)
print("LogisticRegression cost time:{}s".format(time.time()-lr_start))
fpr[2], tpr[2], threshold[2] = roc_curve(y_test, lgr_pred)
roc_auc[2] = auc(fpr[2], tpr[2])
print(rmsle(y_train,lgr_train_pred))

#RadomForest
rf_start=time.time()
model_rf.fit(X_train,y_train)
rf_train_pred = model_rf.predict(X_train)
rf_pred = model_rf.predict(X_test)
print("RadomForest cost time:{}s".format(time.time()-rf_start))
fpr[5], tpr[5], threshold[5] = roc_curve(y_test, rf_pred)
roc_auc[5] = auc(fpr[5], tpr[5])
print(rmsle(y_train,rf_train_pred))

#对三个模型集成,进行加权投票
# FinalEnsemble model
eb_start=time.time()
# model_ensemble = VotingClassifier(estimators=[('stacked',stacked_averaged_models),('lgr',model_lgr),('rf',model_rf)],voting='hard',weights=[3,2,2],n_jobs=-1)
model_ensemble = VotingClassifier(estimators=[('stacked',stacked_averaged_models),('average',averaged_models),('lgr',model_lgr),('rf',model_rf)],voting='hard',weights=[3,2,2,2])
model_ensemble.fit(X_train,y_train)
ensemble_train_pred = model_ensemble.predict(X_train)
ensemble_pred = model_ensemble.predict(X_test)
print("FinalEnsemble model cost time:{}s".format(time.time()-eb_start))
fpr[9], tpr[9], threshold[9] = roc_curve(y_test, ensemble_pred)
roc_auc[9] = auc(fpr[9], tpr[9])
print(confusion_matrix(y_test, ensemble_pred))
print(classification_report(y_test, ensemble_pred))
print(rmsle(y_train,ensemble_train_pred))


#plot ROC
for i in range(len(roc_auc)):
    plt.plot(fpr[i], tpr[i], color=color[i],lw=lw, label=labels[i] % roc_auc[i],linestyle=linestyle[i])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



