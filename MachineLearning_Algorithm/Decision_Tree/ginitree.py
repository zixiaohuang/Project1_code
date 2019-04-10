import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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
    all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                             names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                    'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                    'norm_green', 'cvi', 'green_red_ndvi', 'label'])

    all_dataselect = all_data.drop(
        labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1)
    pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X = all_dataselect.drop('label', axis=1)
    X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)
    # thresholds = np.linspace(0,0.3,50)
    max_depth = range(2,15)
    # 设置参数矩阵
    # param_grid =[{'criterion':['gini'],'max_depth':[9],'min_impurity_decrease':thresholds,'min_samples_split':[4]}]
    param_grid = [{'criterion': ['gini'],'max_depth':max_depth,'min_impurity_decrease': [0.1],'min_samples_split':[4]}]
    clf = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5,n_jobs=-1)
    clf.fit(X,y)
    print("best param:{0}\nbest score:{1}".format(clf.best_params_,clf.best_score_))
    # plot_curve(thresholds,clf.cv_results_,xlabel='gini thresholds')
    plot_curve(max_depth, clf.cv_results_, xlabel='max depth of decision tree')