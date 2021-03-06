'''
adaboost优化参数
'''

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    with open("E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_tune",'rb') as f:
        all_data = pickle.load(f)
    all_data=all_data.sample(n=60000,axis=0)
    X = all_data.drop('label', axis=1)
    y = all_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    bdt = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=19, min_impurity_decrease=0.0, min_samples_leaf=3,min_samples_split=6),algorithm='SAMME.R',random_state=42)
    param_grid = {'n_estimators':range(50,500,30),"learning_rate":[0.001,0.01,0.1,0.4,0.8,1]}
    clf = GridSearchCV(bdt, param_grid, cv=5,n_jobs=-1)
    clf.fit(X_train,y_train)
    print("best param:{0}\nbest_score:{1}".format(clf.best_params_, clf.best_score_))
