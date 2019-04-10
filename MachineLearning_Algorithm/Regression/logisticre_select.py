import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    with open('E:\\VgIndex2.py\\数据\\auto_code', 'rb') as f1:
        code_data = pickle.load(f1)
    code_data = code_data.sample(20000,axis=0)
    X = code_data.drop('label', axis=1)
    y = code_data['label']
    # X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    solvers1=['newton-cg','lbfgs','liblinear','sag','saga']
    warm_starts = ['True','False']
    param_grid1 = {'penalty':['l2'],'solver':solvers1,'warm_start':warm_starts,'max_iter':[10000],'random_state':[42],'n_jobs':[-1]}
    clf = GridSearchCV(LogisticRegression(), param_grid1, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("best param: {0}\n best score: {1}".format(clf.best_params_, clf.best_score_))
    param_grid2 = {'penalty': ['l1'], 'solver': ['liblinear'], 'warm_start': warm_starts,'max_iter':[10000], 'random_state': [42],'n_jobs': [-1]}
    clf2 = GridSearchCV(LogisticRegression(), param_grid2, cv=5, n_jobs=-1)
    clf2.fit(X_train, y_train)
    print("best param: {0}\n best score: {1}".format(clf2.best_params_, clf2.best_score_))

