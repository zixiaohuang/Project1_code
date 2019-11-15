'''
优化模型参数——svm多项式核函数对比参数
'''
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# train_sizes=np.linspace(.1,1.0,5)表示把训练样本数量从0.1~1分成五等分，生成[0.1,0.325,0.55,0.775,1]
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
    print("plot start")
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
    print("plot end")
    return plt

if __name__ == "__main__":
    # 读取数据
    '''
    all_data= pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                             names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                    'soil_reg_veg_index','sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                    'norm_green', 'cvi', 'green_red_ndvi','label'])
    with open("all_data", 'wb') as f:
        pickle.dump(all_data, f)
    '''
    with open("E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_tune", 'rb') as f:
        all_data = pickle.load(f)

    all_dataselect = all_data.sample(n=50000,axis=0)
    y = np.array(all_dataselect['label']).ravel()
    X = all_dataselect.drop('label', axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)  # 选部分样本进行计算
    cv = ShuffleSplit(n_splits=5,test_size= 0.2, random_state=0)
    title = 'Learning Curves with degree={0}'
    degrees = [1,2,3]

    start = time.clock()
    plt.figure(figsize=(12,4),dpi=144)
    print('开始循环')

    for i in range(len(degrees)):
        print('第{}项式svm开始循环'.format(i+1))
        begin=time.time()
        plt.subplot(1,len(degrees),i+1)
        plot_learning_curve(SVC(C=1.0, kernel='poly',degree=degrees[i],gamma='auto',shrinking=True),
                            title.format(degrees[i]),
                            X,y,ylim=(0.8,1.01),cv=cv,n_jobs=-1)
        print('第{}项式svm循环一次的时间'.format(i+1),time.time()-begin)
    plt.savefig("E://txt.png")
    plt.show()