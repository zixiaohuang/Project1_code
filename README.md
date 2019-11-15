# Project1_code
This project uses various machine-learning algorithms to analyze multi-spectral images collected by UAV 

目录结构
```
.
├── MachineLearning_Algorithm
│   ├── DT
│   │   ├── DecisionTree_compare.py
│   │   ├── DecisionTree_select.py
│   │   ├── DecisionTree_select2.py
│   │   ├── DecisionTree_select3.py
│   │   ├── DecisionTree_several_parameter.py
│   │   ├── RandomForest_select.py
│   │   ├── adaboost_select.py
│   │   ├── ensemble_compare.py
│   │   └── xg1.py
│   ├── LR
│   │   ├── linearRe.py
│   │   ├── logRe.py
│   │   ├── logisticre_select.py
│   │   └── regression_compare.py
│   ├── MLP
│   │   ├── MLP_compare.py
│   │   └── MLP_new.py
│   ├── NB
│   │   └── bayes_compare.py
│   ├── knn
│   │   ├── knn_select.py
│   │   └── knnalg_compare.py
│   ├── model_combinate.py
│   └── svm
│       ├── svm_compare.py
│       ├── svm_compare2.py
│       ├── svm_compare3.py
│       ├── svmploy_select.py
│       └── svmrbf_select.py
├── Other
│   ├── MultiProcess
│   │   ├── multicore_thread.py
│   │   ├── pooltest.py
│   │   └── sharememory.py
│   └── MultiThreading
│       ├── multithreading.py
│       ├── threading_lock.py
│       └── threading_queue.py
├── Plot
│   ├── plot_dt.py
│   ├── plot_ensemble.py
│   ├── plot_evenstd.py
│   ├── plot_knn.py
│   ├── plot_lr2.py
│   ├── plot_mlp.py
│   ├── plot_nb.py
│   ├── plot_result2.py
│   ├── plot_svm.py
│   └── plot_threshold.py
├── PreProcessing
│   ├── AutoCode.py
│   ├── Correlation_Analysis
│   │   ├── correlation.py
│   │   └── plot_correlationheatmap.py
│   ├── PCA_Analysis
│   │   ├── PCA.py
│   │   ├── PCA_3Dplant.py
│   │   ├── PCA_predict.py
│   │   └── pca_select.py
│   ├── VgIndex4.py
│   ├── datamean(1)(2).py
│   ├── excel2txt.py
│   ├── feature_process.py
│   ├── multiprocess_datamean.py
│   ├── readdata.py
│   ├── sum_data.py
│   └── txt2excel.py
├── README.md
└── tree.txt

14 directories, 57 files


目录MachineLearning_Algorithm记录着决策树（目录DT）、朴素贝叶斯（目录NB）、逻辑回归（目录LR）、k近邻（目录knn）、多层感知机（目录MLP）、支持向量机（目录SVM）算法优化调参及训练代码，
目录PreProcessing记录特征预处理的过程、包括计算植被指数(datamean(1)(2).py,multiprocess_datamean.py,VgIndex4.py)、pca(目录PCA_Analysis)、autoencoder(AutoCode.py)压缩特征处理的代码
目录other中MultiProcess、MultiThreading个人学习多进程多线程的一些代码，用于加速数据处理
目录Plot是将各算法计算出的混淆矩阵各指数可视化的代码
