import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
linear_acc=np.array([84.74,84.16,85.24,94.32,94.64])
poly_acc=np.array([82.78,85.66,89.06,96.92,97.66])
rbf_acc=np.array([86.58,91.48,91.72,98.76,98.60])

plt.plot(X,linear_acc,color='green',marker='o',label='LinearAccuracy')
for x,y in zip(X,linear_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_acc,color='red',marker='o',label='PolynomialAccuracy')
for x,y in zip(X,poly_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_acc,color='skyblue',marker='o',label='GaussianAccuracy')
for x,y in zip(X,rbf_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()

#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])

plt.figure()
#Specificity
linear_spec=np.array([78.52,78.79,77.57,91.00,91.79])
poly_spec=np.array([74.31,83.10,87.20,95.91,97.12])
rbf_spec=np.array([80.97,88.12,88.34,98.28,97.97])

plt.plot(X,linear_spec,color='green',marker='^',label='LinearAccuracy')
for x,y in zip(X,linear_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_spec,color='red',marker='^',label='PolynomialAccuracy')
for x,y in zip(X,poly_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_spec,color='skyblue',marker='^',label='GaussianAccuracy')
for x,y in zip(X,rbf_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()

#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])

plt.figure()
#Precision
linear_pre=np.array([86.29,85.29,87.94,94.61,94.91])
poly_pre=np.array([86.79,85.85,89.14,96.94,97.67])
rbf_pre=np.array([87.70,91.78,92.02,98.76,98.61])

plt.plot(X,linear_pre,color='green',marker='x',label='LinearAccuracy')
for x,y in zip(X,linear_pre):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_pre,color='red',marker='x',label='PolynomialAccuracy')
for x,y in zip(X,poly_pre):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_pre,color='skyblue',marker='x',label='GaussianAccuracy')
for x,y in zip(X,rbf_pre):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#Recall
linear_recall=np.array([84.74,84.16,85.24,94.32,94.58])
poly_recall=np.array([82.78,85.66,89.06,96.92,97.66])
rbf_recall=np.array([86.58,91.48,91.72,98.76,98.60])

plt.plot(X,linear_recall,color='green',marker='*',label='LinearAccuracy')
for x,y in zip(X,linear_recall):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_recall,color='red',marker='*',label='PolynomialAccuracy')
for x,y in zip(X,poly_recall):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_recall,color='skyblue',marker='*',label='GaussianAccuracy')
for x,y in zip(X,rbf_recall):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#F1Score
linear_f1=np.array([84.61,84.05,85.01,94.31,94.63])
poly_f1=np.array([82.36,85.65,89.06,96.92,97.66])
rbf_f1=np.array([86.51,91.47,91.71,98.76,98.60])

plt.plot(X,linear_f1,color='green',marker='+',label='LinearAccuracy')
for x,y in zip(X,linear_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_f1,color='red',marker='+',label='PolynomialAccuracy')
for x,y in zip(X,poly_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_f1,color='skyblue',marker='+',label='GaussianAccuracy')
for x,y in zip(X,rbf_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()

#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.show()
