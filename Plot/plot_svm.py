

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
linear_acc=np.array([84.25,84.27,84.38,91.64,91.35])
poly_acc=np.array([82.12,84.45,84.37,97.36,95.04])
rbf_acc=np.array([85.95,91.23,85.17,98.76,96.96])

plt.plot(X,linear_acc,color='green',marker='o',label='LinearKernel')
for x,y in zip(X,linear_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_acc,color='red',marker='o',label='PolynomialKernel')
for x,y in zip(X,poly_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_acc,color='skyblue',marker='o',label='GaussianKernel')
for x,y in zip(X,rbf_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.ylabel('Accuracy(%)')
# plt.xlabel('(a)')
plt.legend(loc='lower right')

plt.figure()
#Specificity
linear_spec=np.array([78.06,78.53,78.79,87.74,86.37])
poly_spec=np.array([73.80,80.93,76.87,96.93,91.36])
rbf_spec=np.array([80.19,88.41,78.69,98.28,94.93])

plt.plot(X,linear_spec,color='green',marker='^',label='LinearKernel')
for x,y in zip(X,linear_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_spec,color='red',marker='^',label='PolynomialKernel')
for x,y in zip(X,poly_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_spec,color='skyblue',marker='^',label='GaussianKernel')
for x,y in zip(X,rbf_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='lower right')
plt.ylabel('Specificity(%)')
# plt.xlabel('(b)')
plt.figure()
#F1Score
linear_f1=np.array([84.08,84.10,84.23,91.62,91.31])
poly_f1=np.array([81.61,84.45,84.06,97.36,95.03])
rbf_f1=np.array([85.84,91.23,84.98,98.76,96.96])

plt.plot(X,linear_f1,color='green',marker='+',label='LinearKernel')
for x,y in zip(X,linear_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_f1,color='red',marker='+',label='PolynomialKernel')
for x,y in zip(X,poly_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_f1,color='skyblue',marker='+',label='GaussianKernel')
for x,y in zip(X,rbf_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='lower right')
plt.ylabel('F1-score(%)')
# plt.xlabel('(c)')
plt.figure()
#time
linear_time=np.array([12.67,15.96,21.66,11.81,9.20])
poly_time=np.array([15.74,48.90,59.01,8.98,7.27])
rbf_time=np.array([15.33,12.06,21.79,2.214,6.13])

plt.plot(X,linear_time,color='green',marker='*',label='LinearKernel')
for x,y in zip(X,linear_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_time,color='red',marker='*',label='PolynomialKernel')
for x,y in zip(X,poly_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_time,color='skyblue',marker='*',label='GaussianKernel')
for x,y in zip(X,rbf_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='upper right')
plt.ylabel('Time(s)')
# plt.xlabel('(d)')
plt.show()

#kappa
linear_kappa=np.array([84.25,68.50,68.74,83.27,82.71])
poly_kappa=np.array([64.32,68.99,68.73,94.72,90.08])
rbf_kappa=np.array([71.94,82.47,70.33,97.51,93.92])

plt.plot(X,linear_kappa,color='green',marker='*',label='LinearKernel')
for x,y in zip(X,linear_kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,poly_kappa,color='red',marker='*',label='PolynomialKernel')
for x,y in zip(X,poly_kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,rbf_kappa,color='skyblue',marker='*',label='GaussianKernel')
for x,y in zip(X,rbf_kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='best')
plt.ylabel('Cohen kappa(%)')
# plt.xlabel('(e)')
plt.show()