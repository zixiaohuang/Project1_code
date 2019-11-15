import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
MLP_acc=np.array([91.70,91.68,88.75,99.63,99.19])
MLP_spec=np.array([89.28,90.33,88.75,99.48,98.97])
MLP_f1=np.array([91.70,91.68,88.72,99.62,99.20])

plt.plot(X,MLP_acc,color='green',label='Accuracy')
for x,y in zip(X,MLP_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,MLP_spec,color='red',label='Specificity')
for x,y in zip(X,MLP_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,MLP_f1,color='skyblue',label='F1-score')
for x,y in zip(X,MLP_f1):
    plt.text(x,y-0.5,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.ylabel('Classification Result(%)')
# plt.xlabel('(a)')
plt.legend(loc='lower right')

plt.figure()
#time
lr_time=np.array([54.74,34.58,43.85,33.88,79.66])

plt.plot(X,lr_time,color='green',marker='+',label='Multi-perceptron')
for x,y in zip(X,lr_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='upper right')
plt.ylabel('Time(s)')
# plt.xlabel('(b)')
plt.show()

#kappa
kappa=np.array([83.41,83.37,77.52,99.25,98.39])

plt.plot(X,kappa,color='burlywood',marker='+',label='Multi-perceptron')
for x,y in zip(X,kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='best')
plt.ylabel('Cohen kappa(%)')
# plt.xlabel('(c)')
plt.show()