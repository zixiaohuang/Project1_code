import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
DT_acc=np.array([88.43,89.19,85.91,98.13,97.55])
DT_spec=np.array([88.01,88.94,85.13,98.07,97.22])
DT_f1=np.array([88.43,89.19,85.91,98.13,97.55])

plt.plot(X,DT_acc,color='green',label='Accuracy')
for x,y in zip(X,DT_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,DT_spec,color='red',label='Specificity')
for x,y in zip(X,DT_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,DT_f1,color='skyblue',label='F1-score')
for x,y in zip(X,DT_f1):
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
lr_time=np.array([0.4,0.93,1.39,1.46,1.71])

plt.plot(X,lr_time,color='green',marker='+',label='Decision Tree')
for x,y in zip(X,lr_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='lower right')
plt.ylabel('Time(s)')
# plt.xlabel('(b)')
plt.show()

#kappa
kappa=np.array([76.86,78.39,71.82,96.25,95.10])

plt.plot(X,kappa,color='burlywood',marker='+',label='Decision Tree')
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