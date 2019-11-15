import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
lr_acc=np.array([83.91,83.87,84.89,91.72,91.74])
lr_spec=np.array([79.40,79.48,79.19,88.72,88.15])
lr_f1=np.array([83.84,83.81,84.77,91.71,91.72])

plt.plot(X,lr_acc,color='green',label='Accuracy')
for x,y in zip(X,lr_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,lr_spec,color='red',label='Specificity')
for x,y in zip(X,lr_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,lr_f1,color='skyblue',label='F1-score')
for x,y in zip(X,lr_f1):
    plt.text(x,y-0.5,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.ylabel('Classification Result(%)')
plt.legend(loc='lower right')
# plt.xlabel('(a)')
plt.figure()
#time
lr_time=np.array([0.314,0.314,5.995,0.413,6.914])

plt.plot(X,lr_time,color='green',marker='+',label='LogisticRegression')
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
lr_kappa=np.array([67.88,67.80,69.82,83.45,83.47])

plt.plot(X,lr_kappa,color='burlywood',marker='+',label='LogisticRegression')
for x,y in zip(X,lr_kappa):
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