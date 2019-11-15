import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
Ada_acc=np.array([89.96,91.06,87.13,99.40,98.60])
Xg_acc=np.array([89.08,90.96,87.79,99.12,98.54])
RF_acc=np.array([90.06,91.64,88.76,98.53,97.75])

plt.plot(X,Ada_acc,color='green',marker='o',label='AdaBoost')
for x,y in zip(X,Ada_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Xg_acc,color='red',marker='o',label='XgBoost')
for x,y in zip(X,Xg_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,RF_acc,color='skyblue',marker='o',label='Random Forest')
for x,y in zip(X,RF_acc):
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
Ada_spec=np.array([88.10,89.52,85.12,99.18,97.80])
Xg_spec=np.array([86.29,88.88,84.26,98.85,97.78])
RF_spec=np.array([87.10,89.52,84.94,97.87,96.35])

plt.plot(X,Ada_spec,color='green',marker='^',label='AdaBoost')
for x,y in zip(X,Ada_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Xg_spec,color='red',marker='^',label='XgBoost')
for x,y in zip(X,Xg_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,RF_spec,color='skyblue',marker='^',label='Random Forest')
for x,y in zip(X,RF_spec):
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
Ada_f1=np.array([89.96,91.06,87.12,99.40,98.60])
Xg_f1=np.array([89.08,90.96,87.76,99.12,98.54])
RF_f1=np.array([90.05,91.64,88.73,98.53,97.75])

plt.plot(X,Ada_f1,color='green',marker='+',label='AdaBoost')
for x,y in zip(X,Ada_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Xg_f1,color='red',marker='+',label='XgBoost')
for x,y in zip(X,Xg_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,RF_f1,color='skyblue',marker='+',label='Random Forest')
for x,y in zip(X,RF_f1):
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
Ada_time=np.array([166.10,168.98,165.52,203.98,192.14])
Xg_time=np.array([20.50,135.88,22.40,150.76,153.94])
RF_time=np.array([3.25,3.06,3.41,2.93,3.15])

plt.plot(X,Ada_time,color='green',marker='*',label='AdaBoost')
for x,y in zip(X,Ada_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Xg_time,color='red',marker='*',label='XgBoost')
for x,y in zip(X,Xg_time):
    plt.text(x,y-0.6,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,RF_time,color='skyblue',marker='*',label='Random Forest')
for x,y in zip(X,RF_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='best')
plt.ylabel('Time(s)')
# plt.xlabel('(d)')
plt.show()

#kappa
Ada_kappa=np.array([79.93,82.12,74.26,98.80,97.20])
Xg_kappa=np.array([78.18,81.93,75.59,98.25,97.09])
RF_kappa=np.array([80.13,83.29,77.53,97.06,95.50])

plt.plot(X,Ada_kappa,color='green',marker='*',label='AdaBoost')
for x,y in zip(X,Ada_kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Xg_kappa,color='red',marker='*',label='XgBoost')
for x,y in zip(X,Xg_kappa):
    plt.text(x,y-0.6,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,RF_kappa,color='skyblue',marker='*',label='Random Forest')
for x,y in zip(X,RF_kappa):
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