import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
uniform_acc=np.array([90.93,91.47,88.51,99.07,98.16])
distance_acc=np.array([90.79,91.25,88.25,99.09,98.24])

plt.plot(X,uniform_acc,color='green',marker='o',label='UniformWeight')
for x,y in zip(X,uniform_acc):
    plt.text(x,y-0.3,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,distance_acc,color='red',marker='o',label='DistanceWeight')
for x,y in zip(X,distance_acc):
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
uniform_spec=np.array([88.26,89.08,85.65,98.87,97.00])
distance_spec=np.array([88.32,89.11,85.81,98.89,97.12])

plt.plot(X,uniform_spec,color='green',marker='^',label='UniformWeight')
for x,y in zip(X,uniform_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,distance_spec,color='red',marker='^',label='DistanceWeight')
for x,y in zip(X,distance_spec):
    plt.text(x,y-0.4,y,ha='center',va='bottom',fontsize=8)

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
uniform_f1=np.array([90.92,91.47,88.50,99.07,98.16])
distance_f1=np.array([90.78,91.25,88.24,99.10,98.24])

plt.plot(X,uniform_f1,color='green',marker='+',label='UniformWeight')
for x,y in zip(X,uniform_f1):
    plt.text(x,y-0.3,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,distance_f1,color='red',marker='+',label='DistanceWeight')
for x,y in zip(X,distance_f1):
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
uniform_time=np.array([0.03,0.024,0.027,0.03,0.042])
distance_time=np.array([0.02,0.029,0.029,0.031,0.031])

plt.plot(X,uniform_time,color='green',marker='*',label='UniformWeight')
for x,y in zip(X,uniform_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,distance_time,color='red',marker='*',label='DistanceWeight')
for x,y in zip(X,distance_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)

#获取坐标
ax=plt.gca()
#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([1,2,3,4,5],
           ['Original \n data','PCA \n data','AutoEncoder \n data','PCA\n+Original','AutoEncoder\n+Original'])
plt.legend(loc='lower right')
plt.ylabel('Time(s)')
# plt.xlabel('(d)')
plt.show()

#kappa
uniform_kappa=np.array([81.85,82.94,77.03,98.13,96.32])
distance_kappa=np.array([81.57,82.50,76.50,98.18,96.47])

plt.plot(X,uniform_kappa,color='green',marker='*',label='UniformWeight')
for x,y in zip(X,uniform_kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,distance_kappa,color='red',marker='*',label='DistanceWeight')
for x,y in zip(X,distance_kappa):
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
