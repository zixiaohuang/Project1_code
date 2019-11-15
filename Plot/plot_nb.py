import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])

plt.figure()
#准确率accuracy
Gau_acc=np.array([81.29,83.03,77.13,87.79,83.12])
Multi_acc=np.array([78.35,75.99,75.68,83.79,78.24])
Ber_acc=np.array([72.79,83.36,82.98,85.90,84.76])

plt.plot(X,Gau_acc,color='green',marker='o',label='Gaussian NB')
for x,y in zip(X,Gau_acc):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Multi_acc,color='red',marker='o',label='Polynomial NB')
for x,y in zip(X,Multi_acc):
    plt.text(x,y-0.7,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Ber_acc,color='skyblue',marker='o',label='Bernoulli NB')
for x,y in zip(X,Ber_acc):
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
Gau_spec=np.array([77.16,77.51,69.75,84.06,75.90])
Multi_spec=np.array([81.67,70.78,67.38,76.79,69.60])
Ber_spec=np.array([86.74,76.54,78.91,79.86,81.47])

plt.plot(X,Gau_spec,color='green',marker='^',label='Gaussian NB')
for x,y in zip(X,Gau_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Multi_spec,color='red',marker='^',label='Polynomial NB')
for x,y in zip(X,Multi_spec):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Ber_spec,color='skyblue',marker='^',label='Bernoulli NB')
for x,y in zip(X,Ber_spec):
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
Gau_f1=np.array([81.16,83.03,76.26,88.74,82.81])
Multi_f1=np.array([78.30,75.57,74.14,83.51,77.18])
Ber_f1=np.array([71.84,83.06,82.89,85.75,84.72])

plt.plot(X,Gau_f1,color='green',marker='+',label='Gaussian NB')
for x,y in zip(X,Gau_f1):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Multi_f1,color='red',marker='+',label='Polynomial NB')
for x,y in zip(X,Multi_f1):
    plt.text(x,y-0.6,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Ber_f1,color='skyblue',marker='+',label='Bernoulli NB')
for x,y in zip(X,Ber_f1):
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
Gau_time=np.array([0.009,0.01,0.02,0.009,0.01])
Multi_time=np.array([0.01,0.02,0.02,0.009,0.019])
Ber_time=np.array([0.01,0.019,0.019,0.024,0.016])

plt.plot(X,Gau_time,color='green',marker='*',label='Gaussian NB')
for x,y in zip(X,Gau_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Multi_time,color='red',marker='*',label='Polynomial NB')
for x,y in zip(X,Multi_time):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Ber_time,color='skyblue',marker='*',label='Bernoulli NB')
for x,y in zip(X,Ber_time):
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
Gau_kappa=np.array([62.52,66.03,54.19,77.59,66.28])
Multi_kappa=np.array([78.35,51.92,51.27,67.58,56.54])
Ber_kappa=np.array([45.74,66.68,65.95,71.80,69.53])

plt.plot(X,Gau_kappa,color='green',marker='*',label='Gaussian NB')
for x,y in zip(X,Gau_kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Multi_kappa,color='red',marker='*',label='Polynomial NB')
for x,y in zip(X,Multi_kappa):
    plt.text(x,y,y,ha='center',va='bottom',fontsize=8)
plt.plot(X,Ber_kappa,color='skyblue',marker='*',label='Bernoulli NB')
for x,y in zip(X,Ber_kappa):
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