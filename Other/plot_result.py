# 将实验结果可视化
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from matplotlib.legend_handler import HandlerTuple

# 使用plt.gca获取当前坐标轴信息
# plt.legend(handles=[变量名1，变量名2],labels=[对应名称1，对应名称2],loc=)

plt.figure()
# n=3
X = np.array([0.3,1.8,3.3]) #X是1,2,3 分别对应linearsvm,polysvm,rbfsvm
ori_acc= np.array([84.74,82.78,86.58])#originaldata
pca_acc = np.array([84.16,85.66,91.48])#pcadata
encoder_acc=np.array([85.24,89.06,91.72])#encoder
pcaori_acc=np.array([94.32,96.92,98.76])#pcaori
encoderori_acc=np.array([94.64,97.66,98.60])
#上面准确率的柱状图
#acc
l11=plt.bar(X,ori_acc,alpha=0.9,width=0.2,color='lightcoral',label='Original Data')
for x,y in zip(X,ori_acc):
    plt.text(x,y-50,'%.2f'%y,fontsize=6,ha='center',va='center')
plt.bar(X,pca_acc,alpha=0.9,width=0.2,bottom=ori_acc,color='indianred',label='PCA Data')
for x,y,z in zip(X,pca_acc+ori_acc,pca_acc):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X,encoder_acc,alpha=0.9,width=0.2,bottom=ori_acc+pca_acc,color='brown',label='Encoder Data')
for x,y,z in zip(X,encoder_acc+pca_acc+ori_acc,encoder_acc):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X,pcaori_acc,alpha=0.9,width=0.2,bottom=ori_acc+pca_acc+encoder_acc,color='maroon',label='PCA+Original Data')
for x,y,z in zip(X,pcaori_acc+encoder_acc+pca_acc+ori_acc,pcaori_acc):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X,encoderori_acc,alpha=0.9,width=0.2,bottom=ori_acc+pca_acc+encoder_acc+pcaori_acc,color='darkred',label='Encoder+Original Data')
for x,y,z in zip(X,encoderori_acc+pcaori_acc+encoder_acc+pca_acc+ori_acc,encoderori_acc):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')

#spec
ori_spec=np.array([78.52,74.31,80.97])#originaldata
pca_spec = np.array([78.79,83.10,88.12])#pcadata
encoder_spec=np.array([77.57,87.20,88.34])#encoder
pcaori_spec=np.array([91.00,95.91,98.28])#pcaori
encoderori_spec=np.array([91.79,97.12,97.97])

l12=plt.bar(X+0.2,ori_spec,alpha=0.9,width=0.2,color='peachpuff',label='Original Data')
for x,y in zip(X+0.2,ori_spec):
    plt.text(x,y-50,'%.2f'%y,fontsize=6,ha='center',va='center')
plt.bar(X+0.2,pca_spec,alpha=0.9,width=0.2,bottom=ori_spec,color='bisque',label='PCA Data')
for x,y,z in zip(X+0.2,pca_spec+ori_spec,pca_spec):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.2,encoder_spec,alpha=0.9,width=0.2,bottom=ori_spec+pca_spec,color='lightsalmon',label='Encoder Data')
for x,y,z in zip(X+0.2,encoder_spec+pca_spec+ori_spec,encoder_spec):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.2,pcaori_spec,alpha=0.9,width=0.2,bottom=ori_spec+pca_spec+encoder_spec,color='coral',label='PCA+Original Data')
for x,y,z in zip(X+0.2,pcaori_spec+encoder_spec+pca_spec+ori_spec,pcaori_spec):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.2,encoderori_spec,alpha=0.9,width=0.2,bottom=ori_spec+pca_spec+encoder_spec+pcaori_spec,color='orangered',label='Encoder+Original Data')
for x,y,z in zip(X+0.2,encoderori_spec+pcaori_spec+encoder_spec+pca_spec+ori_spec,pcaori_spec):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')

# precision
ori_pre=np.array([86.29,86.79,87.70])#originaldata
pca_pre = np.array([85.29,85.85,91.78])#pcadata
encoder_pre=np.array([87.94,89.14,92.02])#encoder
pcaori_pre=np.array([94.61,96.94,98.76])#pcaori
encoderori_pre=np.array([94.91,97.67,98.61])

l13=plt.bar(X+0.4,ori_pre,alpha=0.9,width=0.2,color='papayawhip',label='Original Data')
for x,y in zip(X+0.4,ori_pre):
    plt.text(x,y-50,'%.2f'%y,fontsize=6,ha='center',va='center')
plt.bar(X+0.4,pca_pre,alpha=0.9,width=0.2,bottom=ori_pre,color='moccasin',label='PCA Data')
for x,y,z in zip(X+0.4,pca_pre+ori_pre,pca_pre):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.4,encoder_pre,alpha=0.9,width=0.2,bottom=ori_pre+pca_pre,color='wheat',label='Encoder Data')
for x,y,z in zip(X+0.4,encoder_pre+pca_pre+ori_pre,encoder_pre):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.4,pcaori_pre,alpha=0.9,width=0.2,bottom=ori_pre+pca_pre+encoder_pre,color='goldenrod',label='PCA+Original Data')
for x,y,z in zip(X+0.4,pcaori_pre+encoder_pre+pca_pre+ori_pre,pcaori_pre):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.4,encoderori_pre,alpha=0.9,width=0.2,bottom=ori_pre+pca_pre+encoder_pre+pcaori_pre,color='darkgoldenrod',label='Encoder+Original Data')
for x,y,z in zip(X+0.4,encoderori_pre+pcaori_pre+encoder_pre+pca_pre+ori_pre,pcaori_pre):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')

# recall
ori_recall=np.array([84.74,82.78,86.58])#originaldata
pca_recall = np.array([84.16,85.66,91.48])#pcadata
encoder_recall=np.array([85.24,89.06,91.72])#encoder
pcaori_recall=np.array([94.32,96.92,98.76])#pcaori
encoderori_recall=np.array([94.58,97.66,98.60])

l14=plt.bar(X+0.6,ori_recall,alpha=0.9,width=0.2,color='greenyellow',label='Original Data')
for x,y in zip(X+0.6,ori_recall):
    plt.text(x,y-50,'%.2f'%y,fontsize=6,ha='center',va='center')
plt.bar(X+0.6,pca_recall,alpha=0.9,width=0.2,bottom=ori_recall,color='lawngreen',label='PCA Data')
for x,y,z in zip(X+0.6,pca_recall+ori_recall,pca_recall):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.6,encoder_recall,alpha=0.9,width=0.2,bottom=ori_recall+pca_recall,color='forestgreen',label='Encoder Data')
for x,y,z in zip(X+0.6,encoder_recall+pca_recall+ori_recall,encoder_recall):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.6,pcaori_recall,alpha=0.9,width=0.2,bottom=ori_recall+pca_recall+encoder_recall,color='olivedrab',label='PCA+Original Data')
for x,y,z in zip(X+0.6,pcaori_recall+encoder_recall+pca_recall+ori_recall,pcaori_recall):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.6,encoderori_recall,alpha=0.9,width=0.2,bottom=ori_recall+pca_recall+encoder_recall+pcaori_recall,color='darkolivegreen',label='Encoder+Original Data')
for x,y,z in zip(X+0.6,encoderori_recall+pcaori_recall+encoder_recall+pca_recall+ori_recall,pcaori_recall):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')

#f1-score
ori_f1=np.array([84.61,82.36,86.51])#originaldata
pca_f1 = np.array([84.05,85.65,91.47])#pcadata
encoder_f1=np.array([85.01,89.06,91.71])#encoder
pcaori_f1=np.array([94.31,96.92,98.76])#pcaori
encoderori_f1=np.array([94.63,97.66,98.60])

l15=plt.bar(X+0.8,ori_f1,alpha=0.9,width=0.2,color='lightcyan',label='Original Data')
for x,y in zip(X+0.8,ori_f1):
    plt.text(x,y-50,'%.2f'%y,fontsize=6,ha='center',va='center')
plt.bar(X+0.8,pca_recall,alpha=0.9,width=0.2,bottom=ori_f1,color='paleturquoise',label='PCA Data')
for x,y,z in zip(X+0.8,pca_f1+ori_f1,pca_f1):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.8,encoder_f1,alpha=0.9,width=0.2,bottom=ori_f1+pca_f1,color='c',label='Encoder Data')
for x,y,z in zip(X+0.8,encoder_f1+pca_f1+ori_f1,encoder_f1):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.8,pcaori_f1,alpha=0.9,width=0.2,bottom=ori_f1+pca_f1+encoder_f1,color='teal',label='PCA+Original Data')
for x,y,z in zip(X+0.8,pcaori_f1+encoder_f1+pca_f1+ori_f1,pcaori_f1):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')
plt.bar(X+0.8,encoderori_f1,alpha=0.9,width=0.2,bottom=ori_f1+pca_f1+encoder_f1+pcaori_f1,color='darkslategray',label='Encoder+Original Data')
for x,y,z in zip(X+0.8,encoderori_f1+pcaori_f1+encoder_f1+pca_f1+ori_f1,pcaori_f1):
    plt.text(x,y-50,'%.2f'%z,fontsize=6,ha='center',va='center')

#time
X2 = np.array([0.1,1.6,3.1])
ori_time=np.array([35.42,47.07,52.52])
pca_time=np.array([58.1,60.9,63.2])
encoder_time=np.array([71.3,146.2,63.8])
pcaori_time=np.array([32.8,27.7,13.5])
encoderori_time=np.array([38.4,28.1,17.9])

# 下面表示时间的柱状图
l16=plt.bar(X2,-ori_time,alpha=0.9,width=0.2,color='lightpink',label='Original Data')
for x,y in zip(X2,-ori_time):
    plt.text(x,y+10,'%.2f'%(-y),fontsize=6,ha='center',va='center')
plt.bar(X2,-pca_time,alpha=0.9,width=0.2,bottom=-ori_time,color='pink',label='PCA Data')
for x,y,z in zip(X2,-ori_time-pca_time,-pca_time):
    plt.text(x,y+20,'%.2f'%(-z),fontsize=6,ha='center',va='center')
plt.bar(X2,-encoder_time,alpha=0.9,width=0.2,bottom=-ori_time-pca_time,color='deeppink',label='Encoder Data')
for x,y,z in zip(X2,-encoder_time-ori_time-pca_time,-encoder_time):
    plt.text(x,y+30,'%.2f'%(-z),fontsize=6,ha='center',va='center')
plt.bar(X2,-pcaori_time,alpha=0.9,width=0.2,bottom=-ori_time-pca_time-encoder_time,color='mediumvioletred',label='PCA+Original Data')
for x,y,z in zip(X2,-pcaori_time-encoder_time-ori_time-pca_time,-pcaori_time):
    plt.text(x,y+10,'%.2f'%(-z),fontsize=6,ha='center',va='center')
plt.bar(X2,-encoderori_time,alpha=0.9,width=0.2,bottom=-ori_time-pca_time-encoder_time-pcaori_time,color='purple',label='Encoder+Original Data')
for x,y,z in zip(X2,-encoderori_time-pcaori_time-encoder_time-ori_time-pca_time,-encoderori_time):
    plt.text(x,y+10,'%.2f'%(-z),fontsize=6,ha='center',va='center')
#获取坐标
ax=plt.gca()
#隐藏上面坐标
ax.spines['top'].set_color('none')
#移动下面坐标至y=0
ax.spines['bottom'].set_position(('data',0))

#设置x坐标字体大小，名称
plt.xticks(fontsize=8)
plt.xticks([0.3,0.5,0.7,0.9,1.1,1.8,2.0,2.2,2.4,2.6,3.3,3.5,3.7,3.9,4.1],
           [r'Accuracy',r'Specificity',r'Precision',r'Recall',r'F1score',r'Accuracy',r'Specificity',r'Precision',r'Recall',r'F1score',r'Accuracy',r'Specificity',r'Precision',r'Recall',r'F1score'])
# 旋转
pl.xticks(rotation=90)
# plt.xticks([0.1,1.6,3.2],[r'time',r'time',r'time'],rotation=180)
# plt.yticks([-100,-200,-300],[r'100',r'200',r'300'])

# plt.text(0.7,-400,r'$SVM\ Linear\\n kernel$')
plt.annotate('SVM Linear\n     kernel',xy=(0.3,-300))
plt.annotate('SVM Polynomial\n         kernel',xy=(1.8,-300))
plt.annotate('SVM Gaussian\n       kernel',xy=(3.3,-300))

# plt.annotate('Classification \nPerformances(%)',xy=(-0.5,200))
# plt.annotate('Time(s)',xy=(-0.5,-100))
plt.text(-0.9,250,'Result\n  (%)')
plt.text(-0.9,-200,'Time\n  (s)')
label=["Original Data","PCA Data","Encoder Data","PCA+Original Data","Encoder+Original Data"]

# plt.legend(label,loc=4)
plt.show()























