from keras.models import Model
from keras.layers import Dense, Input,Dropout
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest #选择相关性最大的三个特征绘图
from keras.callbacks import TensorBoard


def normalization(dataMat):
    meanVals = np.mean(dataMat, axis=1)  # axis =1 压缩列，对各行求均值 axis=0 压缩行
    meanRemoved = dataMat - meanVals
    rowMax = dataMat.max(axis=1)  # 获取每行的最大值
    rowMin = dataMat.min(axis=1)  # 获取每行的最小值
    rowDiff = rowMax - rowMin
    normalVals = meanRemoved / rowDiff  # 归一化处理
    return normalVals

with open('E:\\modifiedversion\\Datasets\\new_alldata','rb') as f:
    data=pickle.load(f)
X = preprocessing.scale(data.drop(labels=['TVI','MCARI1','DVI','OSAVI','IPVI','GRNDVI','label'], axis=1))
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_select, X_test_select, y_train_select, y_test_select = train_test_split(X_test, y_test, test_size=10000, random_state=42) # select sample to plot

#model
#in order to plot a 3D figure
encoding_dim = 3

# this is our input placeholder
input_data = Input(shape=(14,))
data_drop = Dropout(0.2)(input_data)
# encoder layers
encoded = Dense(12,activation='relu')(input_data)
encoded = Dropout(rate=0.1)(encoded)
encoded = Dense(10,activation='relu')(encoded)
# encoded = Dropout(rate=0.1)(encoded)
encoded = Dense(8,activation='relu')(encoded)
#encoded = Dropout(rate=0.2)(encoded)
encoded = Dense(6,activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)

#decoder layers
decoded = Dense(6,activation='relu')(encoded_output)
decoded = Dense(8,activation='relu')(decoded)
decoded = Dense(10,activation='relu')(decoded)
decoded = Dense(12,activation='relu')(decoded)
decoded = Dense(14,activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(inputs = input_data,outputs=decoded)

# construct the encoder model for plotting
encoder =Model(inputs = input_data, outputs=encoded_output)

# compile autoencoder
autoencoder.compile(optimizer='adam',loss='mse')

tbCallBack=TensorBoard(log_dir='./logs',#log 目录
                       histogram_freq=0,# 按照何等频率（epoch）来计算直方图，0为不计算
                       write_graph=True,# 是否存储网络结构图
                       write_grads=True,# 是否可视化梯度直方图
                       write_images=True,# 是否可视化参数
                       embeddings_freq=0,
                       embeddings_layer_names=None,
                       embeddings_metadata=None)

#trainig
autoencoder.fit(X_train,X_train,epochs=20,batch_size=5000,shuffle=True,callbacks=[tbCallBack])

#plotting tf.cast(X_test_tenosr,tf.float64)
code_data = encoder.predict(X)
df = pd.DataFrame(code_data)
#df = df.append(y)
y_reset = y.reset_index(drop=True)
df =pd.concat([df,y_reset],axis=1)
with open('auto_code','wb') as f:
    pickle.dump(df,f)
encoded_data = encoder.predict(X_test_select)
selector = SelectKBest(k=3)
X_new = selector.fit_transform(encoded_data,y_test_select)

fig = plt.figure()
ax=Axes3D(fig)
ax.set_title('Datasets after AutoEncoder')
ax.scatter(X_new[y_test_select==1][:,0],X_new[y_test_select==1][:,1],X_new[y_test_select==1][:,2],c='g',s=10,marker='o')
ax.scatter(X_new[y_test_select==-1][:,0],X_new[y_test_select==-1][:,1],X_new[y_test_select==-1][:,2],c='r', s=10,marker='^')
ax.set_zlabel('feature3')  # 坐标轴
ax.set_ylabel('feature2')
ax.set_xlabel('feature1')
plt.show()