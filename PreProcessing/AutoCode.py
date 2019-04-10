from keras.models import Model
from keras.layers import Dense, Input,Dropout
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest #选择相关性最大的三个特征绘图
import random

def normalization(dataMat):
    meanVals = np.mean(dataMat, axis=1)  # axis =1 压缩列，对各行求均值 axis=0 压缩行
    meanRemoved = dataMat - meanVals
    rowMax = dataMat.max(axis=1)  # 获取每行的最大值
    rowMin = dataMat.min(axis=1)  # 获取每行的最小值
    rowDiff = rowMax - rowMin
    normalVals = meanRemoved / rowDiff  # 归一化处理
    return normalVals

with open('new_alldata','rb') as f:
    data=pickle.load(f)
# data=data.sample(n=1000,axis=0)
X = preprocessing.scale(data.drop(labels=['TVI','MCARI1','OSAVI','IPVI','GRNDVI','label'], axis=1))
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_select, X_test_select, y_train_select, y_test_select = train_test_split(X_test, y_test, test_size=10000, random_state=42) # select sample to plot
# X_train_tenosr = tf.convert_to_tensor(X_train)
# X_test_tenosr = tf.convert_to_tensor(X_test)
# y_train_tenosr = tf.convert_to_tensor(y_train)
# y_test_tenosr = tf.convert_to_tensor(y_test)

#model
#in order to plot a 3D figure
encoding_dim = 10

# this is our input placeholder
input_data = Input(shape=(15,))

# data_drop = Dropout(0.2)(input_data)
# encoder layers
encoded = Dense(13,activation='relu')(input_data)
#encoded = Dropout(rate=0.2)(encoded)
# encoded = Dense(15,activation='relu')(encoded)
# encoded = Dense(13,activation='relu')(encoded)
encoded = Dense(11,activation='relu')(encoded)
#encoded = Dropout(rate=0.1)(encoded)
#encoded = Dense(8,activation='relu')(encoded)
#encoded = Dropout(rate=0.2)(encoded)
#encoded = Dense(6,activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)

#decoder layers
decoded = Dense(11,activation='relu')(encoded_output)
#decoded = Dense(8,activation='relu')(decoded)
#decoded = Dense(10,activation='relu')(decoded)
# decoded = Dense(13,activation='relu')(decoded)
# decoded = Dense(15,activation='relu')(decoded)
decoded = Dense(13,activation='relu')(decoded)
decoded = Dense(15,activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(inputs = input_data,outputs=decoded)

# construct the encoder model for plotting
encoder =Model(inputs = input_data, outputs=encoded_output)

# compile autoencoder
autoencoder.compile(optimizer='adam',loss='mse')

#trainig
autoencoder.fit(X_train,X_train,epochs=20,batch_size=5000,shuffle=True)

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
# plt.scatter(encoded_data[:,0],encoded_data[:,1],encoded_data[:,2],c=y_test)
# plt.colorbar()
# plt.show()

fig = plt.figure()
ax=Axes3D(fig)
ax.set_title('Datasets after AutoEncoder')
ax.scatter(X_new[y_test_select==1][:,0],X_new[y_test_select==1][:,1],X_new[y_test_select==1][:,2],c='g',s=10,marker='o')
ax.scatter(X_new[y_test_select==-1][:,0],X_new[y_test_select==-1][:,1],X_new[y_test_select==-1][:,2],c='r', s=10,marker='^')
ax.set_zlabel('feature3')  # 坐标轴
ax.set_ylabel('feature2')
ax.set_xlabel('feature1')
plt.show()