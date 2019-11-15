'''处理原始数据、原始数据+pca、原始数据+autoencoder'''
import pickle
import pandas as pd
from sklearn import preprocessing

# #处理原始反射率数据
# all_data = pd.read_table("E:\\combinate_newdata\\oringal_alldata.txt", sep=" ",
#                             names=['NIR','Red','Green','label'])
#
# y=all_data['label']
# X = preprocessing.scale(all_data.drop(labels=['label'], axis=1))
# y_reset = y.reset_index(drop=True)
# y_reset = pd.DataFrame(y_reset)
# X = pd.DataFrame(X)
# df =pd.concat([X,y_reset],axis=1)
#
# with open("all_originaldata", 'wb') as f:
#     pickle.dump(df, f)

# with open('all_originaldata','rb') as f1:
#      data=pickle.load(f1)
# y=data['label'] #保留标签用于合并
# y=pd.DataFrame(y)
# data=data.drop(labels=['label'],axis=1)
# data.columns = ['NIR','Red','Green'] #重新命名防止重复
# with open('new_alldata','rb') as f1:
#     data=pickle.load(f1)
#
# with open('auto_code','rb') as f2:
#      code_data=pickle.load(f2)
#
# code_data = code_data.drop(labels=['label'],axis=1)
# code_original = pd.concat([code_data,data,y],axis=1) # 连接 codedata\原始\y


# index = ['0','1', '2','3','4','5','6','7', '8','9','NIR','Red','Green','labels']
# code_original.reindex(columns=index)

# with open("code_original", 'wb') as f:
#      pickle.dump(code_original, f)
#
# with open('E:\\VgIndex2.py\\其他\\pca_data','rb') as f3:
#      pca_data=pickle.load(f3)
#
# pca_data = pca_data.drop(labels=['label'],axis=1)
# pca_original=pd.concat([pca_data,data,y],axis=1) # 连接 pcadata\原始\y
#
# with open("pca_original", 'wb') as f:
#      pickle.dump(pca_original, f)

with open('E:\\VgIndex2.py\\预处理\\数据\\all_originaldata','rb') as f1:
     data=pickle.load(f1)

with open('E:\\VgIndex2.py\\预处理\\数据\\auto_code','rb') as f2:
     data1=pickle.load(f2)

with open('E:\\VgIndex2.py\\预处理\\数据\\code_original','rb') as f3:
     data2=pickle.load(f3)

with open('E:\\VgIndex2.py\\预处理\\数据\\pca_original','rb') as f4:
     data3=pickle.load(f4)

print("finish")