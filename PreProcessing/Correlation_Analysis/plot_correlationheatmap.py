'''
计算各植被指数间的相似度
'''

import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


with open('E:\\Project1_code\\Datasets\\new_alldata','rb') as f:
    data=pickle.load(f)
plt.figure()
#f,ax2 = plt.subplots(111)
all_dataselect = data.drop(labels=['label'], axis=1).sample(n=10000,axis=0)
data_normal = preprocessing.scale(all_dataselect)
print(data_normal.mean(axis=0))
print(data_normal.std(axis=0))
relation = np.corrcoef(data_normal,rowvar=False)*0.5+0.5
df = pd.DataFrame(relation)

xlabel = ['NDVI','SIPI','TVI', 'DVI','GDVI','OSAVI','RVI','SR','G','NDGI','IPVI',
                                    'CVI','MCARI1','MTVI1','MTVI2', 'RDVI', 'GRNDVI','Norm R','Norm NIR','Norm G']
ylabel = ['NDVI','SIPI','TVI', 'DVI','GDVI','OSAVI','RVI','SR','G','NDGI','IPVI',
                                    'CVI','MCARI1','MTVI1','MTVI2', 'RDVI', 'GRNDVI','Norm R','Norm NIR','Norm G']


sns.heatmap(df,linewidths=0.05,vmax=1.01,vmin=0.85,xticklabels=xlabel,yticklabels=ylabel,cmap='rainbow')#'rainbow'.,vmax=1.05,vmin=0,center=0.7,
plt.title('Vegetation Indices Correlation Analysis')
plt.show()