import pandas as pd
import pickle
import numpy as np
if __name__ == '__main__':
    illdata =pd.read_table('E:\\combinate_newdata\\vegindex_ill_new.txt',sep=" ",names=['NDVI','SIPI','TVI', 'DVI','GDVI','OSAVI','RVI','SR','G','NDGI','IPVI',
                                    'CVI','MCARI1','MTVI1','MTVI2', 'RDVI', 'GRNDVI','Norm R','Norm NIR','Norm G'])
    healthdata =pd.read_table('E:\\combinate_newdata\\vegindex_health_new.txt',sep=" ",names=['NDVI','SIPI','TVI', 'DVI','GDVI','OSAVI','RVI','SR','G','NDGI','IPVI',
                                    'CVI','MCARI1','MTVI1','MTVI2', 'RDVI', 'GRNDVI','Norm R','Norm NIR','Norm G'])

    #添加标签
    illdata['label']=-1
    healthdata['label']=1

    #合并两个数据集
    sumdata = healthdata.append(illdata)

    # pickle
    with open("new_alldata", 'wb') as f:
        pickle.dump(sumdata, f, True)
    # save
    np.savetxt('E:\\combinate_newdata\\alldata_new.txt',sumdata)
