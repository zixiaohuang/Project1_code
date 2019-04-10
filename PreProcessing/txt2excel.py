'''
将导出处理后的txt文件按样本类型转换为excel文档
'''

import numpy as np
import pandas as pd

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    dataMat = []
    for line in fr.readlines(): # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，如果碰到结束符 EOF 则返回空字符串
        stringArr = line.strip().split(delim) # strip()删除s字符串中开头、结尾处，位于 rm删除序列的字符
        datArr = []
        for line in stringArr:
            datArr.append(float(line))
        dataMat.append(datArr)
    return np.mat(dataMat)

def main():
    txtlist=['A54','A55','A57','A58','A61','A62','A63','A65','A66','A68','A70','A74','A76','A77','A79','A81','A82','A83','A84','A85','B51','B54','B56','B80','B83','B84','B85']
    writer = pd.ExcelWriter(r'C:\\Users\\Administrator\\Desktop\\健康\\healthsave.xlsx')
    for index in range(len(txtlist)):
        df = np.array(loadDataSet('C:\\Users\\Administrator\\Desktop\\健康\\%s\\sum.txt'%(txtlist[index])))
        df2 = np.reshape(df,(30,-1))
        data_df = pd.DataFrame(df2.T)
        data_df.index = ['NIR','Red','Green']
        data_df.to_excel(writer,'%s'%(txtlist[index]),float_format='%.10f')
    writer.save()


if __name__ == "__main__":
    main()