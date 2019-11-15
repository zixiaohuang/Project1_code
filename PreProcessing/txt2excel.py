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
    # txtlist=['A54','A55','A57','A58','A61','A62','A63','A65','A66','A68','A70','A74','A76','A77','A79','A81','A82','A83','A84','A85','B51','B54','B56','B80','B83','B84','B85']
    # txtlist = ['A1', 'A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26',
    #            'A27','A28','A29','A30','A31','A32','A33','A34','A35','A36','A37','A38','A39','A40','A41','A42']
    # txtlist = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16','B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24',
    #            'B25','B26','B27','B28','B29','B30','B31','B32','B33','B34','B35','B36','B37','B38','B39']
    # txtlist = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
    #            'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24',
    #            'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39']
    # txtlist = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16',
    #            'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24',
    #            'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37', 'D38', 'D39','D40']
    # txtlist = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16',
    #            'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24','E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37']
    # txtlist = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16',
    #            'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24','F25', 'F26', 'F27', 'F28', 'F29', 'F30', 'F31', 'F32', 'F33', 'F34']
    # txtlist = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16','G17', 'G18', 'G19','G20',
    #            'G21', 'G22', 'G23', 'G24','G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32', 'G33', 'G34', 'G35', 'G36']
    # txtlist = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16','H17', 'H18',
    #            'H19', 'H20', 'H21', 'H22', 'H23', 'H24','H25', 'H26', 'H27', 'H28', 'H29', 'H30', 'H31', 'H32', 'H33', 'H34']
    # txtlist = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16',
    #            'I17', 'I18','I19', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30']
    writer = pd.ExcelWriter(r'E:\\论文\\培敏\\预测ROI\\sequoia_predict\\new\\f2\\predictsave.xlsx')
    # for index in range(len(txtlist)):
    df = np.array(loadDataSet('E:\\论文\\培敏\\预测ROI\\sequoia_predict\\new\\f2\\sum.txt'))
    df2 = np.reshape(df,(30,-1))
    data_df = pd.DataFrame(df2.T)
    data_df.index = ['Red','Green','NIR']
    data_df.to_excel(writer,float_format='%.10f')
    writer.save()


if __name__ == "__main__":
    main()