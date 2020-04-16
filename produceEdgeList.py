import pandas as pd
import numpy as np
from collections import defaultdict

# 导入数据：分隔符为空格
raw_data = pd.read_csv('/Users/m-mac/Desktop/dataset/cora/cora.content', sep='\t', header=None)
num = raw_data.shape[0]  # 样本点数2708

# 将论文的编号转[0,2707]
a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b, a)
map = dict(c)

d = [1,2,3,4,5,6,7]
e = ['Case_Based','Genetic_Algorithms','Neural_Networks','Probabilistic_Methods','Reinforcement_Learning','Rule_Learning','Theory']
map2 = dict(zip(e,d))

# 将数据集的类别信息储存到文件
# fw = open("/Users/m-mac/Desktop/label.txt", 'w')  # 将要输出保存的文件地址
# for temp in zip(raw_data[0], raw_data[1434]):
#     fw.write(str(map[temp[0]]) + " ")  # 将字符串写入文件中
#     fw.write(str(map2[temp[1]]) + "\n")  # 将字符串写入文件中
# fw.close()

raw_data_cites = pd.read_csv('/Users/m-mac/Desktop/dataset/cora/cora.cites', sep='\t', header=None)

# 创建一个规模和邻接矩阵一样大小的矩阵
matrix = np.zeros((num, num))
# 创建邻接矩阵
for temp in zip(raw_data_cites[0], raw_data_cites[1]):
    x = map[temp[0]]
    y = map[temp[1]]  # 替换论文编号为[0,2707]
    matrix[x][y] = matrix[y][x] = 1  # 有引用关系的样本点之间取1
# 查看邻接矩阵的元素和（按每列汇总）
print(sum(matrix))


# converts from adjacency matrix to adjacency list
def convert(a):
    adjList = defaultdict(list)
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] == 1:
                adjList[i].append(j)
    return adjList

# 将临接列表转化为边缘列表
fw = open("/Users/m-mac/Desktop/eList.txt", 'w')  # 将要输出保存的文件地址
adjlist = convert(matrix)
for k in adjlist.keys():
    for item in adjlist[k]:
        fw.write(str(k) + ' ')  # 将字符串写入文件中
        fw.write(str(item) + "\n")  # 将字符串写入文件中
