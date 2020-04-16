from collections import Counter

import networkx as nx
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

# 设置deepWalk、skip-gram算法的参数
WINDOWSIZE = 2
EMBEDDINGSIZE = 4
WALKSPERVERTEX = 3
WALKDEPTH = 10
PRINT_EVERY = 1000
EPOCHS = 1000  # 训练的轮数
BATCH_SIZE = 5  # 每一批训练数据大小
N_SAMPLES = 3  # 负样本大小


# 从一个点集中随机选取一个并且返回
def shuffleChoice(vertexList):
    return np.random.choice(vertexList)


# 对于特定的一个顶点，返回其所有的randomWalk路径
def randomWalk(G, pathPerVertex, pathDepth, currentVertex):
    allPath = []
    for i in range(pathPerVertex):
        a = 0
        path = [currentVertex]
        tempcurrentVertex = currentVertex
        while a < pathDepth:
            node_list = []
            for _nbr in G[tempcurrentVertex].items():
                node_list.append(_nbr[0])
            tempcurrentVertex = shuffleChoice(node_list)
            path.append(tempcurrentVertex)
            a += 1
        allPath.append(path)
    return allPath


def getnoise_dist(path):
    # 计算单词频次
    int_word_counts = Counter(path)
    total_count = len(path)
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}

    # 单词分布
    word_freqs = np.array(list(word_freqs.values()))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
    return noise_dist


# 将str的list转化为int的list
def strToInt(path):
    i = 0
    for str in path:
        path[i] = int(str)
        i += 1
    return path


# 获取目标词汇
def get_target(words, idx, WINDOW_SIZE):  # 中心词（被转换为数字），中心词对应的id，窗口大小
    target_window = np.random.randint(1, WINDOW_SIZE + 1)  # 数据很多的时候，可以随机调整窗口大小，减轻计算压力
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])
    return list(targets)


# 批次化数据
def get_batch(words, BATCH_SIZE, WINDOW_SIZE):
    n_batches = len(words) // BATCH_SIZE
    words = words[:n_batches * BATCH_SIZE]
    for idx in range(0, len(words), BATCH_SIZE):
        batch_x, batch_y = [], []
        batch = words[idx:idx + BATCH_SIZE]
        for i in range(len(batch)):
            x = batch[i]
            y = get_target(batch, i, WINDOW_SIZE)
            batch_x.extend([x] * len(y))  # 中心词扩张，因为一个中心词对应多个周边词
            batch_y.extend(y)
        yield batch_x, batch_y


# 定义模型
class SkipGramNeg(nn.Module):  # 这个nn.Module代表传入的是父类
    def __init__(self, n_vocab, n_embed, noise_dist):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        # 定义词向量层
        self.in_embed = nn.Embedding(n_vocab, n_embed)  # 这个Embedding的作用是把一个整数转换为一个向量，转换的依据是word2Vector的index对应的向量
        self.out_embed = nn.Embedding(n_vocab, n_embed)  # ,但是如果下标没有语义特征，那么这个向量就是下标对饮的热编码
        # 词向量层参数初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    # 输入词的前向过程
    def forward_input(self, input_words):  # 这里传入的是一个一维列表
        input_vectors = self.in_embed(input_words)
        return input_vectors

    # 目标词的前向过程
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    # 负样本词的前向过程
    def forward_noise(self, size, N_SAMPLES):
        noise_dist = self.noise_dist
        # 从词汇分布中采样负样本
        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,  # 每一个词都有N_SAMPLES个负样本向量
                                        replacement=True)
        noise_vectors = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors


# 定义损失函数
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        BATCH_SIZE, embed_size = input_vectors.shape
        # 将输入词向量与目标词向量作维度转化处理
        input_vectors = input_vectors.view(BATCH_SIZE, embed_size, 1)
        output_vectors = output_vectors.view(BATCH_SIZE, 1, embed_size)
        # 目标词损失
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()  # （B,1,1）
        out_loss = out_loss.squeeze()  # 降维为 B
        # 负样本损失
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # 多个负样本，所以要加和
        # 综合计算两类损失
        return -(out_loss + noise_loss).mean()


# 从edjlist文件读入networkx图
G = nx.read_edgelist(path="./eList.txt")

# 将图的所有点加入nodeSet列表
nodeSet = []
for node in nx.nodes(G):
    nodeSet.append(node)

# 通过randomWalk算法，将图中所有点的path全部写入文件
with open('./allPath.txt', 'w') as allPath_file:
    # store allPath information in file
    for node in nodeSet:
        allPathPerNode = randomWalk(G=G, pathPerVertex=WALKSPERVERTEX, pathDepth=WALKDEPTH, currentVertex=node)
        for path in allPathPerNode:
            allPath_file.write('\t'.join(path) + '\n')

allPath = []
# 从文件读取所有通过randomWalk生成的路径，并且将每条路径都放入allPath列表
with open('./allPath.txt', 'r') as f:
    for line in f.readlines():
        path = line.strip().split('\t')
        path = strToInt(path)
        allPath.append(path)

# 计算单词频次
int_word_counts = Counter(path)
total_count = len(path)
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}

# 单词分布
word_freqs = np.array(list(word_freqs.values()))
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

# 模型、损失函数及优化器初始化
model = SkipGramNeg(len(nodeSet), EMBEDDINGSIZE, noise_dist=noise_dist)
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

steps = 0
# 通过skipGram进行模型的训练
for path in allPath:
    # 获取输入词以及目标词
    for input_words, target_words in get_batch(path, BATCH_SIZE, WINDOWSIZE):  # 这里的input_words和target_words都是一维的向量
        model.noise_dist = getnoise_dist(path)
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        # 输入、输出以及负样本向量
        input_vectors = model.forward_input(inputs)  # 传入的是一个有重复index的一维向量
        output_vectors = model.forward_output(targets)
        size, _ = input_vectors.shape
        noise_vectors = model.forward_noise(size, N_SAMPLES)
        # 计算损失
        loss = criterion(input_vectors, output_vectors, noise_vectors)
        # 打印损失
        if steps % PRINT_EVERY == 0:
            print("loss：", loss)
        # 梯度回传
        optimizer.zero_grad()  # 损失置0，每epoch的损失无关，所以每次需要至零。
        loss.backward()  # 回传
        optimizer.step()  # 更新参数

count = 0
# 绘制图像
for i in nodeSet:
    count += 1
    i = int(i)
    if count > 50:
        break
    vectors = model.state_dict()["in_embed.weight"]
    x, y = float(vectors[i][0]), float(vectors[i][1])
    plt.scatter(x, y)
    plt.annotate(i, xy=(x, y))
plt.show()
