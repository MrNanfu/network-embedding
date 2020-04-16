import networkx as nx
import numpy as np
import gensim
import matplotlib


def shuffleChoice(vertexList):
    return np.random.choice(vertexList)


# parameter : G->graph, paths->allPathPerVertex, pathPerVertex->pathPerVertex, pathDepth->the depth of one path, currentVertex->currentVertex
# function : produce numeric path per currentVertex, store all path in variant 'allPath'
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


# generate networkxGraph
G = nx.read_edgelist(path="./eList.txt")

# move all node to nodeSet
nodeSet = []
for node in nx.nodes(G):
    nodeSet.append(node)

pathPerVertex = 3
pathDepth = 10

# file store allPath information
with open('./allPath.txt', 'w') as allPath_file:
    # store allPath information in file
    for node in nodeSet:
        allPathPerNode = randomWalk(G=G, pathPerVertex=pathPerVertex, pathDepth=pathDepth, currentVertex=node)
        for path in allPathPerNode:
            allPath_file.write('\t'.join(path) + '\n')

allPath = []
# unzip information and store them in list allPath
with open('./allPath.txt', 'r') as f:
    for line in f.readlines():
        path = line.strip().split('\t')
        allPath.append(path)

# train model
model = gensim.models.Word2Vec(allPath, sg=1, size=300, alpha=0.025, window=3, min_count=1, max_vocab_size=None, sample=1e-3, seed=1, workers=45, min_alpha=0.0001, hs=0, negative=20, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=1e4)

# save
outfile = './test'
filename = './testmodel'
model.save(filename)
model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

model = gensim.models.Word2Vec.load(filename)
nearest10 = model.most_similar('3')
print(nearest10 )
