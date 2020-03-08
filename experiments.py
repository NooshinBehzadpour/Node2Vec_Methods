import networkx as nx
import random
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

# Create a graph1
nodesNumber=[10,30,60,80,120]
P=[0.2,0.4,0.6,0.8]
pAlpha=[0.25,0.5,1,2,4]
pEdges=[0.2,0.4,0.6,0.8]
methods=[1,2,3,4,5,6,7]
numWalks=[2,3,4]
walkLength=[4,5,6,7]
repeat=10
index=0
resultArray=np.zeros([len(nodesNumber)*len(P)*len(pAlpha)*len(pEdges)*len(methods)*len(numWalks*len(walkLength)),17])
for a in range(len(nodesNumber)):
    for b in range(len(P)):
        for c in range(len(pAlpha)):
            for e in range(len(pEdges)):
                    for f in range(len(numWalks)):
                        for g in range(len(walkLength)):
                            #aa
                            auc_adamic=0
                            auc_jaccard=0
                            auc_resource_allocation=0
                            auc=np.zeros([7])
                            for iter in range(repeat):
                                graphMain = nx.fast_gnp_random_graph(n=nodesNumber[a], p=P[b])
                                nx.draw(graphMain, with_labels=True)
                                proportion_edges = pEdges[e]
                                edge_subset = random.sample(graphMain.edges(),
                                                            int(proportion_edges * graphMain.number_of_edges()))
                                # Create a copy of the graph and remove the edges
                                graph = graphMain.copy()
                                graph.remove_edges_from(edge_subset)
                                pred_adamic = list(nx.adamic_adar_index(graph))
                                score_adamic, label_adamic = zip(*[(s, (u, v) in edge_subset) for (u, v, s) in pred_adamic])
                                try:
                                    fpr_adamic, tpr_adamic, _ = metrics.roc_curve(label_adamic, score_adamic)
                                    t_auc_adamic = metrics.roc_auc_score(label_adamic, score_adamic)
                                except:
                                    t_auc_adamic=0
                                auc_adamic=auc_adamic+t_auc_adamic

                                #j
                                pred_jaccard = list(nx.jaccard_coefficient(graph))
                                score_jaccard, label_jaccard = zip(*[(s, (u, v) in edge_subset) for (u, v, s) in pred_jaccard])
                                try:
                                    fpr_jaccard, tpr_jaccard, _ = metrics.roc_curve(label_jaccard, score_jaccard)
                                    t_auc_jaccard = metrics.roc_auc_score(label_jaccard, score_jaccard)
                                except:
                                    t_auc_jaccard=0
                                auc_jaccard = auc_jaccard + t_auc_jaccard

                                #ra
                                pred_resource_allocation = list(nx.resource_allocation_index(graph))
                                score_resource_allocation, label_resource_allocation = zip(*[(s, (u, v) in edge_subset) for (u, v, s) in pred_resource_allocation])
                                try:
                                    fpr_resource_allocation, tpr_resource_allocation, _ = metrics.roc_curve(label_resource_allocation, score_resource_allocation)
                                    t_auc_resource_allocation = metrics.roc_auc_score(label_resource_allocation, score_resource_allocation)
                                except:
                                    t_auc_resource_allocation=0
                                auc_resource_allocation=auc_resource_allocation+t_auc_resource_allocation

                                for h in range(len(methods)):
                                    methodH = methods[h]
                                    node2vec = Node2Vec(graph,edge_subset,  dimensions=8, walk_length=walkLength[g], num_walks=numWalks[f],p=pAlpha[c],q=1/pAlpha[c], workers=4,method=methodH,directed=False)
                                    auc[h]=auc[h]+node2vec.generate_result_node2vec()

                            resultArray[index][14] = auc_adamic / repeat
                            resultArray[index][15] = auc_jaccard / repeat
                            resultArray[index][16] = auc_resource_allocation / repeat
                            for h in range(len(methods)):
                                resultArray[index][7 + h]=auc[h]/repeat
                            resultArray[index][0] = nodesNumber[a]
                            resultArray[index][1] = P[b]
                            resultArray[index][2] = pAlpha[c]
                            resultArray[index][3] = 1/pAlpha[c]
                            resultArray[index][4] = pEdges[e]
                            resultArray[index][5] = numWalks[f]
                            resultArray[index][6] = walkLength[g]
                            print(index)
                            index=index+1

