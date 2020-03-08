import os
import random as rnd
from collections import defaultdict
from sklearn import metrics
import numpy as np
import networkx as nx
import gensim
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from .parallel import parallel_generate_walks


class Node2Vec:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph,edge_subset: list, dimensions: int = 128, walk_length: int = 80, num_walks: int = 10, p: float = 2,
                 q: float = 0.5, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None, method: int = 1, directed: bool=False):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)
        self.method = method
        n=self.graph.number_of_nodes()+1
        self.resultmatrix=np.zeros([n,n])
        self.resistanceDistance=np.zeros([n-1,n-1])
        self.directed=directed
        self.edge_subset=edge_subset

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"
        if self.method ==6:
            for eachNode in range(n-1):
                rd = 0.1
                for eachNode1 in range(n-1):
                    if eachNode1 != eachNode:
                        try:
                            temp = nx.resistance_distance(self.graph, eachNode, eachNode1, weight=None, invert_weight=True)
                            rd=rd+temp
                        except:
                            rd = rd
                    self.resistanceDistance[eachNode][eachNode1]=rd
        self._precompute_probabilities()
        self.walks = self._generate_walks()
        self.resultmatrix=self._generate_matrix()
        self.sparsematrix = np.zeros([np.count_nonzero(self.resultmatrix),3])
        self.sparsematrix=self._generate_sparsematrix()
        self.auc_pref=self.generate_result_node2vec()
    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes()
            # if self.quiet \
            # else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q
                    if self.method==1:
                        if destination == source:  # Backwards probability
                            ss_weight = self.graph.degree(destination)*self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                        elif destination in self.graph[source]:  # If the neighbor is connected to the source
                            ss_weight = self.graph.degree(destination)*self.graph[current_node][destination].get(self.weight_key, 1)
                        else:
                            ss_weight = self.graph.degree(destination)*self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q
                    elif self.method==2:
                        if destination == source:  # Backwards probability
                            ss_weight = self.graph.degree(destination)*self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                        elif destination in self.graph[source]:  # If the neighbor is connected to the source
                            ss_weight = self.graph.degree(destination)*self.graph[current_node][destination].get(self.weight_key, 1)
                        else:
                            ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q
                    elif self.method==3:
                         defDeg=abs(self.graph.degree(destination)-self.graph.degree(current_node))
                         if defDeg==0:
                            defDeg=0.1
                         defDeg=1/defDeg
                         if destination == source:  # Backwards probability
                            ss_weight = defDeg * self.graph[current_node][destination].get(self.weight_key,
                                                                                                                   1) * 1 / p
                         elif destination in self.graph[source]:  # If the neighbor is connected to the source
                             ss_weight = defDeg * self.graph[current_node][destination].get(self.weight_key,
                                                                                                                   1)
                         else:
                             ss_weight = defDeg * self.graph[current_node][destination].get(self.weight_key,
                                                                                                                   1) * 1 / q
                    elif self.method == 4:
                        defDeg = abs(self.graph.degree(destination) - self.graph.degree(current_node))
                        if defDeg == 0:
                            defDeg = 0.1
                        defDeg = 1 / defDeg
                        if destination == source:  # Backwards probability
                            ss_weight = self.graph[current_node][destination].get(self.weight_key,
                                                                                           1) * 1 / p
                        elif destination in self.graph[source]:  # If the neighbor is connected to the source
                            ss_weight = self.graph[current_node][destination].get(self.weight_key,
                                                                                           1)
                        else:
                            ss_weight = defDeg * self.graph[current_node][destination].get(self.weight_key,
                                                                                           1) * 1 / q
                    elif self.method == 5:
                        defDeg = abs(self.graph.degree(destination) - self.graph.degree(current_node))
                        if defDeg == 0:
                            defDeg = 0.1
                        defDeg = 1 / defDeg
                        if destination == source:  # Backwards probability
                            ss_weight = self.graph.degree(destination)*self.graph[current_node][destination].get(self.weight_key,
                                                                                           1) * 1 / p
                        elif destination in self.graph[source]:  # If the neighbor is connected to the source
                            ss_weight = self.graph.degree(destination)*self.graph[current_node][destination].get(self.weight_key,
                                                                                           1)
                        else:
                            ss_weight = defDeg * self.graph[current_node][destination].get(self.weight_key,
                                                                                           1) * 1 / q
                    elif self.method==6:
                         rd=self.resistanceDistance[current_node][destination]

                         if destination == source:  # Backwards probability
                            ss_weight = (1/rd) * self.graph[current_node][destination].get(self.weight_key,
                                                                                                                   1) * 1 / p
                         elif destination in self.graph[source]:  # If the neighbor is connected to the source
                             ss_weight =  (1/rd) * self.graph[current_node][destination].get(self.weight_key,
                                                                                                                   1)
                         else:
                             ss_weight =  (1/rd) * self.graph[current_node][destination].get(self.weight_key,
                                                                                                                   1) * 1 / q
                    else:
                         if destination == source:  # Backwards probability
                            ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                         elif destination in self.graph[source]:  # If the neighbor is connected to the source
                            ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                         else:
                            ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q
                     # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks
    def _generate_matrix(self):
        resultmatrix=self.resultmatrix
        for walk in self.walks:
            if len(walk)>1:
                index=0
                for node in walk:
                     index1=0
                     if index+1<len(walk):
                       sourceNode=int(walk[index])#row
                       currentNode=int(walk[index+1]) #column
                       prob=self.d_graph[currentNode][self.PROBABILITIES_KEY][sourceNode]
                       negihbors=self.d_graph[currentNode][self.NEIGHBORS_KEY]
                       for nNode in negihbors:
                           if int(nNode)==int(node):
                               probValue = prob[index1]
                               break
                           index1=index1+1
                       resultmatrix[sourceNode][currentNode]=resultmatrix[sourceNode][currentNode]+probValue
                       index=index+1
        for walk in self.walks:
            steper=2
            if len(walk) > 1:
                while steper<len(walk):
                    index=0
                    while index<len(walk)-steper:
                          sourceIdx=index
                          destIdx=index + steper
                          sourceNode = int(walk[index])
                          destNode = int(walk[index + steper])
                          if sourceNode!=destNode:
                              pValue=1
                              startIdx= index
                              stopIdx= index+1
                              for l in range(steper):
                                   startNode=int(walk[startIdx])
                                   stopNode=int(walk[stopIdx])
                                   pValue=pValue*resultmatrix[startNode][stopNode]
                                   startIdx=stopIdx
                                   stopIdx = stopIdx + 1
                              if pValue!=1:
                                 resultmatrix[sourceNode][destNode]=resultmatrix[sourceNode][destNode]+pValue
                          index=index+1
                    steper=steper+1

        return resultmatrix
    def _generate_sparsematrix(self):
        sparsematrixTemp = self.sparsematrix
        index=0
        indexT=0
        if self.directed == False:
            for i  in range(self.graph.number_of_nodes()):
                for j in range(i):
                    if self.resultmatrix[i][j]!=0:
                          sparsematrixTemp[index][0]=j
                          sparsematrixTemp[index][1] = i
                          sparsematrixTemp[index][2] =max( self.resultmatrix[i][j],self.resultmatrix[j][i])
                          index=index+1
            sparsematrix = np.zeros([index, 3])
            for i in range(index):
              if sparsematrixTemp[i][2] != 0:
                  sparsematrix[indexT][0] = int(sparsematrixTemp[i][0])
                  sparsematrix[indexT][1] = int(sparsematrixTemp[i][1])
                  sparsematrix[indexT][2] = sparsematrixTemp[i][2]
                  indexT = indexT + 1
        else:
            for i in range(self.graph.number_of_nodes()):
              for j in range(self.graph.number_of_nodes()):
                  if self.resultmatrix[i][j] != 0:
                        sparsematrix[index][0] = i
                        sparsematrix[index][1] = j
                        sparsematrix[index][2] = self.resultmatrix[i][j]
                        index = index + 1
        temp = np.argsort(sparsematrix[:, 2])
        rtemp = temp[::-1]
        sparsematrix= sparsematrix[rtemp]
        return sparsematrix
    def generate_result_node2vec(self)-> float:
        listMatrix=self.sparsematrix
        G= nx.Graph()
        G.add_weighted_edges_from(listMatrix)
        listG = list(G.edges(data='weight'))
        score_pref, label_pref = zip(*[(s, (u, v) in self.edge_subset) for (u, v, s) in listG])
        try:
           fpr_pref, tpr_pref, _ = metrics.roc_curve(label_pref, score_pref)
           auc_pref = metrics.roc_auc_score(label_pref, score_pref)
        except:
           auc_pref=0
        return auc_pref





























