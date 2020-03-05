import os
from collections import defaultdict

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

    def __init__(self, graph: nx.Graph, dimensions: int = 128, walk_length: int = 80, num_walks: int = 10, p: float = 2,
                 q: float = 0.5, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None, method: int = 1):
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

        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

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
                    # elif self.method==3:
                    #     prediction_jaccard_coefficient = list(nx.jaccard_coefficient(self.graph))
                    #     GJ = nx.Graph()
                    #     GJ.add_weighted_edges_from(prediction_jaccard_coefficient)
                    #     nx.draw(GJ, with_labels=True)
                    #     plt.draw()
                    #     plt.show()
                    #     if destination == source:  # Backwards probability
                    #         ss_weight = GJ[current_node][destination].get(self.weight_key, 1) * 1 / p
                    #     elif destination in GJ[source]:  # If the neighbor is connected to the source
                    #         ss_weight = GJ[current_node][destination].get(self.weight_key, 1)
                    #     else:
                    #         ss_weight = GJ[current_node][destination].get(self.weight_key, 1) * 1 / q
                    elif self.method==3:
                         rd=0.1
                         for eachNode in nodes_generator:
                             if destination != eachNode:
                                temp=nx.resistance_distance(self.graph, eachNode, destination, weight=None, invert_weight=True)
                                rd=rd+temp
                                                                                                                                                    
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

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)
    def _sort_neighbors(self, i) -> list:
        list_neighbors = list(self.graph.neighbors(i))
        '''nx.draw(self.graph,with_labels=True)
        plt.draw()
        plt.show()'''
        j=1;
        print("i:", i)
        print("neighbors:", list_neighbors)
        list_neighbors_sorted_degree = np.empty([len(list_neighbors)])
        list_neighbors_degree = np.empty([len(list_neighbors), 2])
        source_degree = self.graph.degree(i)
        print(len(list_neighbors))
        for k in range(len(list_neighbors)):
            neighbors_degree = self.graph.degree(list_neighbors[k])
            difdeg = abs(source_degree - neighbors_degree)
            neighbor = list_neighbors[k]
            listTemp = [difdeg, neighbor]
            if j == 2:
                if difdeg == 0:
                    difdeg = difdeg + 0.000000001
                list_neighbors_degree[k][0] = 1 / difdeg
            elif j == 1:
                list_neighbors_degree[k][0] = neighbors_degree
            list_neighbors_degree[k][1] = neighbor
        if j == 1:
            list_neighbors_degree[list_neighbors_degree[:, 0].argsort()]
        elif j == 2:
            list_neighbors_degree[list_neighbors_degree[:, 0].argsort()[::-1]]
        for k in range(len(list_neighbors)):
            neighbor = list_neighbors_degree[k][1]
            list_neighbors_sorted_degree[k] = neighbor
        print("list_neighbors_sorted_degree:", list_neighbors_sorted_degree)
        print("list_neighbors_degree:", list_neighbors_degree)
        return list_neighbors_sorted_degree, list_neighbors_degree