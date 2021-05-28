import math
import pandas as pd
import numpy as np
import random
import networkx as nx
from sklearn.metrics import confusion_matrix, roc_auc_score


def common_neighbors(graph, x, y):
    arr = [w for w in graph[x] if w in graph[y]]
    return [1 if z in arr else 0 for z in range(1, len(graph) + 1)]


def find_dict(nodes):
    nodes = np.sort(nodes)
    count = 0
    mapping = {}
    for node in nodes:
        while len(mapping) + count != node:
            count += 1
        mapping[node - count] = node
    return mapping


def unconnected_pairs(graph):
    adj = nx.to_numpy_matrix(graph, nodelist=graph.nodes)
    uncon = []
    mapping = find_dict(graph.nodes)
    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[1]):
            if i != j:
                if adj[i, j] == 0:
                    uncon.append((mapping[i], mapping[j]))
    return uncon


def create_model(graph, metric):
    # random.seed(1240)
    # nodes = graph.nodes

    edges = graph.edges
    edges = [sorted(edge) for edge in edges]

    uncon = unconnected_pairs(graph)

    negative_testing = random.sample(edges, round(len(edges) / 5))  # 1/5 of the edges
    negative_testing = [tuple(l) for l in negative_testing]

    negative_training = list(filter(lambda i: i not in negative_testing, edges))  # 4/5 of the edges
    negative_training = [tuple(l) for l in negative_training]

    testing_pairs = uncon

    print(len(uncon))

    for tup in negative_testing:
        testing_pairs.append(tup)

    new_graph = nx.from_edgelist(negative_training)

    if metric == 'aa':
        scored = list(nx.adamic_adar_index(new_graph, testing_pairs))
    elif metric == 'pa':
        scored = list(nx.preferential_attachment(new_graph, testing_pairs))

    df = pd.DataFrame(scored, columns=['x', 'y', 'index'])

    true_label = np.zeros(len(df))
    true_label[-len(negative_testing):] = 1

    df['true'] = true_label

    df2 = df.sort_values(ascending=False, by=['index'])

    # df2.to_csv('data_out.csv')

    score_label = np.zeros(len(df2))
    score_label[:len(negative_testing)] = 1

    true_label1 = list(df2['true'])

    print(confusion_matrix(true_label1, score_label))
    print(roc_auc_score(true_label1, score_label))


def main():
    # df = pd.read_csv("data/biogrid_drosophila_2010.csv")
    df = pd.read_csv("data/fb-pages-food.csv")
    # df = pd.read_csv("data/facebook-links.csv")
    network = nx.from_pandas_edgelist(df, "x", "y", create_using=nx.Graph())
    metric = 'pa'
    create_model(network, metric)


def pretty_print(matrix):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(pd.DataFrame(matrix))


main()
