# Case Study 6

import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

def practice():
    '''
    Practising various networkx commands.
    '''
    g = nx.Graph()
    g.add_node(1)
    g.add_nodes_from([2, 3])
    g.add_nodes_from(['u', 'v'])

    g.add_edge(1,2)
    g.add_edge('u', 'v')
    g.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6)])
    g.add_edge('u', 'w')

    g.remove_node(2)
    g.remove_nodes_from([4, 5])
    g.remove_edge(1, 3)
    g.remove_edges_from([(1, 2), ('u', 'v')])

    print(g.nodes())
    print(g.edges())
    print(g.number_of_nodes())
    print(g.number_of_edges())

def karate():
    '''
    Explore the provided karate club data.
    '''
    g = nx.karate_club_graph()

    nx.draw(g, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.savefig('karate_graph.png')

    print(g.degree())
    print(g.degree()[33] is g.degree(33))

    print(g.number_of_nodes(), g.number_of_edges())

def er_graph(N, p, print_graph=False):
    '''
    Generate an ER graph with N nodes and p probability of edge.
    Optional argument to save the resultant graph.
    '''
    g = nx.Graph()
    g.add_nodes_from(range(N))
    for x in g:
        for y in g:
            if x < y and bernoulli.rvs(p=p): # x < y as only want to try each potential edge once
                g.add_edge(x, y)

    if print_graph:
        nx.draw(g, with_labels=True, node_color='red', edge_color='blue')
        plt.savefig('er_graph.png')

    return g

def plot_degree_distribution_random_er(N, p, num=1):
    '''
    Plots the degree distribution of the er_graph (N, p) as a histogram.
    num is number of times to generate a graph (default=1).
    '''
    for x in range(num):
        g = er_graph(N, p)
        # use .values to get no.of edges for each node as a list
        plt.hist(list(g.degree().values()), histtype='step')
    plt.xlabel('Degree $k$')
    plt.ylabel('$P(k)$')
    plt.title('Degree distribution N={}, p={}'.format(N, p))
    plt.savefig('hist_3.png')

def plot_degree_distribution(graphs):
    '''
    Plots the degree distribution of given graphs.
    '''
    for g in graphs:
        plt.hist(list(g.degree().values()), histtype='step')
    plt.xlabel('Degree $k$')
    plt.ylabel('$P(k)$')
    plt.title('Degree distribution')
    plt.savefig('village_hist.png')

def print_basic_net_stats(g):
    '''
    Basic stats of a graph.
    '''
    print('Number of nodes: {}'.format(g.number_of_nodes()))
    print('Number of edges: {}'.format(g.number_of_edges()))
    print('Average degree: {0:.2f}'.format(np.mean(list(g.degree().values()))))

def random_er_items():
    '''
    From vid.
    '''
    er_graph(20, 0.2, print_graph=True)
    plot_degree_distribution_random_er(500, 0.08, num=3)

def indian_villages():
    '''
    From vid.
    '''
    a1 = np.loadtxt('adj_allVillageRelationships_vilno_1.csv', delimiter=',')
    a2 = np.loadtxt('adj_allVillageRelationships_vilno_2.csv', delimiter=',')

    g1 = nx.to_networkx_graph(a1)
    g2 = nx.to_networkx_graph(a2)

    #print_basic_net_stats(g1)
    #print_basic_net_stats(g2)

    #plot_degree_distribution([g1, g2])

    gen1 = nx.connected_component_subgraphs(g1)
    g1_lcc = max(nx.connected_component_subgraphs(g1), key=len)
    g2_lcc = max(nx.connected_component_subgraphs(g2), key=len)
    # print(g1_lcc.number_of_nodes())
    # print(g2_lcc.number_of_nodes())
    # print(g1_lcc.number_of_nodes() / g1.number_of_nodes())
    # print(g2_lcc.number_of_nodes() / g2.number_of_nodes())

    plt.figure()
    nx.draw(g1_lcc, node_color='red', edge_color='gray', node_size=20)
    plt.savefig('village1.png')

    plt.figure()
    nx.draw(g2_lcc, node_color='green', edge_color='gray', node_size=20)
    plt.savefig('village2.png')
