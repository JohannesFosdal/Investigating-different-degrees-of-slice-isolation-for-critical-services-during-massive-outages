
import plotsSurvivability
import failuresSurvivability
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

import os


def show_topology(file_path):

    G = nx.read_gml(file_path, label='id', destringizer=None)

    nodes_to_remove = [node for node, attrs in G.nodes(data=True) if 'Latitude' not in attrs or 'Longitude' not in attrs]
    G.remove_nodes_from(nodes_to_remove)
    

    plotsSurvivability.plot_graph(G)

    # # Density
    # density = nx.density(G)
    # print(f'Density for {file_path}: {density}')

    # # Average Degree
    # average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    # print(f'Average Degree for {file_path}: {average_degree}')


    # # Calculate the average clustering coefficient
    # simple_G = nx.Graph(G)
    # average_clustering_coefficient = nx.average_clustering(simple_G)
    # print(f'Average Clustering Coefficient for {file_path}: {average_clustering_coefficient}')

    # import community as community_louvain
    # # Detect communities and calculate modularity for a MultiGraph
    # partition = community_louvain.best_partition(simple_G)
    # modularity = community_louvain.modularity(partition, simple_G)
    # print("Modularity:", modularity)

    # # Calculate node connectivity
    # connectivity = nx.node_connectivity(G)

    # # Output the result
    # print(f'The vertex connectivity of the graph {file_path} is: {connectivity}')

    # # Number of links
    # links = G.number_of_edges()
    # print(f'Number of edges for {file_path}: {links}')



    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)


    plt.bar(*np.unique(degree_sequence, return_counts=True))
    plt.title("Degree histogram")
    plt.xlabel("Degree")
    plt.xticks(list(range(min(degree_sequence), max(degree_sequence)+1, 1)))
    plt.ylabel("Number of Nodes")

    plt.tight_layout()
    plt.show()


def show_topology_fail(file_path, failure_center, failure_radius):

    G = nx.read_gml(file_path, label='id', destringizer=None)

    nodes_to_remove = [node for node, attrs in G.nodes(data=True) if 'Latitude' not in attrs or 'Longitude' not in attrs]
    G.remove_nodes_from(nodes_to_remove)

    #remove failed nodes
    affected_nodes = []
    affected_edges = []

    # Identify affected nodes
    for node, attr in G.nodes(data=True):
        distance = failuresSurvivability.calculate_distance(attr['Latitude'], attr['Longitude'], failure_center['Latitude'], failure_center['Longitude'])
        if distance <= failure_radius:
            affected_nodes.append(node)

    # Store and remove affected nodes and their edges
    for node in affected_nodes:
        # Store edges with attributes
        affected_edges.extend([(u, v, d) for u, v, d in G.edges(node, data=True)])
        # Remove node
        G.remove_node(node)
    G.remove_edges_from(affected_edges)

    plotsSurvivability.plot_graph(G)

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)


    plt.bar(*np.unique(degree_sequence, return_counts=True))
    plt.title("Degree histogram")
    plt.xlabel("Degree")
    plt.xticks(list(range(min(degree_sequence), max(degree_sequence)+1, 1)))
    plt.ylabel("Number of Nodes")

    plt.tight_layout()
    plt.show()
    


show_topology('topology_zoo/Geant2012.gml')
failure_center = {'Latitude': 52, 'Longitude': 5}
failure_radius = 500
show_topology_fail('topology_zoo/Geant2012.gml', failure_center, failure_radius)


show_topology('topology_zoo/Uninett2011.gml')
failure_center = {'Latitude': 59.9, 'Longitude': 10.7}
failure_radius = 80
show_topology_fail('topology_zoo/Uninett2011.gml', failure_center, failure_radius)



