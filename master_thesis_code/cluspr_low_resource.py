import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import interp1d
from itertools import combinations
import copy
from operator import itemgetter

import mainSurvivability



def shortest_path_for_physical_isolation(G, sd_pairs):
    slice_id = sd_pairs[0]['slice'].ID
    G_slice = copy.deepcopy(G)
    unavailable_nodes = []
    for node in G.nodes:
        try:
            if G.nodes[node]['node_reservation'] != slice_id and G.nodes[node]['node_reservation'] != 0:
                unavailable_nodes.append(node)
        except:
            continue
    # print(f'Unavailable nodes: {unavailable_nodes}')
    # sys.exit()
    G_slice.remove_nodes_from(unavailable_nodes)
    shortest_paths_PI = dict(nx.all_pairs_dijkstra_path(G_slice, weight = 'delay'))
    shortest_paths_lenghts_PI = dict(nx.all_pairs_dijkstra_path_length(G_slice, weight= 'delay' ))
    return shortest_paths_PI, shortest_paths_lenghts_PI

### CLUSPR ###


# This section of the code is implementing the CLUSPR algorithm
# CLUSPR is a clustering algorithm for network function placement in SDN

# The algorithm has three main steps:
# 1. Group nodes that are close to each other in a 'cluster'
# 2. Find flows that originate and end in the same clusters and group them in a 'intra-cluster'.
#    Flows who doesn't are 'inter-cluster'
# 3. Place NF instances independently for intra- and inter-cluster flows

# Functions for calculating Dunn index, min_inter_cluster_distance, and max_intra_cluster_distance

# Help function for ClusPR
def dunn_index(clusters, shortest_path_lengths):

    max_intra_cluster_dist = -1
    for set1 in clusters:
        dist = max_intra_cluster_distance(set1, shortest_path_lengths)
        if dist > max_intra_cluster_dist:
            max_intra_cluster_dist = dist


    min_inter_cluster_dist = float('inf')
    for set1 in clusters:
        if min_inter_cluster_dist == 3:
            break
        index_of_set1 = clusters.index(set1)
        for set2 in clusters[index_of_set1+1:]:
            dist = min_inter_cluster_distance(set1, set2, shortest_path_lengths)
            if dist < min_inter_cluster_dist:
                min_inter_cluster_dist = dist
            if min_inter_cluster_dist == 3:
                break

    return min_inter_cluster_dist / max_intra_cluster_dist


# Help function for ClusPR
def min_inter_cluster_distance(set1, set2, shortest_path_lengths):

    min_node_dist = float('inf')
    for node1 in set1:
        for node2 in set2:
            try: 
                dist = shortest_path_lengths[node1][node2]
                if dist < min_node_dist:
                    min_node_dist = dist
                if dist == 3:
                    return min_node_dist
            except:
                continue

    return min_node_dist

# Help function for ClusPR
def max_intra_cluster_distance(set1, shortest_path_lengths):
    max_node_dist = -1
    
    for node1 in set1:
        for node2 in set1:
            if node1 == node2:
                continue
            
            try:
                dist = shortest_path_lengths[node1][node2]
                if dist > max_node_dist:
                    max_node_dist = dist
            except:
                continue
 
                

    return max_node_dist


def cluster_sd_pairs(sd_pairs, blue_nodes, shortest_path_lengths):

    # ALGORITHM 1

    # Sort the source-destination pairs based on the shortest path between them
    sorted_sd_pairs = sorted(sd_pairs, key=lambda x: x['shortest_path'])

    # Initialize the clusters with single-node clusters for all blue nodes
    clusters = []
    for node in blue_nodes:
        clusters.append({node})

    # Initialize the optimal number of clusters (k) and best clusters found so far
    k = len(clusters)
    k_opt = k
    best_clusters = clusters.copy()

    # Calculate the initial Dunn index for the single-node clusters
    best_dunn_index = dunn_index(clusters, shortest_path_lengths)


    # Iterate through the sorted source-destination pairs
    for flow in sorted_sd_pairs:
        # Extract source and destination nodes from the flow dictionary
        source = flow['source']
        destination = flow['destination']

        # Find the clusters containing the source and destination nodes
        for set1 in clusters:
            # If the source node is in set1 and the destination node is not
            if source in set1 and destination not in set1:
                # Find the cluster containing the destination node
                for set2 in clusters:
                    if destination in set2:
                        # Merge the two clusters and update the list of clusters
                        set3 = set1.union(set2)
                        clusters.remove(set1)
                        clusters.remove(set2)
                        clusters.append(set3)

                        # Decrease the number of clusters by 1
                        k = k - 1

                        # Stop if there's only one cluster left
                        if k == 1:
                            break

                        # Calculate the new Dunn index for the merged clusters
                        dunn_index_k = dunn_index(clusters, shortest_path_lengths)

                        # If the new Dunn index is better, update the best clusters and optimal k
                        if dunn_index_k > best_dunn_index:
                            best_dunn_index = dunn_index_k
                            k_opt = k
                            best_clusters = clusters.copy()
                        # Stop searching for the destination cluster
                        break

    

    return best_clusters, k_opt, best_dunn_index


# Finds the minimum required number of each NF based on arrival rate of the flows
# Updates G with weights saying how many times it is visited by a flow's shortest path
# Changed in order to update the weights not only on the shortes path core nodes, but also the virtual nodes connected to the core nodes. 
def find_min_number_of_nfs(G, sd_pairs, shortest_paths, nf_service_rate):

    total_throughput_for_each_nf = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    min_number_of_each_nf = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Count how many times each edge and core node is visited by a flow and
    # count how many instances of each NF are required
    for flow in sd_pairs:
        try:
            flow_shortest_path = shortest_paths[flow['source']][flow['destination']]
        except:
            # print(f'Source and destination not connected for {flow}')
            continue

        flow_app_type = flow['application']  
        flow_arrival_rate = flow_app_type.sending_rate
        flow_required_nfs = flow['required_nfs']

        for node in flow_shortest_path:
            if G.nodes[node]['color'] != 'blue':
                G.nodes[node]['weight'] += 1
                G.nodes[node]['required_nfs'].update(flow_required_nfs)
                virtNodeList = mainSurvivability.id_generator.get_nodes_from_core_node(2, node)
                for virtNode in virtNodeList:
                    if virtNode in G.nodes():
                        G.nodes[virtNode]['weight'] += 1
                        G.nodes[virtNode]['required_nfs'] = copy.deepcopy(G.nodes[node]['required_nfs'])


        for nf in flow_required_nfs:
            total_throughput_for_each_nf[nf] += flow_arrival_rate

    # Calculate the minimum number of each NF based on the arrival rate of sd_pairs
    for nf in range(1, 6):
        total_rate_for_nf = total_throughput_for_each_nf[nf]
        min_number_of_each_nf[nf] = math.ceil(total_rate_for_nf / nf_service_rate)

    return min_number_of_each_nf


# Code to keep track of nf utilization in each node. Only used in the original clusPR implementation.
def initialize_nf_utilization(node_data):
    """Initialize nf_utilization based on hosted_nfs."""
    if 'hosted_nfs' in node_data:
        node_data['nf_utilization'] = {nf: 0 for nf in node_data['hosted_nfs']}

    return node_data

def nf_placement_A(i_v, G, slice_id):
    #nf placement changed to take into account the reservation of nfs. It basically just initializes a reservation value in the node so that the reservation can be kept track of.
    #print(f'Resources for nodes: {G.nodes(data=True)}')
    # Get edge and core nodes and sort them by their weight (number of times visited by a flow) in descending order
    edge_core_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('color') in ['mediumseagreen', 'red']]

    q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)

    # Sort the NFs by the minimum required instances in descending order. i_v = min_number_of_each_nf
    #q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys()) 
    q_nf = list(dict((k, v) for k, v in sorted(i_v.items(), key=lambda x: x[1], reverse=True) if v != 0).keys())

    # active node is the node that an nf was last placed on. It is checked first in order to fill up the node before trying others.
    active_node = None

    print(f'q_nf = {q_nf}')
    print(f'i_v: {i_v}')
    while q_nf != []:
        # v is the nf to be placed
        v = q_nf[0]
        #q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)
        while q_n != []:
            print(f'q_n: {q_n}')
            if active_node and (v in G.nodes[active_node]['required_nfs']):
                if G.nodes[active_node]['available_cores'] >= 1 and G.nodes[active_node]['available_memory'] >= 2: # only one nf can i instatiated per virtual node as there is only enough resource for one nf.
                    G.nodes[active_node]['available_cores'] -= 1
                    G.nodes[active_node]['available_memory'] -= 2
                    G.nodes[active_node]['hosted_nfs'].append(v)
                    G.nodes[active_node]['nf_utilization'][v] = 0
                    G.nodes[active_node]['nf_reservation'][v] = slice_id

                    print(f'placed nf {v} on {active_node} using if')

                    break
                    
                else:
                    active_node = None

            else:
                n = q_n[0]
                q_n.remove(n)
                if v in G.nodes[n]['required_nfs']:
                    if G.nodes[n]['available_cores'] >= 1 and G.nodes[n]['available_memory'] >= 2:
                        active_node = n
                        G.nodes[active_node]['available_cores'] -= 1
                        G.nodes[active_node]['available_memory'] -= 2
                        G.nodes[active_node]['hosted_nfs'].append(v)
                        G.nodes[active_node]['nf_utilization'][v] = 0
                        G.nodes[active_node]['nf_reservation'][v] = slice_id
                        print(f'i_v: {i_v}')
                        print(f'placed nf {v} on {active_node} using else')

                        break
                        
                else:
                    print(f'Not correct required nf')
                    print(f'Required nf: {G.nodes[n]['required_nfs']}')
                    print(f'V: {v}')
       
        if i_v[v] == 0:
            del i_v[v]
        else:
            i_v[v] -= 1
            if i_v[v] == 0:
                del i_v[v]
        
        q_nf = list(dict((k, v) for k, v in sorted(i_v.items(), key=lambda x: x[1], reverse=True) if v != 0).keys())

        # if q_nf == []:
        #     break

        # v = q_nf[0]

def nf_placement(i_v, G, slice_id):
    # Get edge and core nodes and sort them by their weight (number of times visited by a flow) in descending order
    edge_core_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('color') in ['mediumseagreen', 'red']]

    q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)

    # Sort the NFs by the minimum required instances in descending order
    q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys())

    active_node = None

    while q_nf != []:
        v = q_nf[0]

        while q_n != []:
            if active_node and v in G.nodes[active_node]['required_nfs']:
                if G.nodes[active_node]['available_cores'] >= 1 and G.nodes[active_node]['available_memory'] >= 2: 
                    G.nodes[active_node]['available_cores'] -= 1
                    G.nodes[active_node]['available_memory'] -= 2
                    G.nodes[active_node]['hosted_nfs'].append(v)
                    G.nodes[active_node]['nf_utilization'][v] = 0
                    G.nodes[active_node]['nf_reservation'][v] = slice_id
                    i_v[v] -= 1

                    if i_v[v] == 0:
                        del i_v[v]

                    q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys())

                    if q_nf == []:
                        break

                    v = q_nf[0]
                else:
                    active_node = None


            else:
                n = q_n[0]
                q_n.remove(n)
                if v in G.nodes[n]['required_nfs']:
                    if G.nodes[n]['available_cores'] >= 1 and G.nodes[n]['available_memory'] >= 2:
                        active_node = n
                        G.nodes[active_node]['available_cores'] -= 1
                        G.nodes[active_node]['available_memory'] -= 2
                        G.nodes[active_node]['hosted_nfs'].append(v)
                        G.nodes[active_node]['nf_utilization'][v] = 0
                        G.nodes[active_node]['nf_reservation'][v] = slice_id

                        i_v[v] -= 1

                        if i_v[v] == 0:
                            del i_v[v]


                        q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys())

                        if q_nf == []:
                            break

                        v = q_nf[0]


def nf_placement_original_A(i_v, G):
    #nf placement changed to take into account the reservation of nfs. It basically just initializes a reservation value in the node so that the reservation can be kept track of.
    #print(f'Resources for nodes: {G.nodes(data=True)}')
    # Get edge and core nodes and sort them by their weight (number of times visited by a flow) in descending order
    edge_core_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('color') in ['mediumseagreen', 'red']]

    q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)

    # Sort the NFs by the minimum required instances in descending order. i_v = min_number_of_each_nf
    #q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys()) #removed because i must have changed the structure of i_v unintentionally to include nfs with the value 0
    q_nf = list(dict((k, v) for k, v in sorted(i_v.items(), key=lambda x: x[1], reverse=True) if v != 0).keys())

    # active node is the node that an nf was last placed on. It is checked first in order to fill up the node before trying others.
    active_node = None

    print(f'q_nf = {q_nf}')
    print(f'i_v: {i_v}')
    while q_nf != []:
        # v is the nf to be placed
        v = q_nf[0]
        #q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)
        while q_n != []:
            print(f'q_n: {q_n}')
            if active_node and (v in G.nodes[active_node]['required_nfs']):
                if G.nodes[active_node]['available_cores'] >= 1 and G.nodes[active_node]['available_memory'] >= 2: 
                    G.nodes[active_node]['available_cores'] -= 1
                    G.nodes[active_node]['available_memory'] -= 2
                    G.nodes[active_node]['hosted_nfs'].append(v)

                    print(f'placed nf {v} on {active_node} using if')

                    break
                    
                else:
                    active_node = None

            else:
                n = q_n[0]
                q_n.remove(n)
                if v in G.nodes[n]['required_nfs']:
                    if G.nodes[n]['available_cores'] >= 1 and G.nodes[n]['available_memory'] >= 2:
                        active_node = n
                        G.nodes[active_node]['available_cores'] -= 1
                        G.nodes[active_node]['available_memory'] -= 2
                        G.nodes[active_node]['hosted_nfs'].append(v)

                        print(f'i_v: {i_v}')
                        print(f'placed nf {v} on {active_node} using else')

                        break
                       
                else:
                    print(f'Not correct required nf')
                    print(f'Required nf: {G.nodes[n]['required_nfs']}')
                    print(f'V: {v}')
       
        if i_v[v] == 0:
            del i_v[v]
        else:
            i_v[v] -= 1
            if i_v[v] == 0:
                del i_v[v]
        
        q_nf = list(dict((k, v) for k, v in sorted(i_v.items(), key=lambda x: x[1], reverse=True) if v != 0).keys())
    
    # Update each node's NF utilization
    for node, attributes in G.nodes(data=True):
        G.nodes[node].update(initialize_nf_utilization(attributes))

def nf_placement_original(i_v, G):
    # Get edge and core nodes and sort them by their weight (number of times visited by a flow) in descending order
    edge_core_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('color') in ['mediumseagreen', 'red']]

    q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)

    # Sort the NFs by the minimum required instances in descending order
    q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys())

    active_node = None

    while q_nf != []:
        v = q_nf[0]

        while q_n != []:
            if active_node and v in G.nodes[active_node]['required_nfs']:
                if G.nodes[active_node]['available_cores'] >= 1 and G.nodes[active_node]['available_memory'] >= 2: 
                    G.nodes[active_node]['available_cores'] -= 1
                    G.nodes[active_node]['available_memory'] -= 2
                    G.nodes[active_node]['hosted_nfs'].append(v)
                    i_v[v] -= 1

                    if i_v[v] == 0:
                        del i_v[v]

                    q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys())

                    if q_nf == []:
                        break

                    v = q_nf[0]
                else:
                    active_node = None


            else:
                n = q_n[0]
                q_n.remove(n)
                if v in G.nodes[n]['required_nfs']:
                    if G.nodes[n]['available_cores'] >= 1 and G.nodes[n]['available_memory'] >= 2:
                        active_node = n
                        G.nodes[active_node]['available_cores'] -= 1
                        G.nodes[active_node]['available_memory'] -= 2
                        G.nodes[active_node]['hosted_nfs'].append(v)

                        i_v[v] -= 1

                        if i_v[v] == 0:
                            del i_v[v]


                        q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys())

                        if q_nf == []:
                            break

                        v = q_nf[0]

    # Update each node's NF utilization
    for node, attributes in G.nodes(data=True):
        G.nodes[node].update(initialize_nf_utilization(attributes))



def nf_placement_physical_isolation(i_v, G, slice_id):
    #nf placement changed to take into account the reservation of nfs. It basically just initializes a reservation value in the node so that the reservation can be kept track of.
    #print(f'Resources for nodes: {G.nodes(data=True)}')
    # Get edge and core nodes and sort them by their weight (number of times visited by a flow) in descending order
    edge_core_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('color') in ['mediumseagreen', 'red']]

    q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)

    # Sort the NFs by the minimum required instances in descending order. i_v = min_number_of_each_nf
    #q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys()) #removed because i must have changed the structure of i_v unintentionally to include nfs with the value 0
    q_nf = list(dict((k, v) for k, v in sorted(i_v.items(), key=lambda x: x[1], reverse=True) if v != 0).keys())

    # active node is the node that an nf was last placed on. It is checked first in order to fill up the node before trying others.
    active_node = None

    print(f'q_nf = {q_nf}')
    print(f'i_v: {i_v}')
    while q_nf != []:
        # v is the nf to be placed
        v = q_nf[0]
        #q_n = sorted(edge_core_nodes, key=lambda node: G.nodes[node].get('weight'), reverse=True)
        while q_n != []:
            print(f'q_n: {q_n}')
            if active_node and (v in G.nodes[active_node]['required_nfs']):
                if (G.nodes[active_node]['node_reservation'] == slice_id or G.nodes[active_node]['node_reservation'] == 0) and G.nodes[active_node]['available_cores'] >= 1 and G.nodes[active_node]['available_memory'] >= 2: 
                    G.nodes[active_node]['available_cores'] -= 1
                    G.nodes[active_node]['available_memory'] -= 2
                    G.nodes[active_node]['hosted_nfs'].append(v)
                    G.nodes[active_node]['nf_utilization'][v] = 0
                    G.nodes[active_node]['nf_reservation'][v] = slice_id
                    G.nodes[active_node]['node_reservation'] = slice_id

                    print(f'placed nf {v} on {active_node} using if with reservation {G.nodes[active_node]['node_reservation']}')

                    break
                    # if i_v[v] == 0:
                    #     del i_v[v]
                    
                    # else:
                    #     i_v[v] -= 1
                    #     if i_v[v] == 0:
                    #         del i_v[v]
                    
                    #q_nf = list(dict(sorted(i_v.items(), key=lambda x: x[1], reverse=True)).keys()) #removed because i must have changed the structure of i_v unintentionally to include nfs with the value 0
                    # q_nf = list(dict((k, v) for k, v in sorted(i_v.items(), key=lambda x: x[1], reverse=True) if v != 0).keys())

                    # if q_nf == []:
                    #     break

                    # v = q_nf[0]
                else:
                    active_node = None

            else:
                n = q_n[0]
                q_n.remove(n)
                if v in G.nodes[n]['required_nfs']:
                    if (G.nodes[n]['node_reservation'] == slice_id or G.nodes[n]['node_reservation'] == 0) and G.nodes[n]['available_cores'] >= 1 and G.nodes[n]['available_memory'] >= 2:
                        active_node = n
                        G.nodes[active_node]['available_cores'] -= 1
                        G.nodes[active_node]['available_memory'] -= 2
                        G.nodes[active_node]['hosted_nfs'].append(v)
                        G.nodes[active_node]['nf_utilization'][v] = 0
                        G.nodes[active_node]['nf_reservation'][v] = slice_id
                        G.nodes[active_node]['node_reservation'] = slice_id
                        print(f'i_v: {i_v}')
                        print(f'placed nf {v} on {active_node} using else with reservation {G.nodes[active_node]['node_reservation']}')

                        break
                        
                else:
                    print(f'Not correct required nf')
                    print(f'Required nf: {G.nodes[n]['required_nfs']}')
                    print(f'V: {v}')
       
        if i_v[v] == 0:
            del i_v[v]
        else:
            i_v[v] -= 1
            if i_v[v] == 0:
                del i_v[v]
        
        q_nf = list(dict((k, v) for k, v in sorted(i_v.items(), key=lambda x: x[1], reverse=True) if v != 0).keys())

       


def print_nf_placement_stats(min_number_of_each_nf, G):

    print(f'Min number of nfs: {min_number_of_each_nf}')

    num_nfs = {1:0,2:0,3:0,4:0,5:0}

    # Print the final NF placements (removed the print)
    for node, attrs in G.nodes(data=True):
        if G.nodes[node]['color'] != 'blue':
            for nf in G.nodes[node]['hosted_nfs']:
                num_nfs[nf] += 1

    print(f'Number of instantiated nfs: {num_nfs}')

def compute_paths_for_flow(G, flow, nf_service_rate, shortest_path_lengths, k):
    stage_candidates = {}

    source = flow['source']
    destination = flow['destination']
    required_nfs = flow['required_nfs']
    flow_arrival_rate = (flow['application']).sending_rate
    flow_slice = flow['slice'].ID
    
    for num, nf in enumerate(required_nfs):
        stage_candidates[num+1] = []
        for node in G.nodes():
            if (G.nodes[node]['color'] != 'blue')\
                and (required_nfs[num] in G.nodes[node]['hosted_nfs'])\
                and (G.nodes[node]['nf_reservation'][nf] == flow_slice)\
                and (G.nodes[node]['nf_utilization'][nf] + ((flow_arrival_rate/nf_service_rate)*100) < 100):
                      
                stage_candidates[num+1].append(node)

    current_paths = []

    # Add last stage nodes to current path
    for node in stage_candidates[len(stage_candidates)]:
        try:
            delay = shortest_path_lengths[node][destination]
        except:
            continue
        current_paths.append([[node],delay])

    for step in range(len(stage_candidates)-1, 0, -1):
        new_current_paths = []
        for node in stage_candidates[step]:
            for path in current_paths:
                prev_node = path[0][0]

                delay = shortest_path_lengths[node][prev_node] + path[-1]
                new_path = [node] + path[0]
                new_current_paths.append([new_path, delay])

        current_paths = sorted(new_current_paths, key=lambda x: x[-1])
        
        node_dict = {}
        
        new_current_paths = []
        for path in current_paths:
            node = path[0][0]

            if node in node_dict:
                if node_dict[node] < k:
                    node_dict[node] += 1
                    new_current_paths.append(path)
            else:
                node_dict[node] = 1
                new_current_paths.append(path)

        current_paths = new_current_paths



    R = []
    for path in current_paths:
        first_node = path[0][0]
        try:
            first_step_delay = shortest_path_lengths[source][first_node]
        except:
            continue

        R.append([path[0], first_step_delay + path[-1]])

    return R

def compute_paths_for_flow_original(G, flow, nf_service_rate, shortest_path_lengths, k):
    stage_candidates = {}

    source = flow['source']
    destination = flow['destination']
    required_nfs = flow['required_nfs']
    flow_arrival_rate = (flow['application']).sending_rate

    for num, nf in enumerate(required_nfs):
        stage_candidates[num+1] = []
        for node in G.nodes:
            if G.nodes[node]['color'] != 'blue' and required_nfs[num] in G.nodes[node]['hosted_nfs'] and G.nodes[node]['nf_utilization'][nf] + ((flow_arrival_rate/nf_service_rate)*100) < 100:
                stage_candidates[num+1].append(node)

    current_paths = []

    # Add last stage nodes to current path
    for node in stage_candidates[len(stage_candidates)]:
        try:
            delay = shortest_path_lengths[node][destination]
        except:
            continue
        current_paths.append([[node],delay])

    for step in range(len(stage_candidates)-1, 0, -1):
        new_current_paths = []
        for node in stage_candidates[step]:
            for path in current_paths:
                prev_node = path[0][0]

                delay = shortest_path_lengths[node][prev_node] + path[-1]
                new_path = [node] + path[0]
                new_current_paths.append([new_path, delay])

        current_paths = sorted(new_current_paths, key=lambda x: x[-1])
        
        node_dict = {}
        
        new_current_paths = []
        for path in current_paths:
            node = path[0][0]

            if node in node_dict:
                if node_dict[node] < k:
                    node_dict[node] += 1
                    new_current_paths.append(path)
            else:
                node_dict[node] = 1
                new_current_paths.append(path)

        current_paths = new_current_paths



    R = []
    for path in current_paths:
        first_node = path[0][0]
        try:
            first_step_delay = shortest_path_lengths[source][first_node]
        except:
            continue

        R.append([path[0], first_step_delay + path[-1]])

    return R

def compute_paths_for_flow_physical_isolation(G, flow, nf_service_rate, shortest_path_lengths, k):
    stage_candidates = {}

    source = flow['source']
    destination = flow['destination']
    required_nfs = flow['required_nfs']
    flow_arrival_rate = (flow['application']).sending_rate
    flow_slice = flow['slice'].ID
    
    for num, nf in enumerate(required_nfs):
        stage_candidates[num+1] = []
        for node in G.nodes():
            if (G.nodes[node]['color'] != 'blue')\
                and (required_nfs[num] in G.nodes[node]['hosted_nfs'])\
                and (G.nodes[node]['node_reservation'] == flow_slice)\
                and (G.nodes[node]['nf_utilization'][nf] + ((flow_arrival_rate/nf_service_rate)*100) < 100):
                      
                stage_candidates[num+1].append(node)

    current_paths = []

    # Add last stage nodes to current path
    for node in stage_candidates[len(stage_candidates)]:
        try:
            delay = shortest_path_lengths[node][destination]
        except:
            continue
        current_paths.append([[node],delay])

    for step in range(len(stage_candidates)-1, 0, -1):
        new_current_paths = []
        for node in stage_candidates[step]:
            for path in current_paths:
                prev_node = path[0][0]
                try:
                    delay = shortest_path_lengths[node][prev_node] + path[-1]
                    new_path = [node] + path[0]
                    new_current_paths.append([new_path, delay])
                except:
                    continue
        current_paths = sorted(new_current_paths, key=lambda x: x[-1])
        
        node_dict = {}
        
        new_current_paths = []
        for path in current_paths:
            node = path[0][0]

            if node in node_dict:
                if node_dict[node] < k:
                    node_dict[node] += 1
                    new_current_paths.append(path)
            else:
                node_dict[node] = 1
                new_current_paths.append(path)

        current_paths = new_current_paths



    R = []
    for path in current_paths:
        first_node = path[0][0]
        try:
            first_step_delay = shortest_path_lengths[source][first_node]
        except:
            continue

        R.append([path[0], first_step_delay + path[-1]])

    return R


def find_min_max_utilization_path(G, nf_sequence, node_paths):

    min_max_utilization = float('inf')  # Initialize to positive infinity
    best_path = None

    for path in node_paths:
        node_path = path[0]
        max_utilization_in_path = 0  # Track the maximum utilization in the current path

        for node, nf in zip(node_path, nf_sequence):
            # Get the utilization of the current NF in the current node
            utilization = G.nodes[node]['nf_utilization'][nf]
            
            max_utilization_in_path = max(max_utilization_in_path, utilization) 

        # Update the best path if the current path's max utilization is less than the best found so far
        if max_utilization_in_path < min_max_utilization:
            # print(f'new minmax: from {min_max_utilization} to {max_utilization_in_path}')
            min_max_utilization = max_utilization_in_path
            best_path = path
 
    return best_path

def route_flows(G, sorted_sd_pairs, nf_service_rate, shortest_path_lengths, shortest_paths):

    # A flow looks like this:
    # {'source': source, 'destination': destination, 'required_nfs': nfs, 'shortest_path': shortest_path_lengths[source][destination], 'delay_requirement': round(delay_requirement, 1), 'priority': priority, 'application': app_type, 'slice': slice_type}
    # Example: {'source': 15, 'destination': 39, 'required_nfs': [2, 4], 'shortest_path': [18, 23, 2], 'delay_requirement': 28.8, 'priority': 2, 'application': VoIP, 'slice': EMBB}

    # [15, 39, [2, 4], 18, 28.8, 2]
    # [s, d, nfs, shortest_path, delay_req, priority]

    # Find the path of the flow such that it finds nfs.
    # Add path and check if it is valid or invalid in terms of delay requirement.
    # This loop iterates through each flow and determines the route for them
    
    no_paths = []

    for flow_id, flow in enumerate(sorted_sd_pairs):

        all_paths_for_flow = compute_paths_for_flow(G, flow, nf_service_rate, shortest_path_lengths, k=4)

        if len(all_paths_for_flow) == 0:
            #print(flow)
            print('No routs for flow!')

        flow_arrival_rate = flow['application'].sending_rate

        valid_delay_paths = []
        invalid_delay_paths = []

        # priority = flow['priority']

        # Determine valid paths based on delay requirements
        for path in all_paths_for_flow:
            if path[-1] <= flow['delay_requirement']:
                valid_delay_paths.append(path) # This is R_d
            else:
                invalid_delay_paths.append(path)


        # Assign route based on the route with the minimum maximum NF utilization 
        # If no routes fulfill delay requirement --> Assign route with lowest delay
        if len(valid_delay_paths) == 0:
            # Case 1: No found paths with correct SFC meets delay requirement. 
            try:
                flow_path = invalid_delay_paths[0] 
            except:
                try:
                    #path = shortest_paths[flow['source']][flow['destination']]
                    print('No paths found!')
                except:
                    # print('path not connected')
                    continue
                # print('path connected, something else wrong')
                continue

            sorted_sd_pairs[flow_id]['flow_path'] = flow_path[0] # Path
            sorted_sd_pairs[flow_id]['actual_delay'] = flow_path[1] # Delay (including source and destination)
            sorted_sd_pairs[flow_id]['case'] = 'case_1'
            for node, nf in zip(flow['flow_path'], flow['required_nfs']): 
                G.nodes[node]['nf_utilization'][nf] += (flow_arrival_rate/nf_service_rate)*100
                #print(f'Updated util for nf: {nf} in node: {node} to : {G.nodes[node]['nf_utilization'][nf]}')

        else:
            # Case 5: Correct SFC and valid path
            flow_path = find_min_max_utilization_path(G, flow['required_nfs'], valid_delay_paths)
            #flow_path = find_most_reserved_path(G, flow['required_nfs'], valid_delay_paths)
            sorted_sd_pairs[flow_id]['flow_path'] = flow_path[0] 
            sorted_sd_pairs[flow_id]['actual_delay'] = flow_path[1]
            sorted_sd_pairs[flow_id]['case'] = 'case_5'


            for node, nf in zip(flow['flow_path'], flow['required_nfs']): 
                G.nodes[node]['nf_utilization'][nf] += (flow_arrival_rate/nf_service_rate)*100
                #print(f'Updated util for nf: {nf} in node: {node} to : {G.nodes[node]['nf_utilization'][nf]}')


        source = flow['source']
        destination = flow['destination']
        flow_path = flow['flow_path']

        # Build the complete path for the flow
        complete_path = list(shortest_paths[source][flow_path[0]])
        
        # Add paths between consecutive nodes in the flow path
        for i in range(len(flow_path) - 1):
            complete_path += list(shortest_paths[flow_path[i]][flow_path[i + 1]][1:]) # Skip the first node to avoid duplication


        # Add path from the last node in flow_path to destination
        complete_path += list(shortest_paths[flow_path[-1]][destination][1:])  # Skip the first node to avoid duplication

        flow['complete_path'] = complete_path
        
    flows = sorted_sd_pairs

    return flows




def route_flows_original(G, sorted_sd_pairs, nf_service_rate, shortest_path_lengths, shortest_paths):

    # A flow looks like this:
    # {'source': source, 'destination': destination, 'required_nfs': nfs, 'shortest_path': shortest_path_lengths[source][destination], 'delay_requirement': round(delay_requirement, 1), 'priority': priority, 'application': app_type, 'slice': slice_type}
    # Example: {'source': 15, 'destination': 39, 'required_nfs': [2, 4], 'shortest_path': [18, 23, 2], 'delay_requirement': 28.8, 'priority': 2, 'application': VoIP, 'slice': EMBB}

    # [15, 39, [2, 4], 18, 28.8, 2]
    # [s, d, nfs, shortest_path, delay_req, priority]

    # Find the path of the flow such that it finds nfs.
    # Add path and check if it is valid or invalid in terms of delay requirement.
    # This loop iterates through each flow and determines the route for them
    
    no_paths = []

    for flow_id, flow in enumerate(sorted_sd_pairs):

        all_paths_for_flow = compute_paths_for_flow_original(G, flow, nf_service_rate, shortest_path_lengths, k=4)

        if len(all_paths_for_flow) == 0:
            #print(flow)
            print('No routs for flow!')

        flow_arrival_rate = flow['application'].sending_rate

        valid_delay_paths = []
        invalid_delay_paths = []

        # priority = flow['priority']

        # Determine valid paths based on delay requirements
        for path in all_paths_for_flow:
            if path[-1] <= flow['delay_requirement']:
                valid_delay_paths.append(path) # This is R_d
            else:
                invalid_delay_paths.append(path)


        # Assign route based on the route with the minimum maximum NF utilization 
        # If no routes fulfill delay requirement --> Assign route with lowest delay
        if len(valid_delay_paths) == 0:
            # Case 1: No found paths with correct SFC meets delay requirement. 
            try:
                flow_path = invalid_delay_paths[0] 
            except:
                try:
                    #path = shortest_paths[flow['source']][flow['destination']]
                    print('No paths found!')
                except:
                    # print('path not connected')
                    continue
                # print('path connected, something else wrong')
                continue

            sorted_sd_pairs[flow_id]['flow_path'] = flow_path[0] # Path
            sorted_sd_pairs[flow_id]['actual_delay'] = flow_path[1] # Delay (including source and destination)
            sorted_sd_pairs[flow_id]['case'] = 'case_1'
            for node, nf in zip(flow['flow_path'], flow['required_nfs']): 
                G.nodes[node]['nf_utilization'][nf] += (flow_arrival_rate/nf_service_rate)*100
                #print(f'Updated util: {G.nodes[node]['nf_utilization'][nf]}')


        else:
            # Case 5: Correct SFC and valid path
            flow_path = find_min_max_utilization_path(G, flow['required_nfs'], valid_delay_paths)
            sorted_sd_pairs[flow_id]['flow_path'] = flow_path[0] 
            sorted_sd_pairs[flow_id]['actual_delay'] = flow_path[1]
            sorted_sd_pairs[flow_id]['case'] = 'case_5'


            for node, nf in zip(flow['flow_path'], flow['required_nfs']): 
                G.nodes[node]['nf_utilization'][nf] += (flow_arrival_rate/nf_service_rate)*100
                #print(f'Updated util: {G.nodes[node]['nf_utilization'][nf]}')



        source = flow['source']
        destination = flow['destination']
        flow_path = flow['flow_path']

        # Build the complete path for the flow
        complete_path = list(shortest_paths[source][flow_path[0]])
        
        # Add paths between consecutive nodes in the flow path
        for i in range(len(flow_path) - 1):
            complete_path += list(shortest_paths[flow_path[i]][flow_path[i + 1]][1:]) # Skip the first node to avoid duplication


        # Add path from the last node in flow_path to destination
        complete_path += list(shortest_paths[flow_path[-1]][destination][1:])  # Skip the first node to avoid duplication

        flow['complete_path'] = complete_path
        
    flows = sorted_sd_pairs

    return flows

def route_flows_physical_isolation(G, sorted_sd_pairs, nf_service_rate, shortest_path_lengths, shortest_paths, slice_id):

    # A flow looks like this:
    # {'source': source, 'destination': destination, 'required_nfs': nfs, 'shortest_path': shortest_path_lengths[source][destination], 'delay_requirement': round(delay_requirement, 1), 'priority': priority, 'application': app_type, 'slice': slice_type}
    # Example: {'source': 15, 'destination': 39, 'required_nfs': [2, 4], 'shortest_path': [18, 23, 2], 'delay_requirement': 28.8, 'priority': 2, 'application': VoIP, 'slice': EMBB}

    # [15, 39, [2, 4], 18, 28.8, 2]
    # [s, d, nfs, shortest_path, delay_req, priority]

    # Find the path of the flow such that it finds nfs.
    # Add path and check if it is valid or invalid in terms of delay requirement.
    # This loop iterates through each flow and determines the route for them
    
    no_paths = []

    for flow_id, flow in enumerate(sorted_sd_pairs):

        all_paths_for_flow = compute_paths_for_flow_physical_isolation(G, flow, nf_service_rate, shortest_path_lengths, k=4)

        if len(all_paths_for_flow) == 0:
            #print(flow)
            print('No routs for flow!')

        flow_arrival_rate = flow['application'].sending_rate

        valid_delay_paths = []
        invalid_delay_paths = []

        # priority = flow['priority']

        # Determine valid paths based on delay requirements
        for path in all_paths_for_flow:
            if path[-1] <= flow['delay_requirement']:
                valid_delay_paths.append(path) # This is R_d
            else:
                invalid_delay_paths.append(path)


        # Assign route based on the route with the minimum maximum NF utilization 
        # If no routes fulfill delay requirement --> Assign route with lowest delay
        if len(valid_delay_paths) == 0:
            # Case 1: No found paths with correct SFC meets delay requirement. 
            try:
                flow_path = invalid_delay_paths[0] 
            except:
                try:
                    #path = shortest_paths[flow['source']][flow['destination']]
                    print('No paths found!')
                except:
                    # print('path not connected')
                    continue
                # print('path connected, something else wrong')
                continue

            sorted_sd_pairs[flow_id]['flow_path'] = flow_path[0] # Path
            sorted_sd_pairs[flow_id]['actual_delay'] = flow_path[1] # Delay (including source and destination)
            sorted_sd_pairs[flow_id]['case'] = 'case_1'
            for node, nf in zip(flow['flow_path'], flow['required_nfs']): 
                G.nodes[node]['nf_utilization'][nf] += (flow_arrival_rate/nf_service_rate)*100
                #print(f'Updated util for nf: {nf} in node: {node} to : {G.nodes[node]['nf_utilization'][nf]}')

        else:
            # Case 5: Correct SFC and valid path
            flow_path = find_min_max_utilization_path(G, flow['required_nfs'], valid_delay_paths)
            #flow_path = find_most_reserved_path(G, flow['required_nfs'], valid_delay_paths)
            sorted_sd_pairs[flow_id]['flow_path'] = flow_path[0] 
            sorted_sd_pairs[flow_id]['actual_delay'] = flow_path[1]
            sorted_sd_pairs[flow_id]['case'] = 'case_5'


            for node, nf in zip(flow['flow_path'], flow['required_nfs']): 
                G.nodes[node]['nf_utilization'][nf] += (flow_arrival_rate/nf_service_rate)*100
                #print(f'Updated util for nf: {nf} in node: {node} to : {G.nodes[node]['nf_utilization'][nf]}')


        source = flow['source']
        destination = flow['destination']
        flow_path = flow['flow_path']

        # Build the complete path for the flow
        complete_path = list(shortest_paths[source][flow_path[0]])
        
        # Add paths between consecutive nodes in the flow path
        for i in range(len(flow_path) - 1):
            complete_path += list(shortest_paths[flow_path[i]][flow_path[i + 1]][1:]) # Skip the first node to avoid duplication


        # Add path from the last node in flow_path to destination
        complete_path += list(shortest_paths[flow_path[-1]][destination][1:])  # Skip the first node to avoid duplication

        for node in complete_path:
            try:
                G.nodes[node]['node_reservation'] = slice_id
            except:
                continue
        flow['complete_path'] = complete_path
        
    flows = sorted_sd_pairs

    return flows


def update_throughput(G, flows):
    link_capacity_mbps = 1000  # 1 Gbps = 1000 Mbps

    # Ensure that each edge has an initial throughput value
    for u, v, k in G.edges:
        if 'throughput' not in G[u][v][k]:
            G[u][v][k]['throughput'] = 0

    # Process each flow
    for flow in flows:
        try:
            complete_path = flow['complete_path']
        except:
            # print('No complete path')
            continue

        # Update throughput for each link in the complete path
        for i in range(len(complete_path) - 1):
            u, v = complete_path[i], complete_path[i + 1]

            flow_arrival_rate = flow['application'].sending_rate

            # Calculate additional throughput as a percentage of link capacity
            additional_throughput_percentage = (flow_arrival_rate / link_capacity_mbps) * 100

            # Update throughput
            try:
                G[u][v][0]['throughput'] += additional_throughput_percentage
            except:
                # print(f'node {u}, node {v} link not found')
                continue



### CLUSPR ###

def apply_cluspr(G, sd_pairs, blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate, slice_id):
    
    #print(f'Sorted list of source destination pairs:     \n {sort_sd_pairs(copy.deepcopy(sd_pairs))}')

    # Step 1: Group similar data flows together
    # Can be skipped
    best_clusters, k_opt, best_dunn_index = cluster_sd_pairs(sd_pairs, blue_nodes, shortest_path_lengths)

    # Step 2: Determine the least amount of NFs to serve all flows
    min_number_of_each_nf = find_min_number_of_nfs(G, sd_pairs, shortest_paths, nf_service_rate)

    i_v = copy.deepcopy(min_number_of_each_nf)

    # Step 3: Decide where in the network to place NFs.
    nf_placement(i_v, G, slice_id)

    # Prints NF placement stats
    print_nf_placement_stats(min_number_of_each_nf, G)

    # Routing Phase
    # Finding the best path for each flow of data, making sure it goes through the necessary NFs and meets delay requirements.
    
    flows = route_flows(G, sd_pairs, nf_service_rate, shortest_path_lengths, shortest_paths)

    # Sets throughput values to the links based on the routed flows
    update_throughput(G, flows)

    # Returns the updated network and all flows with complete path
    return G, flows

def apply_cluspr_A(G, sd_pairs, blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate, slice_id):
    
    #print(f'Sorted list of source destination pairs:     \n {sort_sd_pairs(copy.deepcopy(sd_pairs))}')

    # Step 1: Group similar data flows together
    # Can be skipped
    best_clusters, k_opt, best_dunn_index = cluster_sd_pairs(sd_pairs, blue_nodes, shortest_path_lengths)

    # Step 2: Determine the least amount of NFs to serve all flows
    min_number_of_each_nf = find_min_number_of_nfs(G, sd_pairs, shortest_paths, nf_service_rate)

    i_v = copy.deepcopy(min_number_of_each_nf)

    # Step 3: Decide where in the network to place NFs.
    nf_placement_A(i_v, G, slice_id)

    # Prints NF placement stats
    print_nf_placement_stats(min_number_of_each_nf, G)

    # Routing Phase
    # Finding the best path for each flow of data, making sure it goes through the necessary NFs and meets delay requirements.
    
    flows = route_flows(G, sd_pairs, nf_service_rate, shortest_path_lengths, shortest_paths)

    # Sets throughput values to the links based on the routed flows
    update_throughput(G, flows)

    # Returns the updated network and all flows with complete path
    return G, flows

def apply_cluspr_original(G, sd_pairs, blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate):
    
    #print(f'Sorted list of source destination pairs:     \n {sort_sd_pairs(copy.deepcopy(sd_pairs))}')

    # Step 1: Group similar data flows together
    # Can be skipped
    best_clusters, k_opt, best_dunn_index = cluster_sd_pairs(sd_pairs, blue_nodes, shortest_path_lengths)

    # Step 2: Determine the least amount of NFs to serve all flows
    min_number_of_each_nf = find_min_number_of_nfs(G, sd_pairs, shortest_paths, nf_service_rate)

    i_v = copy.deepcopy(min_number_of_each_nf)

    # Step 3: Decide where in the network to place NFs.
    nf_placement_original(i_v, G)

    # Prints NF placement stats
    print_nf_placement_stats(min_number_of_each_nf, G)

    # Routing Phase
    # Finding the best path for each flow of data, making sure it goes through the necessary NFs and meets delay requirements.
    
    flows = route_flows_original(G, sd_pairs, nf_service_rate, shortest_path_lengths, shortest_paths)

    # Sets throughput values to the links based on the routed flows
    update_throughput(G, flows)

    # Returns the updated network and all flows with complete path
    return G, flows

def apply_cluspr_original_A(G, sd_pairs, blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate):
    
    #print(f'Sorted list of source destination pairs:     \n {sort_sd_pairs(copy.deepcopy(sd_pairs))}')

    # Step 1: Group similar data flows together
    # Can be skipped
    best_clusters, k_opt, best_dunn_index = cluster_sd_pairs(sd_pairs, blue_nodes, shortest_path_lengths)

    # Step 2: Determine the least amount of NFs to serve all flows
    min_number_of_each_nf = find_min_number_of_nfs(G, sd_pairs, shortest_paths, nf_service_rate)

    i_v = copy.deepcopy(min_number_of_each_nf)

    # Step 3: Decide where in the network to place NFs.
    nf_placement_original_A(i_v, G)

    # Prints NF placement stats
    print_nf_placement_stats(min_number_of_each_nf, G)

    # Routing Phase
    # Finding the best path for each flow of data, making sure it goes through the necessary NFs and meets delay requirements.
    
    flows = route_flows_original(G, sd_pairs, nf_service_rate, shortest_path_lengths, shortest_paths)

    # Sets throughput values to the links based on the routed flows
    update_throughput(G, flows)

    # Returns the updated network and all flows with complete path
    return G, flows


def reset_weight_and_required_nfs(G):
     empty_list = []
     for node_name, node_data in G.nodes(data = True):
            if len(str(node_name)) < 3:
                node_data['weight'] = 0
                node_data['required_nfs'].clear()
                node_data['required_nfs'].update(empty_list)
                virtNodeList = mainSurvivability.id_generator.get_nodes_from_core_node(2, node_name)
                for virtNode in virtNodeList:
                    if virtNode in G.nodes():
                        G.nodes[virtNode]['weight'] = 0
                        G.nodes[virtNode]['required_nfs'] = copy.deepcopy(node_data['required_nfs'])

def apply_cluspr_physical_isolation(G, sd_pairs, blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate):#, slice_id):
    # Split flows into flows from critical slice and other flows, for physical isolation
    sd_pairs_split_by_slice = {}
    for sd_pair in sd_pairs:
        slice_id = sd_pair['slice'].priority

        if slice_id in sd_pairs_split_by_slice:
            sd_pairs_split_by_slice[slice_id].append(sd_pair)
        else:
            sd_pairs_split_by_slice[slice_id] = [sd_pair]
    
    flows_split_by_slice = list(sd_pairs_split_by_slice.values())

    # Apply clusPerSlice physical isolation
    flows = []

    for flow_list in flows_split_by_slice:
        slice_id = flow_list[0]['slice'].priority # makes the slice_id the priority of the slice to re-use code from other strategies, for this strategy i want to physically isolate just the critical slice.
        min_number_of_each_nf = find_min_number_of_nfs(G, flow_list, shortest_paths, nf_service_rate)
        i_v = copy.deepcopy(min_number_of_each_nf)
        nf_placement_physical_isolation(i_v, G, slice_id)
        print_nf_placement_stats(min_number_of_each_nf, G)
        shortest_paths_PI, shortest_path_lengths_PI = shortest_path_for_physical_isolation(G, flow_list)
        flows_part = route_flows_physical_isolation(G, sd_pairs, nf_service_rate, shortest_path_lengths_PI, shortest_paths_PI, slice_id)
        update_throughput(G, flows_part)
        flows.extend(flows_part)
    
    # Returns the updated network and all flows with complete path
    return G, flows
