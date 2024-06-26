import math
import topology_uninett

def calculate_distance(lat1, lon1, lat2, lon2):
    # Approximate radius of earth in km
    R = 6371.0

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def remove_affected_nodes(G, failure_center, failure_radius):
    """
    Remove nodes affected by the failure from the graph.
    """
    affected_nodes = []
    for node, attr in G.nodes(data=True):
        distance = calculate_distance(attr['Latitude'], attr['Longitude'], failure_center['Latitude'], failure_center['Longitude'])


        if distance <= failure_radius:  # Converting radius to degrees approx
            affected_nodes.append(node)
    
    # Remove the affected nodes
    for node in affected_nodes:
        G.remove_node(node)

    return G, affected_nodes

def geographical_failure(G, failure_center, failure_radius):
    affected_nodes = []
    affected_edges = []

    # Identify affected nodes
    for node, attr in G.nodes(data=True):
        distance = calculate_distance(attr['Latitude'], attr['Longitude'], failure_center['Latitude'], failure_center['Longitude'])
        # We only fail core nodes (not access nodes). To fail access nodes: remove  attr['color'] == 'red'
        if distance <= failure_radius and attr['color'] == 'red':
            affected_nodes.append(node)

    # Store and remove affected nodes and their edges
    for node in affected_nodes:
        # Store edges with attributes
        affected_edges.extend([(u, v, d) for u, v, d in G.edges(node, data=True)])
        # Remove node
        G.remove_node(node)


    # Reset network capacity to simulate recovery
    topology_uninett.reset_throughput(G)

    # Re-evaluate the network layout post-failure
    G_fail_shortest_paths, G_fail_shortest_path_lengths, G_fail_blue_nodes = topology_uninett.analyze_G(G)

    # Return graph and affected components for restoration
    #return G, G_fail_shortest_paths, G_fail_shortest_path_lengths, G_fail_blue_nodes, affected_nodes, affected_edges
    return G, affected_nodes


def flow_attack(G, failed_nodes):
    for node in failed_nodes:
        G.remove_node(node)
    topology_uninett.reset_throughput(G)
    return G

def flow_attack_repair(G, repaired_node):
    G.nodes[repaired_node]['available_memory'] = 2
    G.nodes[repaired_node]['available_cores'] = 1

def flow_attack_physical(G, failed_nodes):
    for node in failed_nodes:
        G.nodes[node]['available_memory'] = 0
        G.nodes[node]['available_cores'] = 0
    topology_uninett.reset_throughput(G)
    return G

def flow_attack_physical_repair(G, repaired_node):
    G.nodes[repaired_node]['available_memory'] += 2
    G.nodes[repaired_node]['available_cores'] += 1