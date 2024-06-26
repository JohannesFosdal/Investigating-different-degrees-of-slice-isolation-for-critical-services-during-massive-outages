import networkx as nx
import random
import math
from operator import itemgetter
import copy

import mainSurvivability



def parse_gml_and_create_G(gml_file_path):

    G = nx.read_gml(gml_file_path, label='id', destringizer=None)

    return G

def remove_nodes_without_coordinates(G):
    nodes_to_remove = [node for node, attrs in G.nodes(data=True) if 'Latitude' not in attrs or 'Longitude' not in attrs]
    G.remove_nodes_from(nodes_to_remove)
    return G

def set_node_color_red(graph):
    # Define the color attribute to add to each node
    color_attribute = {node: 'red' for node in graph.nodes()}
    
    # Use set_node_attributes to add the attribute to each node
    nx.set_node_attributes(graph, color_attribute, 'color')

def set_core_node_attributes(G, slices):
    for node in G.nodes:
        if G.nodes[node]['color'] == 'red':
            G.nodes[node]['weight'] = 0
            G.nodes[node]['available_cores'] = 4 #4  #remember to change repair of attacked nodes too, if changing this
            G.nodes[node]['available_memory'] = 8 #8
            G.nodes[node]['required_nfs'] = set()
            G.nodes[node]['hosted_nfs'] = []
            G.nodes[node]['nf_utilization'] = {} 
            G.nodes[node]['nf_reservation'] = {} 
            G.nodes[node]['node_reservation'] = 0 #only for use in physical isolation
    

# Currently delays only set to 10. Could be changed.
def set_delay_10(G):

    # Ensure the delay attribute is set to 10 for each edge
    delay_attr = {edge: 10 for edge in G.edges}
    
    # Use nx.set_edge_attributes to add/update the 'delay' attribute for each edge
    nx.set_edge_attributes(G, delay_attr, 'delay')

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def set_dynamic_delay(G):
    for edge in G.edges:
        node1, node2, key = edge
        # Extract the latitude and longitude of each node
        lat1, lon1 = G.nodes[node1]['Latitude'], G.nodes[node1]['Longitude']
        lat2, lon2 = G.nodes[node2]['Latitude'], G.nodes[node2]['Longitude']
        # Calculate the distance between the nodes
        distance = haversine(lat1, lon1, lat2, lon2)
        # Convert distance to delay here; adjust the conversion logic as needed
        delay = distance_to_delay(distance) # Implement this function based on your delay conversion logic
        # Set the delay for the edge
        nx.set_edge_attributes(G, {(node1, node2, key): delay}, 'delay')

def distance_to_delay(distance):
    # Example conversion: 1ms delay for every 100km
    delay = distance / 100.0
    return delay

def add_blue_nodes(G):
    # Generate a unique ID for new nodes based on the highest existing node ID + 100
    #blue_node_id = 100

    for node, attributes in list(G.nodes(data=True)):
        node_type = attributes.get('type', '').lower()
        new_nodes_count = 0
        core_node_id = node
        #blue_node_id = 100 + (10*node)

        # Determine the number of blue nodes to add based on the node type
        if "large circle" in node_type:
            new_nodes_count = 4
        elif "small circle" in node_type:
            new_nodes_count = 2
        else:  # For other types
            new_nodes_count = 3

        original_latitude = attributes.get('Latitude', 0)
        original_longitude = attributes.get('Longitude', 0)

        # Add the new blue nodes and connect them to the current node
        for i in range(new_nodes_count):
            # Generate small offsets for latitude and longitude
            latitude_offset = random.uniform(-1, 1)
            longitude_offset = random.uniform(-1, 1)

            new_latitude = original_latitude + latitude_offset
            new_longitude = original_longitude + longitude_offset

            new_node_id = mainSurvivability.id_generator.generate_id(1,core_node_id)

            G.add_node(new_node_id, color='blue', type='access',
                       Latitude=new_latitude, Longitude=new_longitude)
            G.add_edge(node, new_node_id)

        # new_pos_list = points_on_circle(original_latitude, original_longitude, 0.002, new_nodes_count)

        # for pos in new_pos_list:
        #     new_node_id = main.id_generator.generate_id(1,core_node_id)
        #     G.add_node(new_node_id, color='blue', type='access',
        #                Latitude=pos[0], Longitude=pos[1])
        #     G.add_edge(node, new_node_id)


    return G

def points_on_circle(center_lat, center_lon, radius, num_points):
    # Convert latitude and longitude to radians
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)
    
    # Initialize list to store points
    points = []
    
    # Calculate angle between points
    angle_between_points = 2 * math.pi / num_points
    
    # Generate points on the circle
    for i in range(num_points):
        # Calculate the angle for this point
        theta = i * angle_between_points
        
        # Calculate the coordinates of the point in Cartesian system
        x = center_lon_rad + radius * math.cos(theta)
        y = center_lat_rad + radius * math.sin(theta)
        
        # Convert Cartesian coordinates back to latitude and longitude
        lat = math.degrees(y)
        lon = math.degrees(x)
        
        # Add the point to the list
        points.append((lat, lon))
    
    return points



def analyze_G(G):
    
    # 2. Calculate shortest paths between all pairs of nodes
    shortest_paths = dict(nx.all_pairs_dijkstra_path(G, weight = 'delay'))
    
    # 3. Calculate shortest path lengths between all pairs of nodes 
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='delay'))

    # 4. Identify blue nodes
    blue_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('color') == 'blue']
    
    return shortest_paths, shortest_path_lengths, blue_nodes

def generate_flow(source, dest, nfs, app_type, slice_type, shortest_path_lengths, slices):
    slice_type1 = [slic for slic in slices if slic.type == slice_type][0]
    application_type = [app for app in slice_type1.application_types if app.type == app_type][0]

    sd_pair = {
        'source': source,
        'destination': dest,
        'required_nfs': nfs,
        'shortest_path': shortest_path_lengths[source][dest],
        'delay_requirement': 5 * shortest_path_lengths[source][dest],
        'priority': slice_type1.priority,
        'application': application_type,
        'slice': slice_type1
        }
    return sd_pair

# Generate flows based on random endpoints and random selction of slice, application and required network functions.
def generate_flows(G, shortest_path_lengths, blue_nodes, number_of_flows, slices, application_types): 
    sd_pairs = []

    slice_choice_probabilities = [0.25, 0.25, 0.25, 0.25]

    for i in range(number_of_flows):
        source, destination = random.sample(blue_nodes, 2)

        number_of_nfs = random.choice([2, 3, 4])
        number_of_nfs = 2
        nfs = random.sample(range(1, 6), number_of_nfs) # Will only draw one NF per type

        # Calculate the delay requirement based on the shortest path length (1 - 2.5 times the shortest path)
        #delay_requirement = (random.randint(20, 50) / 10) * shortest_path_lengths[source][destination]
        delay_requirement = 5 * shortest_path_lengths[source][destination]

        # Select slice type
        slice_type = random.choices(slices, slice_choice_probabilities, k=1)[0]

        # Select one application type based on the defined probabilities and slice type
        app_type = random.choices(slice_type.application_types, slice_type.app_probabilities, k=1)[0]
        
        # Get priority
        priority = slice_type.priority

        # Create a flow dictionary with relevant information (source, destination, required_nfs, etc.)
        sd_pair = {
            'source': source,
            'destination': destination,
            'required_nfs': nfs,
            'shortest_path': shortest_path_lengths[source][destination],
            'delay_requirement': round(delay_requirement, 1),
            'priority': priority,
            'application': app_type,
            'slice': slice_type
        }

        sd_pairs.append(sd_pair)
    return sd_pairs

# Generate flows based on random endpoints, but a given set of flow types with given application and required network functions.
def generate_flows_set(G, shortest_path_lengths, blue_nodes, number_of_flows, slices, application_types): 
    sd_pairs = []
    flows_per_slice = number_of_flows['flows_per_slice']

    for slice in slices:
        #for i in range(flows_per_slice):
        slice_type = slice
        for n in range(number_of_flows['VoIP']):
            app_type = application_types[1]
            source, destination = random.sample(blue_nodes, 2) 
            sd_pair = {
                    'source': source,
                    'destination': destination,
                    'required_nfs': app_type.req_nfs,
                    'shortest_path': shortest_path_lengths[source][destination],
                    'delay_requirement': app_type.delay_req,
                    'priority': slice_type.priority,
                    'application': app_type,
                    'slice': slice_type
                }
            sd_pairs.append(sd_pair)
        
        for n in range(number_of_flows['VoD']):
            app_type = application_types[0]
            source, destination = random.sample(blue_nodes, 2) 
            sd_pair = {
                    'source': source,
                    'destination': destination,
                    'required_nfs': app_type.req_nfs,
                    'shortest_path': shortest_path_lengths[source][destination],
                    'delay_requirement': app_type.delay_req,
                    'priority': slice_type.priority,
                    'application': app_type,
                    'slice': slice_type
                }
            sd_pairs.append(sd_pair)

        for n in range(number_of_flows['LVD']):
            app_type = application_types[2]
            source, destination = random.sample(blue_nodes, 2) 
            sd_pair = {
                    'source': source,
                    'destination': destination,
                    'required_nfs': app_type.req_nfs,
                    'shortest_path': shortest_path_lengths[source][destination],
                    'delay_requirement': app_type.delay_req,
                    'priority': slice_type.priority,
                    'application': app_type,
                    'slice': slice_type
                }
            sd_pairs.append(sd_pair)

        for n in range(number_of_flows['FD']):
            app_type = application_types[3]
            source, destination = random.sample(blue_nodes, 2) 
            sd_pair = {
                    'source': source,
                    'destination': destination,
                    'required_nfs': app_type.req_nfs,
                    'shortest_path': shortest_path_lengths[source][destination],
                    'delay_requirement': app_type.delay_req,
                    'priority': slice_type.priority,
                    'application': app_type,
                    'slice': slice_type
                }
            sd_pairs.append(sd_pair)
    
    random.shuffle(sd_pairs)

    return sd_pairs


# Generate flows based on random endpoints, with application type chosen according to given probabilities. The app type is connected to specific required nfs.
def generate_flows_new(G, shortest_path_lengths, blue_nodes, number_of_flows, slices, application_types): 
    sd_pairs = []
    
    slice_choice_probabilities = [0.25, 0.25, 0.25, 0.25]

    for i in range(number_of_flows):
        source, destination = random.sample(blue_nodes, 2)

        # Select slice type
        slice_type = random.choices(slices, slice_choice_probabilities, k=1)[0]

        # Select one application type based on the defined probabilities and slice type
        app_type = random.choices(slice_type.application_types, slice_type.app_probabilities, k=1)[0]

        nfs = app_type.req_nfs
    

        # Calculate the delay requirement based on the shortest path length (1 - 2.5 times the shortest path)
        #delay_requirement = (random.randint(20, 50) / 10) * shortest_path_lengths[source][destination]
        delay_requirement = 5 * shortest_path_lengths[source][destination]
        
        # Get priority
        priority = slice_type.priority

        # Create a flow dictionary with relevant information (source, destination, required_nfs, etc.)
        sd_pair = {
            'source': source,
            'destination': destination,
            'required_nfs': nfs,
            'shortest_path': shortest_path_lengths[source][destination],
            'delay_requirement': round(delay_requirement, 1),
            'priority': priority,
            'application': app_type,
            'slice': slice_type
        }

        sd_pairs.append(sd_pair)
    return sd_pairs



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
            continue

        # Update throughput for each link in the complete path
        for i in range(len(complete_path) - 1):
            u, v, k = complete_path[i], complete_path[i + 1]

            flow_arrival_rate = flow['application'].sending_rate

            # Calculate additional throughput as a percentage of link capacity
            additional_throughput_percentage = (flow_arrival_rate / link_capacity_mbps) * 100

            # Update throughput
            G[u][v][k]['throughput'] += additional_throughput_percentage


def reset_throughput(G):
    # Iterate through all edges in the graph
    for u, v, k in G.edges:
        # Set the throughput of each edge to 0
        if 'throughput' in G[u][v][k]:
            G[u][v][k]['throughput'] = 0

# Sorts the list of source destination pairs by priority     
def sort_sd_pairs(sd_pairs):
    sorted_sd_pairs = sorted(sd_pairs, key=itemgetter('priority')) 
    return sorted_sd_pairs

def create_virtualized_topology(G, split_number):
    # Values for attribute describing what software the virtual nodes are running (for use in failure simulation)
    software_ids = [1, 2, 3]
    software_probabilities = [0.2, 0.7, 0.1]

    core_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('color') == 'red']
    for node in core_nodes: 
        for i in range(split_number):
            new_node_id = mainSurvivability.id_generator.generate_id(2, node)
            G.add_node(new_node_id,  
                       color = G.nodes[node]['color'],
                       Latitude = G.nodes[node]['Latitude'],
                       Longitude = G.nodes[node]['Longitude'],
                       weight = copy.deepcopy(G.nodes[node]['weight']),
                       available_cores = (copy.deepcopy(G.nodes[node]['available_cores']))//split_number,
                       available_memory = (copy.deepcopy(G.nodes[node]['available_memory']))//split_number,
                       required_nfs = [],
                       hosted_nfs = [],
                       nf_utilization = {},
                       nf_reservation = {},
                       software = random.choices(software_ids, weights=software_probabilities, k=1)[0]
                       ) 
            G.add_edge(node, new_node_id)
        G.nodes[node]['available_cores'] = 0
        G.nodes[node]['available_memory'] = 0
    return G

def generate_topology_and_flows(gml_file_path, number_of_flows, application_types, slices):
    G = parse_gml_and_create_G(gml_file_path)
    G = remove_nodes_without_coordinates(G)
    set_node_color_red(G)
    set_core_node_attributes(G)
    G = add_blue_nodes(G)
    set_dynamic_delay(G)
    set_dynamic_delay(G)
    shortest_paths, shortest_path_lengths, blue_nodes = analyze_G(G)
    sd_pairs = generate_flows(G, shortest_path_lengths, blue_nodes, number_of_flows, slices)
    update_throughput(G, sd_pairs)

    return G, shortest_paths, shortest_path_lengths, blue_nodes, sd_pairs

def generate_topology_and_flows_virtualized(gml_file_path, number_of_flows, application_types, slices, split_number):
    mainSurvivability.id_generator.reset_generator()
    G = parse_gml_and_create_G(gml_file_path)
    G = remove_nodes_without_coordinates(G)
    set_node_color_red(G)
    set_core_node_attributes(G, slices)
    G = add_blue_nodes(G)
    #plots.plot_graph(G)
    VirtualizedG = create_virtualized_topology(copy.deepcopy(G), split_number)
    #plots.plot_graph(VirtualizedG)
    set_dynamic_delay(G)
    set_dynamic_delay(VirtualizedG)
    shortest_paths, shortest_path_lengths, blue_nodes = analyze_G(VirtualizedG)
    sd_pairs = generate_flows_set(VirtualizedG, shortest_path_lengths, blue_nodes, number_of_flows, slices, application_types)
    update_throughput(VirtualizedG, sd_pairs)
    update_throughput(G, sd_pairs)

    return G, VirtualizedG, shortest_paths, shortest_path_lengths, blue_nodes, sd_pairs

