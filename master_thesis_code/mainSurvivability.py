# Import files from project
import cluspr_low_resource
import topology_uninett
import plotsSurvivability
import evaluateSurvivability
import failuresSurvivability
import survivability
import general_tools

import matplotlib.pyplot as plt
import copy
import numpy as np

id_generator = general_tools.ID_generator()

### VALUES ###

# Speed at which the network functions operate
nf_service_rate = 70 # Mbps

# {Total number of flows, flows per slice, flows of VoD app type, flows of VoIP type, flows of LVD type, flows of FD type}
number_of_flows = {'total_flows': 400, 'flows_per_slice': 100, 'VoD': 15, 'VoIP': 70, 'LVD': 10, 'FD': 5}

### DEFINING THE SLICE AND APP TYPE OBJECTS ###

# Defining the object type slice
class slice:

    def __init__(self, ID, type, priority, application_types, app_probabilities):
        self.ID = ID
        self.type = type
        self.priority = priority
        self.application_types = application_types
        self.app_probabilities = app_probabilities
    
    def __str__(self):
        return f"Slice ID: {self.ID}, Type: {self.type}, Priority: {self.priority}, Possible applications: {self.application_types}, Probabilities: {self.app_probabilities}"

    def __repr__(self):
        return f"{self.type}"

# Defining the object type aplication    
class application:

    def __init__(self, type, sending_rate, req_nfs, delay_requirement):
        self.type = type
        self.sending_rate = sending_rate
        self.req_nfs = req_nfs
        self.delay_req = delay_requirement
    
    def __str__(self):
        return f"Application type: {self.type}, Sending rate: {self.sending_rate}, Required NFs: {self.req_nfs}, Delay requirement: {self.delay_req}"
    
    def __repr__(self):
        return f"{self.type}"


### CREATING CASES ###

def create_case_1():    
    # Creating a the applications and adding them to a list
    VoD = application('VoD', 11.2, [1, 3, 5], 120) 
    VoIP = application('VoIP', 0.3, [2, 4, 5], 70) 
    LVD = application('LVD', 18.2, [2, 5, 3], 80)
    FD = application('FD', 22.4, [2 ,4, 1], 120)

    application_types = [VoD, VoIP, LVD, FD]

    # Creating a list of the slices in the model
    nødnett = slice(1, 'nødnett', 1, [VoD, VoIP, LVD, FD], [0.19, 0.63, 0.07, 0.11])
    URLLC = slice(2, 'URLLC', 2, [VoD, VoIP, LVD, FD], [0.19, 0.63, 0.07, 0.11])
    EMBB = slice(3, 'EMBB', 2, [VoD, VoIP, LVD, FD], [0.19, 0.63, 0.07, 0.11])
    MMTC = slice(4, 'MMTC', 2, [VoD, VoIP, LVD, FD], [0.19, 0.63, 0.07, 0.11])

    slices = [nødnett, URLLC, EMBB, MMTC]

    return slices, application_types

slices, application_types = create_case_1()

### TOPOLOGY USED ###

# Some failures for the Uninett2011 topology
oslo = {'Latitude': 59.9, 'Longitude': 10.7}

# Failures for other topologies
geant2012_fail = {'Latitude': 52, 'Longitude': 5}

# Path to the network layout file
#gml_file_path = 'surviveNet/topology_zoo/Geant2012.gml'
gml_file_path = 'surviveNet/topology_zoo/Uninett2011.gml'

# Failure area to use in simulation
#failure_area = copy.deepcopy(geant2012_fail)
failure_area = copy.deepcopy(oslo)
#failure_radius = 500
failure_radius = 80

def main():

        # INITIALIZE TOPOLOGY AND FLOWS
    split_number = 4 #number of virtual nodes per core node
    # G is the topology with just physical nodes. G_initial is the topology with added virtual nodes within the core nodes.
    G, G_initial, shortest_paths_initial, shortest_path_lengths_initial, blue_nodes_initial, flows_initial = topology_uninett.generate_topology_and_flows_virtualized(gml_file_path, number_of_flows, application_types, slices, split_number)
    
    

    ### WITOUT VIRTUAL NODES ###

        # INITIALIZE CLUSPR VERSIONS
    # Apply ClusPR_original
    G_cluspr_P, flows_cluspr_P = cluspr_low_resource.apply_cluspr_original(copy.deepcopy(G), copy.deepcopy(flows_initial), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    clusPR_nf_amount_P = plotsSurvivability.print_total_amount_of_nfs_physical(G_cluspr_P)
    clusPR_nf_utilization_P = plotsSurvivability.get_utilization_list(G_cluspr_P)
    initial_performance_p = evaluateSurvivability.check_flows(flows_cluspr_P)
    initial_performance_pc = evaluateSurvivability.check_priority_flows(flows_cluspr_P,1)
    

    # Sort flows for priority routing
    flows_initial_sorted_P = topology_uninett.sort_sd_pairs(copy.deepcopy(flows_initial))
    
    # Apply ClusPR_original on sorted flow list
    G_cluspr_sorted_P, flows_cluspr_sorted_P = cluspr_low_resource.apply_cluspr_original(copy.deepcopy(G), copy.deepcopy(flows_initial_sorted_P), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    clusPR_sorted_nf_amount_P = plotsSurvivability.print_total_amount_of_nfs_physical(G_cluspr_sorted_P)
    clusPR_sorted_nf_utilization_P = plotsSurvivability.get_utilization_list(G_cluspr_sorted_P)

    G_cluspr_isol_P, flows_cluspr_isol_P = cluspr_low_resource.apply_cluspr_physical_isolation(copy.deepcopy(G), copy.deepcopy(flows_initial_sorted_P), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)#, slice_id)



    
    cluspr_isol_P_nf_amount = plotsSurvivability.print_total_amount_of_nfs(G_cluspr_isol_P)

    
        # SIMULATE GEOGRAPHICAL FAILURE
    failure_center = failure_area

    # Show the original network layout
    #plotsSurvivability.plot_graph(G_cluspr_P)

    # Highlight the area affected by the failure on the network layout
    #plotsSurvivability.plot_graph(G_cluspr_P, True, failure_center, failure_radius)

    #G_geofail_initial, G_geofail_shortest_paths, G_geofail_shortest_path_lengths, G_geofail_blue_nodes, failed_nodes, failed_edges = failures.geographical_failure(copy.deepcopy(G_initial), failure_center, failure_radius)
    G_geofail_initial_P, failed_nodes_P = failuresSurvivability.geographical_failure(copy.deepcopy(G), failure_center, failure_radius)
    #print(f'Number of failed nodes: {len(failed_nodes_P)}')
    # Show the network layout after removing the failed parts
    #plotsSurvivability.plot_graph(G_geofail_initial_P) 

    
        # SURVIVABILITY ASSESSMENT


    # Without priority routing for flows, original clusPR
    survivability.survivability(copy.deepcopy(G), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr_P), copy.deepcopy(G_geofail_initial_P), copy.deepcopy(failed_nodes_P), nf_service_rate, originalClusPR = True, priority = False, physical = True, filename = "original_clusPR_physical_nodes")
    
    # With priority routing for flows, original clusPR
    survivability.survivability(copy.deepcopy(G), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr_sorted_P), copy.deepcopy(G_geofail_initial_P), copy.deepcopy(failed_nodes_P), nf_service_rate, originalClusPR = True, priority = True, physical = True, filename = "clusPRI_physical_nodes")

    # With physical isolation using modified clusPR 
    survivability.survivability(copy.deepcopy(G), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr_isol_P), copy.deepcopy(G_geofail_initial_P), copy.deepcopy(failed_nodes_P), nf_service_rate, originalClusPR = False, priority = True, physical = True, filename = "clusPR_physical_isolation")


    # Print amounts of NFs instantiated
    print(f'Amounts of NFs instatiated for original clusPR: {clusPR_nf_amount_P}')
    print(f'Amounts of NFs instatiated for sorted flows clusPR: {clusPR_sorted_nf_amount_P}')
    


    
    ### WITH VIRTUAL NODES ###

        # INITIALIZE CLUSPR VERSIONS
    # Apply ClusPR_original
    G_cluspr, flows_cluspr = cluspr_low_resource.apply_cluspr_original(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    clusPR_nf_amount = plotsSurvivability.print_total_amount_of_nfs(G_cluspr)
    clusPR_nf_utilization = plotsSurvivability.get_utilization_list(G_cluspr)
    initial_performance_v = evaluateSurvivability.check_flows(flows_cluspr)
    initial_performance_vc = evaluateSurvivability.check_priority_flows(flows_cluspr,1)
    
    #sys.exit()

    # Sort flows for priority routing
    flows_initial_sorted = topology_uninett.sort_sd_pairs(copy.deepcopy(flows_initial))
    
    # Apply ClusPR_original on sorted flow list
    G_cluspr_sorted, flows_cluspr_sorted = cluspr_low_resource.apply_cluspr_original(copy.deepcopy(G_initial), copy.deepcopy(flows_initial_sorted), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    clusPR_sorted_nf_amount = plotsSurvivability.print_total_amount_of_nfs(G_cluspr_sorted)
    clusPR_sorted_nf_utilization = plotsSurvivability.get_utilization_list(G_cluspr_sorted)


    # Split flow per slice for clusPerSlice
    sd_pairs_split_by_slice = {}
    for sd_pair in flows_initial_sorted:
        slice_id = sd_pair['slice'].ID

        if slice_id in sd_pairs_split_by_slice:
            sd_pairs_split_by_slice[slice_id].append(sd_pair)
        else:
            sd_pairs_split_by_slice[slice_id] = [sd_pair]
    
    flows_split_by_slice = list(sd_pairs_split_by_slice.values())

    # Apply clusPerSlice
    flows_clusperslice = []
    G_clusperslice = copy.deepcopy(G_initial)
    for flow_list in flows_split_by_slice:
        slice_id = flow_list[0]['slice'].ID
        G_clusperslice, flows_clusperslice_part = cluspr_low_resource.apply_cluspr(copy.deepcopy(G_clusperslice), copy.deepcopy(flow_list), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate, slice_id)
        #G_normal = copy.deepcopy(G_normal)
        flows_clusperslice.extend(copy.deepcopy(flows_clusperslice_part))
        cluspr_low_resource.reset_weight_and_required_nfs(G_clusperslice)
    
    clusPerSlice_nf_amount = plotsSurvivability.print_total_amount_of_nfs(G_clusperslice)
    clusPerSlice_nf_utilization = plotsSurvivability.get_utilization_list(G_clusperslice)


    # SIMULATE GEOGRAPHICAL FAILURE

    failure_center = failure_area

    #G_geofail_initial, G_geofail_shortest_paths, G_geofail_shortest_path_lengths, G_geofail_blue_nodes, failed_nodes, failed_edges = failures.geographical_failure(copy.deepcopy(G_initial), failure_center, failure_radius)
    G_geofail_initial, failed_nodes = failuresSurvivability.geographical_failure(copy.deepcopy(G_initial), failure_center, failure_radius)
    
    # SURVIVABILITY ASSESSMENT


    # Without priority routing for flows, original clusPR
    survivability.survivability(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr), copy.deepcopy(G_geofail_initial), copy.deepcopy(failed_nodes), nf_service_rate, originalClusPR = True, priority = False, physical = False, filename = "original_clusPR_virtual_nodes")
    
    # With priority routing for flows, original clusPR
    survivability.survivability(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr_sorted), copy.deepcopy(G_geofail_initial), copy.deepcopy(failed_nodes), nf_service_rate, originalClusPR = True, priority = True, physical = False, filename = "clusPRI_virtual_nodes")
    
    # With virtualization isolation, clusPerSlice

    survivability.survivability(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(flows_clusperslice), copy.deepcopy(G_geofail_initial), copy.deepcopy(failed_nodes), nf_service_rate, originalClusPR = False, priority = True, physical = False,filename = "virtual_isolation")

    # Print amounts of NFs instantiated
    print(f'Amounts of NFs instatiated for original clusPR: {clusPR_nf_amount}')
    print(f'Amounts of NFs instatiated for sorted flows clusPR: {clusPR_sorted_nf_amount}')
    print(f'Amounts of NFs instatiated for clusPerSlice: {clusPerSlice_nf_amount}')

    #plotsSurvivability.plot_amount_of_nfs_total(clusPR_nf_amount_P, clusPR_sorted_nf_amount_P, clusPR_nf_amount, clusPR_sorted_nf_amount, clusPerSlice_nf_amount, 'amounts_of_nfs_geographical_failure')
    
    #plotsSurvivability.plot_utilization_cdf(clusPR_nf_utilization_P, clusPR_sorted_nf_utilization_P, clusPR_nf_utilization, clusPR_sorted_nf_utilization, clusPerSlice_nf_utilization, 'nf_utilization_CDF_geographical_failure')

    print(f'Initial performance physical total: {initial_performance_p} Critical: {initial_performance_pc} Initial performance virtual total: {initial_performance_v} Critical: {initial_performance_vc}')

    failed_nodes_physical = []
    for node in failed_nodes:
        if len(str(node)) < 3 or str(node)[0] == 1:
            failed_nodes_physical.append(node)

    n1 = len(failed_nodes_physical)
    n2 = len(failed_nodes_physical)
    n3 = len(failed_nodes_physical)
    mean_repair_time = 5
    # time_points1 = np.linspace(0, n1 * mean_repair_time * 1.5, 100)
    # time_points2 = np.linspace(0, n2 * mean_repair_time * 1.5, 100)
    # time_points3 = np.linspace(0, n3 * mean_repair_time * 1.5, 100)
    time_points1 = np.linspace(0, n1 * mean_repair_time * 2, 100)
    time_points2 = np.linspace(0, n2 * mean_repair_time * 2, 100)
    time_points3 = np.linspace(0, n3 * mean_repair_time * 2, 100)

    plotsSurvivability.plot_performance_curves_combined(time_points1, 'new_data/original_clusPR_physical_nodes_critical', time_points2, 'new_data/original_clusPR_virtual_nodes_critical', time_points3, 'new_data/virtual_isolation_critical', 'combined_performance_curve_geographical_failure')

    original_clusPR_curve = plotsSurvivability.read_file_and_put_in_list('new_data/original_clusPR_physical_nodes_critical')
    virtual_clusPR_curve = plotsSurvivability.read_file_and_put_in_list('new_data/original_clusPR_virtual_nodes_critical')
    virtual_isolation_curve = plotsSurvivability.read_file_and_put_in_list('new_data/virtual_isolation_critical')

    original_clusPR_util = clusPR_nf_utilization_P
    virtual_clusPR_util = clusPR_nf_utilization
    virtual_isolation_util = clusPerSlice_nf_utilization

    original_clusPR_nfs = sum(clusPR_nf_amount_P.values())
    virtual_clusPR_nfs = sum(clusPR_nf_amount.values())
    virtual_isolation_nfs = sum(clusPerSlice_nf_amount.values())

    return original_clusPR_curve, virtual_clusPR_curve, virtual_isolation_curve, original_clusPR_util, virtual_clusPR_util, virtual_isolation_util, original_clusPR_nfs, virtual_clusPR_nfs, virtual_isolation_nfs, len(failed_nodes_physical)

if __name__ == "__main__":
    original_clusPR_curve, virtual_clusPR_curve, virtual_isolation_curve, original_clusPR_util, virtual_clusPR_util, virtual_isolation_util, original_clusPR_nfs, virtual_clusPR_nfs, virtual_isolation_nfs, number_of_failed_nodes = main()
    
