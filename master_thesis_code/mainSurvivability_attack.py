# Import files from project
import cluspr_low_resource
import topology_uninett
import plotsSurvivability
import evaluateSurvivability
import failuresSurvivability
import survivability_attack
import general_tools

import matplotlib.pyplot as plt
import numpy as np
import random
import copy

id_generator = general_tools.ID_generator()

### VALUES ###

# Speed at which the network functions operate
nf_service_rate = 70 #30 # Mbps

# {Total number of flows, flows per slice, flows of VoD app type, flows of VoIP type, flows of LVD type, flows of FD type}
number_of_flows = {'total_flows': 600, 'flows_per_slice': 150, 'VoD': 22, 'VoIP': 105, 'LVD': 15, 'FD': 8} #for Geant2012, since it has less nodes
#number_of_flows = {'total_flows': 1000, 'flows_per_slice': 250, 'VoD': 37, 'VoIP': 175, 'LVD': 25, 'FD': 13} #for Uninett2011


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

# Path to the network layout file
#gml_file_path = 'surviveNet/topology_zoo/Uninett2011.gml'
gml_file_path = 'surviveNet/topology_zoo/Geant2012.gml'


def main():
    # INITIAL SETUP
    # Load the network layout

    split_number = 4 #number of virtual nodes per core node
    G, G_initial, shortest_paths_initial, shortest_path_lengths_initial, blue_nodes_initial, flows_initial = topology_uninett.generate_topology_and_flows_virtualized(gml_file_path, number_of_flows, application_types, slices, split_number)

     
     ### WITOUT VIRTUAL NODES ###

        # INITIALIZE CLUSPR VERSIONS
    # Apply ClusPR_original
    G_cluspr_P, flows_cluspr_P = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G), copy.deepcopy(flows_initial), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    clusPR_nf_amount_P = plotsSurvivability.print_total_amount_of_nfs_physical(G_cluspr_P)
    clusPR_nf_utilization_P = plotsSurvivability.get_utilization_list(G_cluspr_P)
    
    
    # Sort flows for priority routing
    flows_initial_sorted_P = topology_uninett.sort_sd_pairs(copy.deepcopy(flows_initial))
    
    # Apply ClusPR_original on sorted flow list
    G_cluspr_sorted_P, flows_cluspr_sorted_P = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G), copy.deepcopy(flows_initial_sorted_P), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    clusPR_sorted_nf_amount_P = plotsSurvivability.print_total_amount_of_nfs_physical(G_cluspr_sorted_P)
    clusPR_sorted_nf_utilization_P = plotsSurvivability.get_utilization_list(G_cluspr_sorted_P)

        # SIMULATE ATTACK FROM SLICE/FLOWS
    
    # Corrupt slice
    corrupt_slice = 2
    number_of_corrupt_flows = 60#60 #100

    # Corrupt flow and nodes in original clusPR
    possible_corrupt_flows_original_P = [flow for flow in flows_cluspr_P if flow['slice'].ID == corrupt_slice]
    corrupt_flows_original_P = random.choices(possible_corrupt_flows_original_P, k=number_of_corrupt_flows)
    failed_nodes_original_P = []
    [failed_nodes_original_P.extend(flow['flow_path']) for flow in corrupt_flows_original_P if 'flow_path' in flow]
    failed_nodes_original_P = [node for node in failed_nodes_original_P if len(str(node))<3]
    failed_nodes_original_P = list(set(failed_nodes_original_P))

    # Corrupt flow and nodes in sorted clusPR
    possible_corrupt_flows_sorted_P = [flow for flow in flows_cluspr_sorted_P if flow['slice'].ID == corrupt_slice]
    corrupt_flows_sorted_P = random.choices(possible_corrupt_flows_sorted_P, k=number_of_corrupt_flows)
    failed_nodes_sorted_P = []
    [failed_nodes_sorted_P.extend(flow['flow_path']) for flow in corrupt_flows_sorted_P if 'flow_path' in flow]
    failed_nodes_sorted_P = [node for node in failed_nodes_sorted_P if len(str(node))<3]
    failed_nodes_sorted_P = list(set(failed_nodes_sorted_P))

    print(f'Complete path original: {failed_nodes_original_P}')
    print(f'Complete path sorted: {failed_nodes_sorted_P}')


    # Show the original network layout
    #plotsSurvivability.plot_graph(G_cluspr)

    # Highlight the area affected by the failure on the network layout
    #plotsSurvivability.plot_graph(G_cluspr, True, failure_center, failure_radius)

    #G_geofail_initial, G_geofail_shortest_paths, G_geofail_shortest_path_lengths, G_geofail_blue_nodes, failed_nodes, failed_edges = failures.geographical_failure(copy.deepcopy(G_initial), failure_center, failure_radius)
    #G_geofail_initial, failed_nodes = failuresSurvivability.geographical_failure(copy.deepcopy(G_initial), failure_center, failure_radius)
    G_fail_original_P = failuresSurvivability.flow_attack_physical(copy.deepcopy(G), failed_nodes_original_P)
    G_fail_sorted_P = failuresSurvivability.flow_attack_physical(copy.deepcopy(G), failed_nodes_sorted_P)



    # Show the network layout after removing the failed parts
    #plotsSurvivability.plot_graph(G_geofail_initial) 
    
        # ASSESS INITIAL PERFORMANCE
    inital_performance_cluspr_physical = []
    inital_performance_clusPRI_physical = []

    inital_performance_cluspr_physical.append(evaluateSurvivability.check_priority_flows(flows_cluspr_P, 1))
    inital_performance_cluspr_physical.append(evaluateSurvivability.check_priority_flows_failure(flows_cluspr_P, 1, failed_nodes_original_P))

    G_not_used, flows_cluspr_P_after_fail = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G_fail_original_P), copy.deepcopy(flows_initial), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    inital_performance_cluspr_physical.append(evaluateSurvivability.check_priority_flows(flows_cluspr_P_after_fail, 1))
    
    
    inital_performance_clusPRI_physical.append(evaluateSurvivability.check_priority_flows(flows_cluspr_sorted_P, 1))
    inital_performance_clusPRI_physical.append(evaluateSurvivability.check_priority_flows_failure(flows_cluspr_sorted_P, 1, failed_nodes_sorted_P))

    G_not_used, flows_clusPRI_P_after_fail = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G_fail_sorted_P), copy.deepcopy(flows_initial_sorted_P), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    inital_performance_clusPRI_physical.append(evaluateSurvivability.check_priority_flows(flows_clusPRI_P_after_fail, 1))

    
    print(f'List of percentages of priority flows in original clusPRI no isolation meeting requirements before fail, after fail and after cluspr: {inital_performance_clusPRI_physical}')
    print(f'List of percentages of priority flows in original clusPR no isolation meeting requirements before fail, after fail and after cluspr: {inital_performance_cluspr_physical}')

    #sys.exit()

        # SURVIVABILITY ASSESSMENT


    # Without priority routing for flows, original clusPR
    survivability_attack.survivability(copy.deepcopy(G), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr_P), copy.deepcopy(G_fail_original_P), copy.deepcopy(failed_nodes_original_P), nf_service_rate, originalClusPR=True, priority=False, physical=True, filename="original_clusPR_physical_nodes_attack")
    
    # With priority routing for flows, original clusPR
    #survivability_attack.survivability(copy.deepcopy(G), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr_sorted_P), copy.deepcopy(G_fail_sorted_P), copy.deepcopy(failed_nodes_sorted_P), nf_service_rate, originalClusPR=True, priority=True, physical= True, filename="clusPRI_physical_nodes_attack")

    # Print amounts of NFs instantiated
    print(f'Amounts of NFs instatiated for original clusPR: {clusPR_nf_amount_P}')
    print(f'Amounts of NFs instatiated for sorted flows clusPR: {clusPR_sorted_nf_amount_P}')
    
    ### WITH VIRTUAL NODES ###

        # INITIALIZE CLUSPR VERSIONS
    # Apply ClusPR_original
    G_cluspr, flows_cluspr = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    clusPR_nf_amount = plotsSurvivability.print_total_amount_of_nfs(G_cluspr)
    clusPR_nf_utilization = plotsSurvivability.get_utilization_list(G_cluspr)
    
    # Sort flows for priority routing
    flows_initial_sorted = topology_uninett.sort_sd_pairs(copy.deepcopy(flows_initial))
    
    # Apply ClusPR_original on sorted flow list
    G_cluspr_sorted, flows_cluspr_sorted = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G_initial), copy.deepcopy(flows_initial_sorted), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
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
        G_clusperslice, flows_clusperslice_part = cluspr_low_resource.apply_cluspr_A(copy.deepcopy(G_clusperslice), copy.deepcopy(flow_list), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate, slice_id)
        #G_normal = copy.deepcopy(G_normal)
        flows_clusperslice.extend(copy.deepcopy(flows_clusperslice_part))
        cluspr_low_resource.reset_weight_and_required_nfs(G_clusperslice)
    
    clusPerSlice_nf_amount = plotsSurvivability.print_total_amount_of_nfs(G_clusperslice)
    clusPerSlice_nf_utilization = plotsSurvivability.get_utilization_list(G_clusperslice)
    
        # SIMULATE ATTACK FROM ONE SPECIFI FLOW IN A SPECIFIC SLICE
        # The flow is corrupting each VIRTUAL NODE it visits in the network topology
    
    # Corrupt slice
    corrupt_slice = 2
    number_of_corrupt_flows = 60 #60 #100
    
    # Corrupt flow and nodes in original clusPR
    possible_corrupt_flows_original = [flow for flow in flows_cluspr if flow['slice'].ID == corrupt_slice]
    corrupt_flows_original = random.choices(possible_corrupt_flows_original, k=number_of_corrupt_flows)
    failed_nodes_original = []
    [failed_nodes_original.extend(flow['complete_path']) for flow in corrupt_flows_original if 'complete_path' in flow]
    failed_nodes_original = [node for node in failed_nodes_original if str(node)[0]=='2' and len(str(node))>3]
    failed_nodes_original = list(set(failed_nodes_original))

    # Corrupt flow and nodes in sorted clusPR
    possible_corrupt_flows_sorted = [flow for flow in flows_cluspr_sorted if flow['slice'].ID == corrupt_slice]
    corrupt_flows_sorted = random.choices(possible_corrupt_flows_sorted, k=number_of_corrupt_flows)
    failed_nodes_sorted = []
    [failed_nodes_sorted.extend(flow['complete_path']) for flow in corrupt_flows_sorted if 'complete_path' in flow]
    failed_nodes_sorted = [node for node in failed_nodes_sorted if str(node)[0]=='2' and len(str(node))>3]
    failed_nodes_sorted = list(set(failed_nodes_sorted))

    # Corrupt flow and nodes in virtual isolation clusPR
    possible_corrupt_flows_isol = [flow for flow in flows_clusperslice if flow['slice'].ID == corrupt_slice]
    corrupt_flows_isol = random.choices(possible_corrupt_flows_isol, k=number_of_corrupt_flows)
    failed_nodes_isol = []
    [failed_nodes_isol.extend(flow['complete_path']) for flow in corrupt_flows_isol if 'complete_path' in flow]
    failed_nodes_isol = [node for node in failed_nodes_isol if str(node)[0]=='2' and len(str(node))>3]
    failed_nodes_isol = list(set(failed_nodes_isol))

    print(f'Number of failed node original: {len(failed_nodes_original)}')
    print(f'Number of failed nodes sorted: {len(failed_nodes_sorted)}')
    print(f'Number of failed nodes: {len(failed_nodes_isol)}')

    
    # Show the original network layout
    #plotsSurvivability.plot_graph(G_cluspr)

    # Highlight the area affected by the failure on the network layout
    #plotsSurvivability.plot_graph(G_cluspr, True, failure_center, failure_radius)

    #G_geofail_initial, G_geofail_shortest_paths, G_geofail_shortest_path_lengths, G_geofail_blue_nodes, failed_nodes, failed_edges = failures.geographical_failure(copy.deepcopy(G_initial), failure_center, failure_radius)
    #G_geofail_initial, failed_nodes = failuresSurvivability.geographical_failure(copy.deepcopy(G_initial), failure_center, failure_radius)
    G_fail_original = failuresSurvivability.flow_attack(copy.deepcopy(G_initial), failed_nodes_original)
    G_fail_sorted = failuresSurvivability.flow_attack(copy.deepcopy(G_initial), failed_nodes_sorted)
    G_fail_isol = failuresSurvivability.flow_attack(copy.deepcopy(G_initial), failed_nodes_isol)


    # Show the network layout after removing the failed parts
    #plotsSurvivability.plot_graph(G_geofail_initial) 

    # ASSES INITIAL PERFORMANCE
    inital_performance_cluspr_virtual = []
    inital_performance_clusPRI_virtual = []
    inital_performance_clusperslice_virtual = []

    inital_performance_cluspr_virtual.append(evaluateSurvivability.check_priority_flows(flows_cluspr, 1))
    inital_performance_cluspr_virtual.append(evaluateSurvivability.check_priority_flows_failure(flows_cluspr, 1, failed_nodes_original))

    G_not_used, flows_cluspr_V_after_fail = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G_fail_original), copy.deepcopy(flows_initial), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    inital_performance_cluspr_virtual.append(evaluateSurvivability.check_priority_flows(flows_cluspr_V_after_fail, 1))
    
    

    inital_performance_clusPRI_virtual.append(evaluateSurvivability.check_priority_flows(flows_cluspr_sorted, 1))
    inital_performance_clusPRI_virtual.append(evaluateSurvivability.check_priority_flows_failure(flows_cluspr_sorted, 1, failed_nodes_sorted))

    G_not_used, flows_clusPRI_V_after_fail = cluspr_low_resource.apply_cluspr_original_A(copy.deepcopy(G_fail_sorted), copy.deepcopy(flows_initial_sorted), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate)
    inital_performance_clusPRI_virtual.append(evaluateSurvivability.check_priority_flows(flows_clusPRI_V_after_fail, 1))



    inital_performance_clusperslice_virtual.append(evaluateSurvivability.check_priority_flows(flows_clusperslice, 1))
    inital_performance_clusperslice_virtual.append(evaluateSurvivability.check_priority_flows_failure(flows_clusperslice, 1, failed_nodes_isol))

    flows_clusperslice_V_after_fail = []
    G_not_used = copy.deepcopy(G_fail_isol)
    for flow_list in flows_split_by_slice:
        slice_id = flow_list[0]['slice'].ID
        G_not_used, flows_clusperslice_V_after_fail_part = cluspr_low_resource.apply_cluspr_A(copy.deepcopy(G_not_used), copy.deepcopy(flow_list), copy.deepcopy(blue_nodes_initial), shortest_paths_initial, shortest_path_lengths_initial, nf_service_rate, slice_id)
        flows_clusperslice_V_after_fail.extend(copy.deepcopy(flows_clusperslice_V_after_fail_part))
        cluspr_low_resource.reset_weight_and_required_nfs(G_not_used)

    inital_performance_clusperslice_virtual.append(evaluateSurvivability.check_priority_flows(flows_clusperslice_V_after_fail, 1))

    
    print(f'List of percentages of priority flows in original clusPR network function isolation meeting requirements before fail, after fail and after cluspr: {inital_performance_cluspr_virtual}')
    print(f'List of percentages of priority flows in original clusPRI network function isolation meeting requirements before fail, after fail and after cluspr: {inital_performance_clusPRI_virtual}')
    print(f'List of percentages of priority flows in clusperslice network function isolation meeting requirements before fail, after fail and after cluspr: {inital_performance_clusperslice_virtual}')

    #plotsSurvivability.plot_initial_performance(inital_performance_cluspr_physical, inital_performance_clusPRI_physical, inital_performance_cluspr_virtual, inital_performance_clusPRI_virtual, inital_performance_clusperslice_virtual, 'initial_performance_uninett')


        # SURVIVABILITY ASSESSMENT


    # Without priority routing for flows, original clusPR
    survivability_attack.survivability(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr), copy.deepcopy(G_fail_original), copy.deepcopy(failed_nodes_original), nf_service_rate, originalClusPR=True, priority=False, physical=False, filename="original_clusPR_virtual_nodes_attack")
    
    # With priority routing for flows, original clusPR
    #survivability_attack.survivability(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(flows_cluspr_sorted), copy.deepcopy(G_fail_sorted), copy.deepcopy(failed_nodes_sorted), nf_service_rate, originalClusPR=True, priority=True, physical=False, filename="clusPRI_virtual_nodes_attack")
    
    # With virtualization isolation, clusPerSlice

    survivability_attack.survivability(copy.deepcopy(G_initial), copy.deepcopy(flows_initial), copy.deepcopy(flows_clusperslice), copy.deepcopy(G_fail_isol), copy.deepcopy(failed_nodes_isol), nf_service_rate, originalClusPR=False, priority=True, physical=False, filename="virtual_isolation_attack")

    # Print amounts of NFs instantiated
    print(f'Amounts of NFs instatiated for original clusPR: {clusPR_nf_amount}')
    print(f'Amounts of NFs instatiated for sorted flows clusPR: {clusPR_sorted_nf_amount}')
    print(f'Amounts of NFs instatiated for clusPerSlice: {clusPerSlice_nf_amount}')
    
    #plotsSurvivability.plot_amount_of_nfs_total(clusPR_nf_amount_P, clusPR_sorted_nf_amount_P, clusPR_nf_amount, clusPR_sorted_nf_amount, clusPerSlice_nf_amount, 'uninett_amounts_of_nfs_attack')
    
    #plotsSurvivability.plot_utilization_cdf(clusPR_nf_utilization_P, clusPR_sorted_nf_utilization_P, clusPR_nf_utilization, clusPR_sorted_nf_utilization, clusPerSlice_nf_utilization, 'uninett_nf_utilization_CDF')

    n1 = len(failed_nodes_original_P)
    n2 = len(failed_nodes_original)
    n3 = len(failed_nodes_isol)
    mean_repair_time = 0.1
    time_points1 = np.linspace(0, n1 * mean_repair_time * 2.5, 100)
    time_points2 = np.linspace(0, n2 * mean_repair_time * 2.5, 100)
    time_points3 = np.linspace(0, n3 * mean_repair_time * 2.5, 100)

    plotsSurvivability.plot_performance_curves_combined(time_points1, 'new_data/original_clusPR_physical_nodes_attack_critical', time_points2, 'new_data/original_clusPR_virtual_nodes_attack_critical', time_points3, 'new_data/virtual_isolation_attack_critical', 'combined_performance_curve_1')

    original_clusPR_curve = plotsSurvivability.read_file_and_put_in_list('new_data/original_clusPR_physical_nodes_attack_critical')
    virtual_clusPR_curve = plotsSurvivability.read_file_and_put_in_list('new_data/original_clusPR_virtual_nodes_attack_critical')
    virtual_isolation_curve = plotsSurvivability.read_file_and_put_in_list('new_data/virtual_isolation_attack_critical')

    original_clusPR_util = clusPR_nf_utilization_P
    virtual_clusPR_util = clusPR_nf_utilization
    virtual_isolation_util = clusPerSlice_nf_utilization

    original_clusPR_nfs = sum(clusPR_nf_amount_P.values())
    virtual_clusPR_nfs = sum(clusPR_nf_amount.values())
    virtual_isolation_nfs = sum(clusPerSlice_nf_amount.values())

    initial_performance_original = inital_performance_cluspr_physical
    initial_performance_virt = inital_performance_cluspr_virtual
    initial_performance_virt_isol = inital_performance_clusperslice_virtual

    return original_clusPR_curve, virtual_clusPR_curve, virtual_isolation_curve, original_clusPR_util, virtual_clusPR_util, virtual_isolation_util, original_clusPR_nfs, virtual_clusPR_nfs, virtual_isolation_nfs, initial_performance_original, initial_performance_virt, initial_performance_virt_isol


if __name__ == "__main__":
    original_clusPR_curve, virtual_clusPR_curve, virtual_isolation_curve, original_clusPR_util, virtual_clusPR_util, virtual_isolation_util, original_clusPR_nfs, virtual_clusPR_nfs, virtual_isolation_nfs, initial_performance_original, initial_performance_virt, initial_performance_virt_isol = main()
