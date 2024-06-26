from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy

import cluspr_low_resource
import evaluateSurvivability
import topology_uninett
import plotsSurvivability
import failuresSurvivability

def compute_state_probabilities(n, lambd, t):
    # Initialize the transition rate matrix Q
    Q = np.zeros((n+1, n+1))
    
    # Populate the diagonal
    np.fill_diagonal(Q, -lambd)
    Q[0, 0] = 0  # State 0 is absorbing, so it has no rate out
    
    # Populate the subdiagonal
    for i in range(1, n+1):
        Q[i, i-1] = lambd
    
    # Compute the matrix exponential of Qt
    P_t = expm(Q * t)
    
    # Extract the probabilities of being in each state at time t,
    # starting from state n
    state_probabilities_at_t = P_t[n, :]

    
    return state_probabilities_at_t.tolist()


# Returns state rewards for critical traffic and for non-critical traffic
def calculate_state_rewards(G_initial, flows_initial, failed_nodes_sorted, failed_nodes_total, nf_service_rate, originalClusPR, priority, physical):
    # repair_order is a list of nodes in the order they will be repaired.
    critical_state_rewards = {}
    uncritical_state_rewards = {}
    total_state_rewards = {}

    flows_initial_sorted = topology_uninett.sort_sd_pairs(copy.deepcopy(flows_initial))

    sd_pairs_split_by_slice = {}
    for sd_pair in flows_initial_sorted:
        slice_id = sd_pair['slice'].ID

        if slice_id in sd_pairs_split_by_slice:
            sd_pairs_split_by_slice[slice_id].append(sd_pair)
        else:
            sd_pairs_split_by_slice[slice_id] = [sd_pair]
    
    flows_split_by_slice = list(sd_pairs_split_by_slice.values())
            

    if physical:
        for i in range(len(failed_nodes_sorted) + 1): 
            G_current = copy.deepcopy(G_initial)
            current_failed = failed_nodes_sorted[i:]
            failuresSurvivability.flow_attack_physical(G_current, current_failed)
            
            shortest_paths, shortest_path_lengths, blue_nodes = topology_uninett.analyze_G(G_current)
            #flows_initial_sorted = topology_uninett.sort_sd_pairs(copy.deepcopy(flows_initial))

            if priority:
                G_cluspr, flows_cluspr = cluspr_low_resource.apply_cluspr_original_A(G_current, copy.deepcopy(flows_initial_sorted), blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate)
            else:
                G_cluspr, flows_cluspr = cluspr_low_resource.apply_cluspr_original_A(G_current, copy.deepcopy(flows_initial), blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate)

            # Calculate the reward for the current graph state - current: percentage of critical flows meeting their delay requirement
            critical_reward = evaluateSurvivability.check_priority_flows(flows_cluspr, 1)
            uncritical_reward = evaluateSurvivability.check_priority_flows(flows_cluspr, 2)
            total_reward = evaluateSurvivability.check_flows(flows_cluspr)

            # Store the reward in the state_rewards dictionary
            critical_state_rewards[i] = critical_reward
            uncritical_state_rewards[i] = uncritical_reward
            total_state_rewards[i] = total_reward


    else:
        for i in range(len(failed_nodes_sorted) + 1):
            # Create a copy of the initial graph so that we can manipulate it without altering the original
            G_current = copy.deepcopy(G_initial)
            
            # For each state, calculate the reward
            # A state is defined by the number of nodes that are still failed
            current_failed = failed_nodes_sorted[i:]
            print(f'Amount of current failed nodes: {len(current_failed)}')

            print(f'Amount of nodes before failure: {len(G_current.nodes)}')
            # Remove the currently failed nodes from the graph copy
            #G_current.remove_nodes_from(current_failed)
            failuresSurvivability.flow_attack(G_current, current_failed)
            print(f'Amount of nodes after failure: {len(G_current.nodes)}')

    
            shortest_paths, shortest_path_lengths, blue_nodes = topology_uninett.analyze_G(G_current)

            #flows_initial_sorted = topology_uninett.sort_sd_pairs(copy.deepcopy(flows_initial))

            if originalClusPR:
                if priority:
                    G_cluspr, flows_cluspr = cluspr_low_resource.apply_cluspr_original_A(G_current, copy.deepcopy(flows_initial_sorted), blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate)

                else:
                    G_cluspr, flows_cluspr = cluspr_low_resource.apply_cluspr_original_A(G_current, copy.deepcopy(flows_initial), blue_nodes, shortest_paths, shortest_path_lengths, nf_service_rate)
            
            if not originalClusPR:
                # Split flow per slice for clusPerSlice
                # sd_pairs_split_by_slice = {}
                # for sd_pair in flows_initial_sorted:
                #     slice_id = sd_pair['slice'].ID

                #     if slice_id in sd_pairs_split_by_slice:
                #         sd_pairs_split_by_slice[slice_id].append(sd_pair)
                #     else:
                #         sd_pairs_split_by_slice[slice_id] = [sd_pair]
                
                # flows_split_by_slice = list(sd_pairs_split_by_slice.values())

                # Apply clusPerSlice
                flows_clusperslice = []
                G_clusperslice = copy.deepcopy(G_current)
                for flow_list in flows_split_by_slice:
                    slice_id = flow_list[0]['slice'].ID
                    G_clusperslice, flows_clusperslice_part = cluspr_low_resource.apply_cluspr_A(copy.deepcopy(G_clusperslice), copy.deepcopy(flow_list), copy.deepcopy(blue_nodes), shortest_paths, shortest_path_lengths, nf_service_rate, slice_id)
                    #G_normal = copy.deepcopy(G_normal)
                    flows_clusperslice.extend(copy.deepcopy(flows_clusperslice_part))
                    cluspr_low_resource.reset_weight_and_required_nfs(G_clusperslice)
                
                flows_cluspr = copy.deepcopy(flows_clusperslice)

            # Calculate the reward for the current graph state - current: percentage of critical flows meeting their delay requirement
            critical_reward = evaluateSurvivability.check_priority_flows(flows_cluspr, 1)
            uncritical_reward = evaluateSurvivability.check_priority_flows(flows_cluspr, 2)
            total_reward = evaluateSurvivability.check_flows(flows_cluspr)
            
            # Store the reward in the state_rewards dictionary
            critical_state_rewards[i] = critical_reward
            uncritical_state_rewards[i] = uncritical_reward
            total_state_rewards[i] = total_reward

    n = len(failed_nodes_sorted)
    # if physical:
    #     n = (len(failed_nodes_sorted) + 1) * 4 - 1
    
    # Sort state rewards so they correspond to the state probabilities
    corrected_critical_state_rewards = {n - key: value for key, value in critical_state_rewards.items()}
    corrected_uncritical_state_rewards = {n - key: value for key, value in uncritical_state_rewards.items()}
    corrected_total_state_rewards = {n - key: value for key, value in total_state_rewards.items()}

    return corrected_critical_state_rewards, corrected_uncritical_state_rewards, corrected_total_state_rewards

# Define the function to compute the weighted sum S(t)
def compute_survivability(n, lambd, t, state_rewards):
    # Calculate the state probabilities at time t
    state_probabilities = compute_state_probabilities(n, lambd, t)

    # Compute the weighted sum S(t) using the state_rewards
    S_t = sum(state_rewards[i] * p for i, p in enumerate(state_probabilities))

    return S_t

def find_critical_flow_centrality(G, flows_cluspr):
    # Initialize a dictionary to store the critical flow centrality of each node
    critical_flow_centrality = {node: 0 for node in G.nodes()}
    
    # Iterate through each flow in flows_cluspr
    for flow in flows_cluspr:
        # Check if the priority of the flow is 1
        if flow['priority'] == 1:
            # Iterate through each node in the complete path of the flow
            try:
                for node in flow['complete_path']:
                    # Increase the critical flow centrality of the node by 1
                    if node in critical_flow_centrality:
                        critical_flow_centrality[node] += 1
            except:
                # flow does not have a complete path
                continue
    
    return critical_flow_centrality


def survivability(G_initial, flows_initial, flows_cluspr, G_failed, failed_nodes, nf_service_rate, originalClusPR, priority, physical, filename):


    print(f'''
    {filename} --- {filename} --- {filename} --- {filename}

    ''')

    # Set values
    n = len(failed_nodes)
    mean_repair_time = 0.1
    lambd = 1/mean_repair_time
    lambd_physical = 1/(mean_repair_time*4)

    time_points = np.linspace(0, 65 * mean_repair_time * 2.5, 100)

    # Calculate "critical flow centrality" based the flow configuration from before the failure
    critical_flow_centrality = find_critical_flow_centrality(G_initial, flows_cluspr)

    # Sort the failed nodes based on critical centrality 
    failed_nodes_critical_centrality = sorted(failed_nodes, key=lambda x: critical_flow_centrality[x], reverse=True)
    # Calculate state rewards for critical centrality repair order
    critical_state_rewards_critical_centrality, uncritical_state_rewards_critical_centrality, total_state_rewards_critical_centrality = calculate_state_rewards(G_initial, flows_initial, failed_nodes_critical_centrality, failed_nodes, nf_service_rate, originalClusPR, priority, physical)

    # Loop through time points and simulate repairs based on strategy
    #time_points = np.linspace(0, n*mean_repair_time*1.5, 100)
    time_points = np.linspace(0, 65 * mean_repair_time * 2.5, 100)
    performance_curve_critical_centrality = []

    u_performance_curve_critical_centrality = []
    c_performance_curve_critical_centrality = []
    
    
    for t in time_points:
        if physical:
            # Simulate repair and compute performance for critical centrality
            performance_critical_centrality = compute_survivability(n, lambd_physical, t, critical_state_rewards_critical_centrality)
            performance_curve_critical_centrality.append(performance_critical_centrality)

            # Uncritical
            u_performance_critical_centrality = compute_survivability(n, lambd_physical, t, uncritical_state_rewards_critical_centrality)
            u_performance_curve_critical_centrality.append(u_performance_critical_centrality)

            c_performance_critical_centrality = compute_survivability(n, lambd_physical, t, total_state_rewards_critical_centrality)
            c_performance_curve_critical_centrality.append(c_performance_critical_centrality)
        
        else:
            # Simulate repair and compute performance for critical centrality
            performance_critical_centrality = compute_survivability(n, lambd, t, critical_state_rewards_critical_centrality)
            performance_curve_critical_centrality.append(performance_critical_centrality)

            # Uncritical
            u_performance_critical_centrality = compute_survivability(n, lambd, t, uncritical_state_rewards_critical_centrality)
            u_performance_curve_critical_centrality.append(u_performance_critical_centrality)

            c_performance_critical_centrality = compute_survivability(n, lambd, t, total_state_rewards_critical_centrality)
            c_performance_curve_critical_centrality.append(c_performance_critical_centrality)

    target_percentage = 0.95   # This is just an example; can set it to any value wanted
    
    adjusted_target_percentage = (critical_state_rewards_critical_centrality[0]*target_percentage)/100

    print('')
    print(f'Failed nodes: {failed_nodes}')
    print(f'Node failures: {n}')

    print('')
    print(f'Performance right after failure: {critical_state_rewards_critical_centrality[n]}')
    print(f'Performance after all repairs: {critical_state_rewards_critical_centrality[0]}')

    print('')
    # Plot med priority (se forskjell på critical og uncritical)
    if True:
        # Write to file for plotting/processing later
        file_path = 'new_data/' + filename + '_critical'
        with open(file_path, 'w') as file:

            file.write('\ncritical_centrality\n')
            critical_centrality_str = '\n'.join(str(item) for item in performance_curve_critical_centrality)
            file.write(critical_centrality_str)

        file_path = 'new_data/' + filename + '_non_critical'

        with open(file_path, 'w') as file:
            file.write('\nu_critical_centrality\n')
            critical_centrality_str = '\n'.join(str(item) for item in u_performance_curve_critical_centrality)
            file.write(critical_centrality_str)
        
        print("With priority:")
        #plotsSurvivability.plot_performance_curves(time_points, performance_curve_critical_centrality, target_percentage, adjusted_target_percentage, n, mean_repair_time, filename)
        
        filename = filename + '_with_non_critical'
        print('')
        print("With priority, non-critical included")
        plotsSurvivability.plot_performance_curves(time_points, performance_curve_critical_centrality, target_percentage, adjusted_target_percentage, n, mean_repair_time, filename, u_performance_curve_critical_centrality, c_performance_curve_critical_centrality)
