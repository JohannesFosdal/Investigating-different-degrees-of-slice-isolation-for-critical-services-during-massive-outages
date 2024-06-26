import matplotlib.pyplot as plt



def check_flows(flows):
    met_delay = 0
    for flow in flows:
        try:
            if flow['actual_delay'] <= flow['delay_requirement']:
                met_delay += 1
        except:
            # Flow has no path (is dropped)
            continue
    return (met_delay / len(flows)) * 100

def check_priority_flows(flows, priority):
    met_delay = 0
    total = 0
    for flow in flows:
        if flow['priority'] != priority:
            continue
        total += 1
        try:
            if flow['actual_delay'] <= flow['delay_requirement']:
                met_delay += 1
        except:
            # Flow has no path (is dropped)
            continue
    return (met_delay / total) * 100

def plot_cdf(data, capacity=20):
    data = [x / capacity for x in data]

    data_sorted = sorted(data)

    n = len(data_sorted)
    cdf_y = [(i + 1) / n for i in range(n)]

    plt.plot(data_sorted, cdf_y, marker='.', linestyle='none')
    
    plt.xlabel('NF Utilization (%)')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF) of NF Utilizations')
    plt.grid(True)  # Add grid lines for readability

    plt.show()



def plot_cdf(data, capacity=20):
    # need to be made
    print('hei')

def plot_cdf_utilization(all_iterations_nodes):
    # This will hold all utilizations across all iterations
    all_utilizations = []

    # Iterate over list of node data from each iteration
    for nodes in all_iterations_nodes:
        for node in nodes:
            attr = node[1]
            if attr['color'] != 'cornflowerblue':
                for util in attr['nf_utilization'].values():
                    # This collects the utilization data for each iteration
                    all_utilizations.append(util)

    # Now, this list contains all utilization data across all iterations
    plot_cdf(all_utilizations)


def find_bottleneck_flows_and_applications(G, flows, application_types):
    # Find the edge with the highest throughput
    max_throughput = 0
    bottleneck_edge = None

    for u, v, data in G.edges(data=True):
        if data['throughput'] > max_throughput:
            max_throughput = data['throughput']
            bottleneck_edge = (u, v)

    # Find flows and count applications that go through the bottleneck
    bottleneck_flows = []

    application_counts = {} 
    for app_type in application_types:
        application_counts[app_type] = 0

    for flow in flows:
        try:
            flow_path = flow['complete_path']
        except:
            continue

        # Check if the bottleneck edge is in the flow path
        for i in range(len(flow_path) - 1):
            if (bottleneck_edge[0], bottleneck_edge[1]) == (flow_path[i], flow_path[i + 1]):
                bottleneck_flows.append(flow) # NOTE: this will count a flow twice if it passes through the same link twice
                application = flow['application']
                application_counts[application] += 1

    return bottleneck_edge, max_throughput, bottleneck_flows, application_counts



def check_priority_flows_failure(flows, priority, failed_nodes):
    met_delay = 0
    total = 0
    for flow in flows:
        if flow['priority'] != priority:
            continue
        total += 1
        try:
            if flow['actual_delay'] <= flow['delay_requirement'] and not any(num in failed_nodes for num in flow['flow_path']):
                met_delay += 1
        except:
            # Flow has no path (is dropped)
            continue
    return (met_delay / total) * 100
