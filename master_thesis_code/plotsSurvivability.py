import networkx as nx
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import pandas as pd
import survivability
import copy

def plot_graph(G, highlight_failure=False, failure_center=None, failure_radius=0):
    """
    Plot the graph with optional highlighting of a failure region
    """
    # Use Seaborn for styling
    sns.set_style("white")
    sns.set_context("talk")

    plt.figure(figsize=(12, 8))

    # Adjusted color for failure region
    failure_region_facecolor = 'crimson'
    failure_region_edgecolor = 'darkred'

    # Define a default fallback color for nodes
    default_node_color = 'skyblue'

    # Define color palette for nodes with specific attributes
    node_colors = {'red': 'darkred'}

    # Plot nodes and edges first
    for node, attr in G.nodes(data=True):
        node_size = 100 if attr.get('color') == 'red' else 30
        node_color = node_colors.get(attr.get('color'), default_node_color)
        plt.scatter(attr['Longitude'], attr['Latitude'], s=node_size, c=node_color, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=2)

    for edge in G.edges():
        lat_start, lon_start = G.nodes[edge[0]]['Latitude'], G.nodes[edge[0]]['Longitude']
        lat_end, lon_end = G.nodes[edge[1]]['Latitude'], G.nodes[edge[1]]['Longitude']
        plt.plot([lon_start, lon_end], [lat_start, lat_end], color='gray', linestyle='-', alpha=0.5, linewidth=1, zorder=1)

    # If highlighting failure, draw the failure region above nodes and links
    if highlight_failure and failure_center:
        failure_region = plt.Circle((failure_center['Longitude'], failure_center['Latitude']),
                                    failure_radius / 111,  # Approx conversion from km to degrees
                                    facecolor=failure_region_facecolor, alpha=0.8, label='Failure Region',
                                    edgecolor=failure_region_edgecolor, linewidth=1.5, zorder=3)
        plt.gca().add_patch(failure_region)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    title = 'Network Graph' + (' with Failure Simulation' if highlight_failure else '')
    #plt.title(title)

    if highlight_failure:
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('figs/topo.pdf')
    plt.show()

def calculate_distance(lat1, lon1, lat2, lon2):
    # Approximate radius of earth in km
    R = 6371.0

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# Used for checking while developing
def plot_state_probabilities(n, mu):

    # Parameters
    n = 5  # Number of nodes initially failed
    mu = 1/60  # Mean repair time of one node (in minutes)

    # Time points to evaluate
    time_points = np.linspace(0, 300, 100)  # From 0 to 300 minutes

    # Calculate state probabilities at each time point
    state_probabilities_over_time = np.array([survivability.compute_state_probabilities(n, mu, t) for t in time_points])

    # Plot the state probabilities over time
    plt.figure(figsize=(14, 8))
    for i in range(n + 1):
        plt.plot(time_points, state_probabilities_over_time[:, i], label=f'State {i}')

    plt.title('State Probabilities Over Time')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()






def plot_performance_curves(time_points, performance_curve_critical_centrality, target_percentage, adjusted_target_percentage, n, mean_repair_time, filename, u_performance_curve_critical_centrality=None, c_performance_curve_critical_centrality=None):
    sns.set_style("whitegrid")
    sns.set_context("poster")

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("dark")
    time_hours = time_points

    plt.plot(time_hours, performance_curve_critical_centrality, label="Critical Flow Centrality", color=colors[0], linestyle='-', linewidth=2)



    if u_performance_curve_critical_centrality is not None:
        plt.plot(time_hours, u_performance_curve_critical_centrality, color=colors[0], linestyle='--', linewidth=2, alpha=0.7)

    if c_performance_curve_critical_centrality is not None:
        plt.plot(time_hours, c_performance_curve_critical_centrality, color=colors[0], linestyle='-', linewidth=2, alpha=0.7)


    
    # Check if any non-critical flow data is provided and add a general label
    if any([u_performance_curve_critical_centrality]):
        plt.plot([], [], color=colors[0], linestyle='--', linewidth=2, alpha=0.7, label='Non-Critical Flows')



    plt.xlabel("Time (Hours)")
    plt.ylabel("Critical Flows Admitted (%)")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    #plt.title("Survivability Over Time")
    plt.legend()
    plt.tight_layout()

    file_path = 'new_figs/' + filename + '.pdf'

    plt.savefig(file_path)

    plt.show()

    print(f'{filename} --- {filename} --- {filename}')


def plot_performance_curves(time_points, performance_curve_critical_centrality, target_percentage, adjusted_target_percentage, n, mean_repair_time, filename, u_performance_curve_critical_centrality=None, c_performance_curve_critical_centrality=None):
    sns.set_style("whitegrid")
    sns.set_context("poster")

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("dark")
    time_hours = time_points

    plt.plot(time_hours, performance_curve_critical_centrality, label="Critical Flows", color=colors[2], linestyle='-', linewidth=2)


    if u_performance_curve_critical_centrality is not None:
        plt.plot(time_hours, u_performance_curve_critical_centrality, color=colors[2], linestyle='--', linewidth=2, alpha=0.7)

    if c_performance_curve_critical_centrality is not None:
        plt.plot(time_hours, c_performance_curve_critical_centrality, color=colors[2], linestyle='-', linewidth=2, alpha=0.7)

    
    # Check if any non-critical flow data is provided and add a general label
    if any([u_performance_curve_critical_centrality]):
        plt.plot([], [], color=colors[2], linestyle='--', linewidth=2, alpha=0.7, label='Non-Critical Flows')


    plt.xlabel("Time (Hours)")
    plt.ylabel("Flows Completed (%)")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend()
    plt.tight_layout()

    file_path = 'new_figs/' + filename + '.pdf'

    plt.savefig(file_path)

    plt.show()

    # print(f'{filename} --- {filename} --- {filename}')

    # sns.set_style("whitegrid")
    # sns.set_context("poster")

    # plt.figure(figsize=(12, 8))
    # colors = sns.color_palette("dark")
    # time_hours = time_points

    # plt.plot(time_hours, performance_curve_critical_centrality, label="Critical Flows", color=colors[3], linestyle='-', linewidth=2)


    # if u_performance_curve_critical_centrality is not None:
    #     plt.plot(time_hours, u_performance_curve_critical_centrality, color=colors[3], linestyle='--', linewidth=2, alpha=0.7)

    # if c_performance_curve_critical_centrality is not None:
    #     plt.plot(time_hours, c_performance_curve_critical_centrality, color=colors[3], linestyle='-', linewidth=2, alpha=0.7)

    
    # # Check if any non-critical flow data is provided and add a general label
    # if any([u_performance_curve_critical_centrality]):
    #     plt.plot([], [], color=colors[3], linestyle='--', linewidth=2, alpha=0.7, label='Non-Critical Flows')


    # plt.xlabel("Time (Hours)")
    # plt.ylabel("Flows Completed (%)")
    # plt.ylim(0, 100)
    # plt.xlim(left=0)
    # plt.legend()
    # plt.tight_layout()

    # file_path = 'new_figs/' + filename + '_red_' + '.pdf'

    # plt.savefig(file_path)

    # plt.show()

    # print(f'{filename} --- {filename} --- {filename}')




# For plotting non-nomrmalized data
def read_and_structure_data(file_path, strategies=['betweenness', 'greedy', 'critical_centrality', 'random'], time_points=[5, 25, 50, 75]):
    results = []
    
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    
    current_strategy = None
    for line in lines:
        if line in ["u_betweenness"]:
            break
        if line in strategies:
            current_strategy = line
            continue
        if current_strategy:
            results.append({
                'Strategy': current_strategy,
                'Performance': float(line),
            })
            
    df = pd.DataFrame(results)
    # Add a column for time point based on row number within each strategy
    df['Time Point'] = df.groupby('Strategy').cumcount() + 1
    # Filter rows by the specified time points
    df = df[df['Time Point'].isin(time_points)]
    
    return df

# For plotting normalized data
def read_and_structure_data_normalized(file_path, strategies=['betweenness', 'greedy', 'critical_centrality', 'random'], time_points=[5, 25, 50, 75, 95]):
    results = []
    
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    
    current_strategy = None
    strategy_data = {strategy: [] for strategy in strategies}
    
    for line in lines:
        if line in ["u_betweenness"]:
            break
        if line in strategies:
            current_strategy = line
            continue
        if current_strategy:
            strategy_data[current_strategy].append(float(line))
            
    for strategy, performances in strategy_data.items():
        min_performance = performances[0]
        max_performance = performances[-1]
        for time_point, performance in enumerate(performances):
            if time_point + 1 in time_points:  # Only include specified time points
                normalized_performance = (performance - min_performance) / (max_performance - min_performance) * 100
                results.append({
                    'Strategy': strategy,
                    'Performance': normalized_performance,
                    'Time Point': time_point + 1
                })
    
    df = pd.DataFrame(results)
    df = df[df['Time Point'].isin(time_points)]
    
    return df



# Combine data from multiple files, make ready for plotting (remember to change to read_and_structure_data or read_and_structure_data_normalized accordingly
def combine_and_calc_stats(file_paths):
    all_data = pd.DataFrame()

    for file_path in file_paths:
        df = read_and_structure_data_normalized(file_path)
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # Calculate mean and standard deviation for each strategy at each time point
    stats_data = all_data.groupby(['Strategy', 'Time Point']).agg(['mean', 'std']).reset_index()
    # Flatten the column names
    stats_data.columns = [' '.join(col).strip() for col in stats_data.columns.values]
    # Rename columns for clarity
    stats_data.rename(columns={'Performance mean': 'Mean', 'Performance std': 'Std'}, inplace=True)
    
    return stats_data






def read_and_structure_data_comb(file_path, identifier, strategies=['betweenness', 'greedy', 'critical_centrality', 'random'], time_points=[5, 25, 50, 75]):
    results = []
    
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    
    current_strategy = None
    for line in lines:
        if line in ["u_betweenness"]:
            break
        if line in strategies:
            current_strategy = line  # Keep strategy name clean
            continue
        if current_strategy:
            results.append({
                'File Identifier': identifier,  # Add file identifier separately
                'Strategy': current_strategy,
                'Performance': float(line),
            })
            
    df = pd.DataFrame(results)
    df['Time Point'] = df.groupby(['File Identifier', 'Strategy']).cumcount() + 1
    df = df[df['Time Point'].isin(time_points)]

    
    
    return df


def combine_data_from_two_files(file_path1, file_path2, identifiers):
    df1 = read_and_structure_data_comb(file_path1, identifiers[0], time_points=[5, 25, 50, 75])
    df2 = read_and_structure_data_comb(file_path2, identifiers[1], time_points=[5, 25, 50, 75])
    combined_df = pd.concat([df1, df2])
    return combined_df

# Plots combined data bars of two scenarios (one solid, one hatched). identifiers should be a list, e.g., ['pri', 'nopri']
def plot_combined_data(file_path1, file_path2, identifiers, normalized=False):
    df1 = read_and_structure_data_comb(file_path1, identifiers[0])
    df2 = read_and_structure_data_comb(file_path2, identifiers[1])
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Combine 'Strategy' and 'File Identifier' into a new 'Hue' column
    combined_df['Hue'] = combined_df['File Identifier'] + "-" + combined_df['Strategy']
    
    sns.set_style("whitegrid")
    sns.set_context("poster")
    plt.figure(figsize=(12, 8))
    
    colors = sns.color_palette("deep")
    strategy_colors = {
        'betweenness': colors[0],
        'greedy': colors[1],
        'critical_centrality': colors[3],
        'random': colors[2]
    }
    
    # Map only strategy part of 'Hue' to colors
    hue_colors = {hue: strategy_colors[hue.split('-')[1]] for hue in combined_df['Hue'].unique()}
    
    ax = sns.barplot(x='Time Point', y='Performance', hue='Hue', data=combined_df, palette=hue_colors, dodge=True)
    
    # Custom legend as before
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4, label=strategy) for strategy, color in strategy_colors.items()]
    plt.legend(handles=legend_patches, title='Strategy', loc='lower right', bbox_to_anchor=(1, 0), frameon=True)
    
    plt.ylim(0, 100)
    plt.ylabel('Critical Flows Admitted (%)')
    plt.xlabel('Time Point')
    
    # Apply hatching to bars of one file
    bars = ax.patches
    half = 16  # Assuming equal numbers of bars from each file
    for bar in bars[half:]:
        bar.set_hatch('////')  # Apply hatching to the second half of the bars
    
    plt.tight_layout()

    plt.savefig('new_figs/combined_data.pdf')


    plt.show()


def print_total_amount_of_nfs(G):
    total_NF_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for node in G.nodes(data=True):
        node_name = node[0]
        node_data = node[1]
        if str(node_name)[0] == '2' and len(str(node_name)) > 2:
            for nf in node_data['hosted_nfs']:
                total_NF_count[nf] += 1
    print(f'Total amount of instantiated network functions is: {total_NF_count}')
    return total_NF_count

def print_total_amount_of_nfs_physical(G):
    total_NF_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for node in G.nodes(data=True):
        node_name = node[0]
        node_data = node[1]
        try:      
            for nf in node_data['hosted_nfs']:
                    total_NF_count[nf] += 1
        except:
            #print('no hosted nfs')
            continue
    print(f'Total amount of instantiated network functions is: {total_NF_count}')
    return total_NF_count





def plot_amount_of_nfs(nfs1, nfs2, nfs3, nfs4, nfs5, filename):
    # set width of bars
    barWidth = 0.20
    
    # set heights of bars
    nfs1 = list(nfs1.values())
    nfs2 = list(nfs2.values())
    nfs3 = list(nfs3.values())
    nfs4 = list(nfs4.values())
    nfs5 = list(nfs5.values())
    
    # Set position of bar on X axis
    r1 = np.arange(len(nfs1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    
    # Make the plot
    plt.bar(r1, nfs1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Original ClusPR')
    plt.bar(r2, nfs2, color='#557f2d', width=barWidth, edgecolor='white', label='ClusPR with priority routing')
    plt.bar(r3, nfs3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Original ClusPR virtual nodes')
    plt.bar(r4, nfs4, color='#8fcf9f', width=barWidth, edgecolor='white', label='ClusPR with priority routing virtual nodes')
    plt.bar(r5, nfs5, color='#5e4c3c', width=barWidth, edgecolor='white', label='ClusPR per slice, virutalization isolation virtual nodes')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Type of network function')
    plt.ylabel('Number of instantiated network functions')
    plt.xticks([r + barWidth for r in range(len(nfs1))], ['NF 1', 'NF 2', 'NF 3', 'NF 4', 'NF 5'])
    
    # Create legend & Show graphic
    plt.legend()
    plt.savefig(f'new_figs/{filename}')
    plt.show()


def plot_amount_of_nfs_total(nfs1, nfs2, nfs3, nfs4, nfs5, filename):
    # set width of bars
    barWidth = 0.95

    # define colors
    colors = sns.color_palette("dark")

    sns.set_style("whitegrid")
    sns.set_context("poster")
    
    # set heights of bars
    nfs1 = sum(list(nfs1.values()))
    nfs2 = sum(list(nfs2.values()))
    nfs3 = sum(list(nfs3.values()))
    nfs4 = sum(list(nfs4.values()))
    nfs5 = sum(list(nfs5.values()))
        
    # Make the plot
    plt.bar(0, height=nfs1, color=colors[0], width=barWidth, edgecolor='white', label='Original ClusPR')
    #plt.bar(0.5, height=nfs2, color=colors[4], width=barWidth, edgecolor='white', label='ClusPR with priority routing')
    plt.bar(1, height=nfs3, color=colors[1], width=barWidth, edgecolor='white', label='Original ClusPR virtual nodes')
    #plt.bar(1.5, height=nfs4, color=colors[3], width=barWidth, edgecolor='white', label='ClusPR with priority routing virtual nodes')
    plt.bar(2, height=nfs5, color=colors[2], width=barWidth, edgecolor='white', label='ClusPR per slice, virutalization isolation virtual nodes')
    
    plt.ylabel('Number of instantiated network functions', fontsize=17)
    #plt.xticks([0, 0.5, 1, 1.5, 2], ['clusPR physical', 'clusPRI physical', 'clusPR virtual', 'clusPRI virtual', 'Virtual isolation'], rotation=45)
    plt.xticks([0, 1, 2], ['clusPR physical', 'clusPR virtual', 'Virtual isolation'], fontsize=12)#, rotation=45)

    plt.ylim(0,250)
    plt.yticks(list(range(0, 251, 10)), fontsize=10)
    plt.tight_layout()
    
    plt.savefig(f'new_figs/{filename}')
    plt.show()




def plot_initial_performance(cluspr_p, clusPRI_p, cluspr_v, clusPRI_v, clusperslice, filename):
    x = ['Before failure', 'After failure', 'After first re-routing']

    colors = sns.color_palette("dark")
    sns.set_style("whitegrid")
    sns.set_context("poster")

    # Plotting the lines
    plt.plot(x, cluspr_p, label='clusPR physical nodes', color=colors[0])
    plt.plot(x, clusPRI_p, label='clusPRI physical nodes', color=colors[1])
    plt.plot(x, cluspr_v, label='clusPR virtual nodes', color=colors[2])
    plt.plot(x, clusPRI_v, label='clusPRI virtual nodes', color=colors[3])
    plt.plot(x, clusperslice, label='clusPerSlise virtual nodes', color=colors[4])

    # Adding labels, title and legend
    plt.xlabel('Persentage of flows completed')
    plt.ylabel('State')
    plt.title('Initial performance of priority slice')

    plt.legend(fontsize=8)

    # Save and show plot
    plt.savefig(f'new_figs/{filename}')
    plt.show()

def get_utilization_list(G):
    utilization_list = []
    for node in G.nodes(data=True):
        node_data = node[1]
        try:
            hosted_nfs = node_data.get('hosted_nfs', 'Attribute not found')
            for nf in hosted_nfs:
                utilization_list.append(copy.deepcopy(node_data['nf_utilization'][nf]))
        except:
            print(f'Node doesnt host nfs')
    return utilization_list

def plot_utilization_cdf(util1, util2, util3, util4, util5, filename):

    colors = sns.color_palette("dark")

    def plot_cdf(data, label, color):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data))/float(len(sorted_data))
        plt.plot(sorted_data, yvals, label=label, color=color)

    # Plot CDF for each list
    plot_cdf(util1, 'Physical nodes clusPR', color=colors[0])
    #plot_cdf(util2, 'Physical nodes clusPRI')
    plot_cdf(util3, 'Virtual nodes clusPR', color=colors[1])
    #plot_cdf(util4, 'Virtual nodes clusPRI')
    plot_cdf(util5, 'Virtual isolation clusPR per slice', color=colors[2])

    # Plot settings
    plt.xlabel('Utilization (%)', fontsize=18)
    plt.ylabel('Percentage of Network Functions', fontsize=18)
    plt.title('CDF of Utilization of Network Functions', fontsize=20)
    plt.legend(loc='upper left', framealpha=0.5, fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'new_figs/{filename}')
    plt.show()



def read_file_and_put_in_list(file_path):
    strategy = ''
    performance_curve = []
    
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    
    for line in lines:
        if any(char.isalpha() for char in line):
            strategy = line
        else:
            if any(char.isdigit() for char in line):
                performance_curve.append(float(line))

    return performance_curve

def plot_performance_curves_combined(time_points1, performance_curve_physical, time_points2, performance_curve_virtual, time_points3, performance_curve_virtual_isolation, filename, u_performance_curve_physical=None, u_performance_curve_virtual=None, u_performance_curve_virtual_isolation=None):
    sns.set_style("whitegrid")
    sns.set_context("poster")

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("dark")
    time_hours1 = list(time_points1)
    time_hours2 = list(time_points2)
    time_hours3 = list(time_points3)

    performance_curve1 = read_file_and_put_in_list(performance_curve_physical)
    performance_curve2 = read_file_and_put_in_list(performance_curve_virtual)
    performance_curve3 = read_file_and_put_in_list(performance_curve_virtual_isolation)

    max_time_value = max(time_hours1[-1], time_hours2[-1], time_hours3[-1])

    time_hours1.append(max_time_value)
    time_hours2.append(max_time_value)
    time_hours3.append(max_time_value)

    performance_curve1.append(performance_curve1[-1])
    performance_curve2.append(performance_curve2[-1])
    performance_curve3.append(performance_curve3[-1])

    plt.plot(time_hours1, performance_curve1, label="Physical nodes", color=colors[0], linestyle='-', linewidth=2)
    plt.plot(time_hours2, performance_curve2, label="Virtual nodes", color=colors[3], linestyle='-', linewidth=2)
    plt.plot(time_hours3, performance_curve3, label="Virtual isolation", color=colors[1], linestyle='-', linewidth=2)


    # Optionally plot for non-critical traffic
    if u_performance_curve_physical is not None:
        plt.plot(time_hours1, u_performance_curve_physical, color=colors[0], linestyle='--', linewidth=2, alpha=0.7)
    if u_performance_curve_virtual is not None:
        plt.plot(time_hours2, u_performance_curve_virtual, color=colors[3], linestyle='--', linewidth=2, alpha=0.7)
    if u_performance_curve_virtual_isolation is not None:
        plt.plot(time_hours3, u_performance_curve_virtual_isolation, color=colors[1], linestyle='--', linewidth=2, alpha=0.7)


    
    # Check if any non-critical flow data is provided and add a general label
    if any([u_performance_curve_physical]):
        plt.plot([], [], color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Non-Critical Flows')

  

    plt.xlabel("Time (Hours)")
    plt.ylabel("Critical Flows Completed (%)")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    #plt.title("Survivability Over Time")
    plt.legend()
    plt.tight_layout()

    file_path = 'new_figs/' + filename + '.pdf'

    plt.savefig(file_path)

    plt.show()

    print(f'{filename} --- {filename} --- {filename}')

   

# Help function for finding the average of performance curves
def average_of_curves(curve_dict, n):
    # average_curve = []
    # length = len(curve_dict[0])

    values = np.array(list(curve_dict.values()))
    
    means = np.mean(values, axis=0)
    std_err = np.std(values, axis=0) / np.sqrt(n)
    curve = means.tolist()
    stderr = std_err.tolist()
    
    return curve, stderr

    # for i in range(length):
    #     for x in range[n]:
   
    # return average



def plot_average_performance_curves_combined(time_points1, performance_curve_physical, time_points2, performance_curve_virtual, time_points3, performance_curve_virtual_isolation, filename, n, u_performance_curve_physical=None, u_performance_curve_virtual=None, u_performance_curve_virtual_isolation=None):
    sns.set_style("whitegrid")
    sns.set_context("poster")

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("dark")
    time_hours1 = list(time_points1)
    time_hours2 = list(time_points2)
    time_hours3 = list(time_points3)

    performance_curve1, std_err1 = average_of_curves(performance_curve_physical, n)
    performance_curve2, std_err2 = average_of_curves(performance_curve_virtual, n)
    performance_curve3, std_err3 = average_of_curves(performance_curve_virtual_isolation, n)

    max_time_value = max(time_hours1[-1], time_hours2[-1], time_hours3[-1])

    time_hours1.append(max_time_value)
    time_hours2.append(max_time_value)
    time_hours3.append(max_time_value)

    performance_curve1.append(performance_curve1[-1])
    performance_curve2.append(performance_curve2[-1])
    performance_curve3.append(performance_curve3[-1])

    plt.plot(time_hours1, performance_curve1, label="Physical baseline", color=colors[0], linestyle='-', linewidth=2)
    plt.plot(time_hours2, performance_curve2, label="Virtual separation of NFs", color=colors[1], linestyle='-', linewidth=2)
    plt.plot(time_hours3, performance_curve3, label="Virtual isolation of slices", color=colors[2], linestyle='-', linewidth=2)

    # Plot bars for standard deviation of the average for performance curve 1:
    x_error_points = np.array([time_hours1[0], time_hours1[9], time_hours1[19], time_hours1[29], time_hours1[39], time_hours1[49], time_hours1[59], time_hours1[69], time_hours1[79], time_hours1[89], time_hours1[99]])
    y_error_points = np.array([performance_curve1[0], performance_curve1[9], performance_curve1[19], performance_curve1[29], performance_curve1[39], performance_curve1[49], performance_curve1[59], performance_curve1[69], performance_curve1[79], performance_curve1[89], performance_curve1[99]])
    errors = np.array([std_err1[0], std_err1[9], std_err1[19], std_err1[29], std_err1[39], std_err1[49], std_err1[59], std_err1[69], std_err1[79], std_err1[89], std_err1[99]])
    plt.errorbar(x_error_points, y_error_points, yerr=errors, fmt='.', capsize = 3, color = colors[0], elinewidth=1, ecolor=colors[0]) #, label='Standard Deviation Error bars')

    # Plot bars for standard deviation of the average for performance curve 2:
    x_error_points = np.array([time_hours2[0], time_hours2[9], time_hours2[19], time_hours2[29], time_hours2[39], time_hours2[49], time_hours2[59], time_hours2[69], time_hours2[79], time_hours2[89], time_hours2[99]])
    y_error_points = np.array([performance_curve2[0], performance_curve2[9], performance_curve2[19], performance_curve2[29], performance_curve2[39], performance_curve2[49], performance_curve2[59], performance_curve2[69], performance_curve2[79], performance_curve2[89], performance_curve2[99]])
    errors = np.array([std_err2[0], std_err2[9], std_err2[19], std_err2[29], std_err2[39], std_err2[49], std_err2[59], std_err2[69], std_err2[79], std_err2[89], std_err2[99]])
    plt.errorbar(x_error_points, y_error_points, yerr=errors, fmt='.', capsize = 3, color = colors[1], elinewidth=1, ecolor=colors[1])

    # Plot bars for standard deviation of the average for performance curve 2:
    x_error_points = np.array([time_hours3[0], time_hours3[9], time_hours3[19], time_hours3[29], time_hours3[39], time_hours3[49], time_hours3[59], time_hours3[69], time_hours3[79], time_hours3[89], time_hours3[99]])
    y_error_points = np.array([performance_curve3[0], performance_curve3[9], performance_curve3[19], performance_curve3[29], performance_curve3[39], performance_curve3[49], performance_curve3[59], performance_curve3[69], performance_curve3[79], performance_curve3[89], performance_curve3[99]])
    errors = np.array([std_err1[0], std_err3[9], std_err3[19], std_err3[29], std_err3[39], std_err3[49], std_err3[59], std_err3[69], std_err3[79], std_err3[89], std_err3[99]])
    plt.errorbar(x_error_points, y_error_points, yerr=errors, fmt='.', capsize = 3, color = colors[2], elinewidth=1, ecolor=colors[2])



    # Optionally plot for non-critical traffic
    if u_performance_curve_physical is not None:
        plt.plot(time_hours1, u_performance_curve_physical, color=colors[0], linestyle='--', linewidth=2, alpha=0.7)
    if u_performance_curve_virtual is not None:
        plt.plot(time_hours2, u_performance_curve_virtual, color=colors[3], linestyle='--', linewidth=2, alpha=0.7)
    if u_performance_curve_virtual_isolation is not None:
        plt.plot(time_hours3, u_performance_curve_virtual_isolation, color=colors[1], linestyle='--', linewidth=2, alpha=0.7)

    
    # Check if any non-critical flow data is provided and add a general label
    if any([u_performance_curve_physical]):
        plt.plot([], [], color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Non-Critical Flows')
        

    def avg_95(performance_curve, time_hours):
        time95 = []
        for curve in performance_curve.values():
            maks = curve[-1]
            perc_95 = maks*0.95
            for index, value in enumerate(curve):
                if value >= perc_95:
                    time95.append(time_hours[index])
                    break
            
        avg95 = np.mean(time95)
        stderr = np.std(time95) / np.sqrt(len(time95))
        
        return avg95, stderr
        
    # Time to 95% of normal performance
    def time_95(curve, time_hours):
        original_perf = curve[-1]
        perf_95 = 0.95 * original_perf
        indeks = 0
        for index, value in enumerate(curve):
            if value >= perf_95:
                indeks = index
                return time_hours[indeks] 
        return 0      
    
    print(f'Time to 95% of normal performane physical: {time_95(performance_curve1, time_hours1)}')
    print(f'Time to 95% of normal performane virtual: {time_95(performance_curve2, time_hours2)}')
    print(f'Time to 95% of normal performane virtual isolation: {time_95(performance_curve3, time_hours3)}')

    print(f'Time to 95% of normal performane physical test: {avg_95(performance_curve_physical, time_hours1)}')
    print(f'Time to 95% of normal performane virtual test: {avg_95(performance_curve_virtual, time_hours2)}')
    print(f'Time to 95% of normal performane virtual isolation tes: {avg_95(performance_curve_virtual_isolation, time_hours3)}')

    print(f'Normal performance physical: {performance_curve1[-1]} and stderr: {std_err1[-1]}')
    print(f'Normal performance virtual: {performance_curve2[-1]} and stderr: {std_err2[-1]}')
    print(f'Normal performance virtual isolation: {performance_curve3[-1]} and stderr: {std_err3[-1]}')


    plt.xlabel("Time (Hours)")
    plt.ylabel("Critical Flows Completed (%)")
    plt.ylim(0, 100)
    plt.yticks(list(range(0, 101, 10)))
    plt.xlim(left=0)
    #plt.title("Survivability Over Time")
    plt.legend()
    plt.tight_layout()

    file_path = 'new_figs_average/' + filename + '.pdf'

    plt.savefig(file_path)

    plt.show()

    print(f'{filename} --- {filename} --- {filename}')


def calculate_average_cumulative(dict):
    data_list = [np.array(values) for values in dict.values()]
    data_list = data_list
    n = len(data_list)  
    m = len(data_list[0])

    # Calculate the points at which we want to evaluate the distributions (5% steps)
    percent_steps = np.arange(0, 101, 5)  

    # Calculate cumulative distributions without interpolation
    cumulative_distributions = np.zeros((n, len(percent_steps)))

    for i in range(n):
        for j, step in enumerate(percent_steps):
            cumulative_distributions[i, j] = np.sum(data_list[i] <= step) / m

    # Calculate the average cumulative distribution
    mean_cum_dist = np.mean(cumulative_distributions, axis=0) *100

    # Calculate the standard error
    stderr_cum_dist = np.std(cumulative_distributions, axis=0) / np.sqrt(n) *100

    return mean_cum_dist, stderr_cum_dist, percent_steps


def plot_average_utilization_cdf(util1, util2, util3, filename):
    sns.set_style("whitegrid")
    sns.set_context("poster")
    colors = sns.color_palette("dark")

    def find_average_util(dict):
        values = [value for list in dict.values() for value in list]
        average_util = sum(values) / len(values)

        stderr = np.std(values) / np.sqrt(len(values))


        return average_util, stderr

    def plot_cdf(data, label, color):
        mean_cum_dist, stderr, x = calculate_average_cumulative(data)
        
        plt.plot(x, mean_cum_dist, label=label, color=color)
        plt.errorbar(x, mean_cum_dist, yerr=stderr, fmt='.', markersize=4, capsize = 5, color = 'black', elinewidth=1, ecolor=color)

    # Plot CDF for each list
    plot_cdf(util1, 'Physical baseline', color=colors[0])
    plot_cdf(util2, 'Virtual separation of NFs', color=colors[1])
    plot_cdf(util3, 'Virtual isolation of slices', color=colors[2])



    print(f'Physical util: {find_average_util(util1)}')
    print(f'Virtual util: {find_average_util(util2)}')
    print(f'Virtual isolation util: {find_average_util(util3)}')

    # Plot settings
    plt.xlabel('Utilization (%)', fontsize=18)
    plt.ylabel('Percentage of Network Functions', fontsize=18)
    plt.ylim(0,100)
    plt.xlim(0,100)
    plt.xticks(list(range(0, 101, 5)), fontsize=10)
    plt.yticks(list(range(0, 101, 5)), fontsize=10)
    plt.title('CDF of Utilization of Network Functions', fontsize=20)
    plt.legend(loc='upper left', framealpha=0.5, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'new_figs_average/{filename}.pdf')
    plt.show()

def plot_average_amount_of_nfs(phys_nfs, virt_nfs, virt_isol_nfs, filename):
    # set width of bars
    barWidth = 0.95

    n = len(phys_nfs)

    # define colors
    colors = sns.color_palette("dark")

    sns.set_style("whitegrid")
    sns.set_context("poster")
    
    # set heights of bars
    nfs1, nfs1_std = average_of_curves(phys_nfs, n)
    nfs2, nfs2_std = average_of_curves(virt_nfs, n)
    nfs3, nfs3_std = average_of_curves(virt_isol_nfs, n)

    print(f'Physical: {nfs1}')
    print(f'Virtual: {nfs2}')
    print(f'Virtual isolation: {nfs3}')
    
    plt.errorbar(0, nfs1, yerr=nfs1_std, fmt='o', markersize=5, capsize = 7, color = colors[0], elinewidth=1, ecolor=colors[0])
    plt.errorbar(1, nfs2, yerr=nfs2_std, fmt='o', markersize=5, capsize = 7, color = colors[1], elinewidth=1, ecolor=colors[1])
    plt.errorbar(2, nfs3, yerr=nfs3_std, fmt='o', markersize=5, capsize = 7, color = colors[2], elinewidth=1, ecolor=colors[2])


    plt.ylabel('Number of instantiated network functions', fontsize=17)
    plt.xticks([0, 1, 2], ['Physical baseline', 'Virtual separation of NFs', 'Virtual isolation of slices'], fontsize=12)#, rotation=45)
    plt.axhline(y=nfs1, color='grey', linestyle='--', alpha=0.7, linewidth=1.2)

    if nfs3 > 110:
        plt.ylim(110,240)
        plt.yticks(list(range(110, 241, 5)), fontsize=10)
    else: 
        plt.ylim(80,100)
        plt.yticks(list(range(80, 101, 2)), fontsize=10)
    
    plt.tight_layout()
    plt.grid(True, alpha= 0.3)
   
    plt.savefig(f'new_figs_average/{filename}.pdf')
    plt.show()

    

def plot_average_initial_performance(init_perf_orig, inti_perf_virt, int_perf_virt_isol, n, filename):
    barWidth = 0.95
    colors = sns.color_palette("dark")
    sns.set_style("whitegrid")
    sns.set_context("poster")

    # set heights of bars
    init_perf_orig, init_perf_orig_stderr = average_of_curves(init_perf_orig, n)
    inti_perf_virt, inti_perf_virt_stderr = average_of_curves(inti_perf_virt, n)
    int_perf_virt_isol, int_perf_virt_isol_stderr = average_of_curves(int_perf_virt_isol, n)

    print(f'Physical: {init_perf_orig}, stderr: {init_perf_orig_stderr}')
    print(f'Virtual: {inti_perf_virt}, stderr: {inti_perf_virt_stderr}')
    print(f'Virtual isolation: {int_perf_virt_isol}, stderr: {int_perf_virt_isol_stderr}')
    

    plt.errorbar([0, 4, 8], init_perf_orig, yerr=init_perf_orig_stderr, fmt='o', markersize=5, capsize = 7, color = colors[0], elinewidth=1, ecolor=colors[0], label='Physical baseline')
    plt.errorbar([1, 5, 9], inti_perf_virt, yerr=inti_perf_virt_stderr, fmt='o', markersize=5, capsize = 7, color = colors[1], elinewidth=1, ecolor=colors[1], label='Virtual separation of NFs')
    plt.errorbar([2, 6, 10], int_perf_virt_isol, yerr=int_perf_virt_isol_stderr, fmt='o', markersize=5, capsize = 7, color = colors[2], elinewidth=1, ecolor=colors[2], label='Virtual isolation of slices')

    plt.xticks([1, 5, 9], ['Before failure', 'After failure', 'After first re-routing'], fontsize=12)#, rotation=45)

    plt.axvspan(3, 7, color='grey', alpha=0.3)
    
    plt.ylabel('Critical flows completed (%)', fontsize=12)

    plt.ylim(0,100)
    plt.yticks(list(range(0, 101, 10)), fontsize=12)

    plt.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
    plt.tight_layout(pad=1)

    plt.savefig(f'new_figs_average/{filename}.pdf')
    plt.show()



def plot_performance(perf_critical_phys, perf_non_critical_phys, perf_critical_virt, perf_non_critical_virt, perf_critical_virt_isol, perf_non_critical_virt_isol):
    barWidth = 0.95
    colors = sns.color_palette("dark")
    colors_bright = sns.color_palette("pastel")
    sns.set_style("whitegrid")
    sns.set_context("poster")

    plt.bar([0, 7, 14, 21, 28], height=perf_critical_phys, color=colors[0], width=barWidth, edgecolor='white', label='Physical nodes critical/non-critical')
    plt.bar([1, 8, 15, 22, 29], height=perf_non_critical_phys, color=colors[0], alpha=0.5, hatch='/', width=barWidth, edgecolor='white')
    plt.bar([2, 9, 16, 23, 30], height=perf_critical_virt, color=colors[1], width=barWidth, edgecolor='white', label='Virtual nodes critical/non-critical')
    plt.bar([3, 10, 17, 24, 31], height=perf_non_critical_virt, color=colors[1], alpha=0.5, hatch='/', width=barWidth, edgecolor='white')
    plt.bar([4, 11, 18, 25, 32], height=perf_critical_virt_isol, color=colors[2], width=barWidth, edgecolor='white', label='Virtual isolation critical/non-critical')
    plt.bar([5, 12, 19, 26, 33], height=perf_non_critical_virt_isol, color=colors[2], alpha=0.5, hatch='/', width=barWidth, edgecolor='white')
    plt.ylim(0,100)
    plt.xticks([2.5, 9.5, 16.5, 23.5, 30.5], ['400 flows', '600 flows', '800 flows', '1000 flows', '1200 flows'], fontsize=12)
    plt.ylabel('Persentage of flows completed', fontsize=14)
    plt.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
    plt.tight_layout(pad=1)

    plt.savefig(f'new_figs/performance_for_more_flows.pdf')
    plt.show()
    


def plot_amount_of_nfs_total_2(nfs1, nfs2, nfs3, filename):
    # set width of bars
    barWidth = 0.95

    # define colors
    colors = sns.color_palette("dark")

    sns.set_style("whitegrid")
    sns.set_context("poster")
    
    # set heights of bars
    nfs1_list = []
    nfs2_list = []
    nfs3_list = []
    
    for dict in nfs1:
        nfs = sum(list(dict.values()))
        nfs1_list.append(nfs)

    for dict in nfs2:
        nfs = sum(list(dict.values()))
        nfs2_list.append(nfs)
    
    for dict in nfs3:
        nfs = sum(list(dict.values()))
        nfs3_list.append(nfs)
    
        
    plt.bar([0, 4, 8, 12, 16], height=nfs1_list, color=colors[0], width=barWidth, edgecolor='white', label='Physical nodes')
    plt.bar([1, 5, 9, 13, 17], height=nfs2_list, color=colors[1], width=barWidth, edgecolor='white', label='Virtual nodes')
    plt.bar([2, 6, 10, 14, 18], height=nfs3_list, color=colors[2], width=barWidth, edgecolor='white', label='Virtual isolation')
 
    # Add xticks on the middle of the group bars
    plt.ylabel('Number of instantiated network functions', fontsize=17)
    plt.xticks([1, 5, 9, 13, 17], ['400', '600', '800', '1000', '1200'], fontsize=12)

    plt.ylim(0,150)
    plt.yticks(list(range(0, 151, 10)), fontsize=10)
    plt.tight_layout()
    # Create legend & Show graphic
    #plt.legend()
    plt.savefig(f'new_figs/{filename}')
    plt.show()