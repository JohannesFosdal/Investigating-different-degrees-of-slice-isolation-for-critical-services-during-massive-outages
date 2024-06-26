import plotsSurvivability
import numpy as np

def read_files_and_put_in_dict(file_list):
    strategy = ''
    dict = {}
    for i in range(len(file_list)):
        list = []
        with open(file_list[i], 'r') as file:
            lines = file.read().splitlines()
        
        for line in lines:
            if any(char.isalpha() for char in line):
                strategy = line
            else:
                if any(char.isdigit() for char in line):
                    list.append(float(line))
        dict[i] = list

    return dict


def plot_from_dicts(file_list1, file_list2, file_list3, n):
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    mean_repair_time = 0.1
    time_points = np.linspace(0, 65 * mean_repair_time * 2.5, 100)
    
    plotsSurvivability.plot_average_performance_curves_combined(time_points, dict1, time_points, dict2, time_points, dict3, 'test_new_average_cumulative_util', n)


def plot_avg_performance():
    # uninett attack
    n = 10
    file_list1 = ['new_data_average/10_runs_uninett_physical_attack0', 'new_data_average/10_runs_uninett_physical_attack1', 'new_data_average/10_runs_uninett_physical_attack2', 'new_data_average/10_runs_uninett_physical_attack3', 'new_data_average/10_runs_uninett_physical_attack4', 'new_data_average/10_runs_uninett_physical_attack5', 'new_data_average/10_runs_uninett_physical_attack6', 'new_data_average/10_runs_uninett_physical_attack7', 'new_data_average/10_runs_uninett_physical_attack8', 'new_data_average/10_runs_uninett_physical_attack9']
    file_list2 = ['new_data_average/10_runs_uninett_virtual_attack0', 'new_data_average/10_runs_uninett_virtual_attack1', 'new_data_average/10_runs_uninett_virtual_attack2', 'new_data_average/10_runs_uninett_virtual_attack3', 'new_data_average/10_runs_uninett_virtual_attack4', 'new_data_average/10_runs_uninett_virtual_attack5', 'new_data_average/10_runs_uninett_virtual_attack6', 'new_data_average/10_runs_uninett_virtual_attack7', 'new_data_average/10_runs_uninett_virtual_attack8', 'new_data_average/10_runs_uninett_virtual_attack9']
    file_list3 = ['new_data_average/10_runs_uninett_virtual_isolation_attack0', 'new_data_average/10_runs_uninett_virtual_isolation_attack1', 'new_data_average/10_runs_uninett_virtual_isolation_attack2', 'new_data_average/10_runs_uninett_virtual_isolation_attack3', 'new_data_average/10_runs_uninett_virtual_isolation_attack4', 'new_data_average/10_runs_uninett_virtual_isolation_attack5', 'new_data_average/10_runs_uninett_virtual_isolation_attack6', 'new_data_average/10_runs_uninett_virtual_isolation_attack7', 'new_data_average/10_runs_uninett_virtual_isolation_attack8', 'new_data_average/10_runs_uninett_virtual_isolation_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    mean_repair_time = 0.1
    time_points = np.linspace(0, 65 * mean_repair_time * 2.5, 100)
    
    plotsSurvivability.plot_average_performance_curves_combined(time_points, dict1, time_points, dict2, time_points, dict3, 'new_fig_uninett_attack_performance', n)

    #geant attack
    n = 10
    file_list1 = ['new_data_average/10_runs_geant_physical_attack0', 'new_data_average/10_runs_geant_physical_attack1', 'new_data_average/10_runs_geant_physical_attack2', 'new_data_average/10_runs_geant_physical_attack3', 'new_data_average/10_runs_geant_physical_attack4', 'new_data_average/10_runs_geant_physical_attack5', 'new_data_average/10_runs_geant_physical_attack6', 'new_data_average/10_runs_geant_physical_attack7', 'new_data_average/10_runs_geant_physical_attack8', 'new_data_average/10_runs_geant_physical_attack9']
    file_list2 = ['new_data_average/10_runs_geant_virtual_attack0', 'new_data_average/10_runs_geant_virtual_attack1', 'new_data_average/10_runs_geant_virtual_attack2', 'new_data_average/10_runs_geant_virtual_attack3', 'new_data_average/10_runs_geant_virtual_attack4', 'new_data_average/10_runs_geant_virtual_attack5', 'new_data_average/10_runs_geant_virtual_attack6', 'new_data_average/10_runs_geant_virtual_attack7', 'new_data_average/10_runs_geant_virtual_attack8', 'new_data_average/10_runs_geant_virtual_attack9']
    file_list3 = ['new_data_average/10_runs_geant_virtual_isolation_attack0', 'new_data_average/10_runs_geant_virtual_isolation_attack1', 'new_data_average/10_runs_geant_virtual_isolation_attack2', 'new_data_average/10_runs_geant_virtual_isolation_attack3', 'new_data_average/10_runs_geant_virtual_isolation_attack4', 'new_data_average/10_runs_geant_virtual_isolation_attack5', 'new_data_average/10_runs_geant_virtual_isolation_attack6', 'new_data_average/10_runs_geant_virtual_isolation_attack7', 'new_data_average/10_runs_geant_virtual_isolation_attack8', 'new_data_average/10_runs_geant_virtual_isolation_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    mean_repair_time = 0.1
    time_points = np.linspace(0, 65 * mean_repair_time * 2.5, 100)
    
    plotsSurvivability.plot_average_performance_curves_combined(time_points, dict1, time_points, dict2, time_points, dict3, 'new_fig_geant_attack_performance', n)


    # uninett geographical
    n = 10
    file_list1 = ['new_data_average/10_runs_uninett_physicalgeographical0', 'new_data_average/10_runs_uninett_physicalgeographical1', 'new_data_average/10_runs_uninett_physicalgeographical2', 'new_data_average/10_runs_uninett_physicalgeographical3', 'new_data_average/10_runs_uninett_physicalgeographical4', 'new_data_average/10_runs_uninett_physicalgeographical5', 'new_data_average/10_runs_uninett_physicalgeographical6', 'new_data_average/10_runs_uninett_physicalgeographical7', 'new_data_average/10_runs_uninett_physicalgeographical8', 'new_data_average/10_runs_uninett_physicalgeographical9']
    file_list2 = ['new_data_average/10_runs_uninett_virtualgeographical0', 'new_data_average/10_runs_uninett_virtualgeographical1', 'new_data_average/10_runs_uninett_virtualgeographical2', 'new_data_average/10_runs_uninett_virtualgeographical3', 'new_data_average/10_runs_uninett_virtualgeographical4', 'new_data_average/10_runs_uninett_virtualgeographical5', 'new_data_average/10_runs_uninett_virtualgeographical6', 'new_data_average/10_runs_uninett_virtualgeographical7', 'new_data_average/10_runs_uninett_virtualgeographical8', 'new_data_average/10_runs_uninett_virtualgeographical9']
    file_list3 = ['new_data_average/10_runs_uninett_virtual_isolationgeographical0', 'new_data_average/10_runs_uninett_virtual_isolationgeographical1', 'new_data_average/10_runs_uninett_virtual_isolationgeographical2', 'new_data_average/10_runs_uninett_virtual_isolationgeographical3', 'new_data_average/10_runs_uninett_virtual_isolationgeographical4', 'new_data_average/10_runs_uninett_virtual_isolationgeographical5', 'new_data_average/10_runs_uninett_virtual_isolationgeographical6', 'new_data_average/10_runs_uninett_virtual_isolationgeographical7', 'new_data_average/10_runs_uninett_virtual_isolationgeographical8', 'new_data_average/10_runs_uninett_virtual_isolationgeographical9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    number_of_failed_nodes = 11
    mean_repair_time = 5
    time_points = np.linspace(0, number_of_failed_nodes * mean_repair_time * 1.5, 100)
    
    plotsSurvivability.plot_average_performance_curves_combined(time_points, dict1, time_points, dict2, time_points, dict3, 'new_fig_uninett_geo_performance', n)


    # geant geographical
    n = 10
    file_list1 = ['new_data_average/10_runs_geant_physicalgeographical0', 'new_data_average/10_runs_geant_physicalgeographical1', 'new_data_average/10_runs_geant_physicalgeographical2', 'new_data_average/10_runs_geant_physicalgeographical3', 'new_data_average/10_runs_geant_physicalgeographical4', 'new_data_average/10_runs_geant_physicalgeographical5', 'new_data_average/10_runs_geant_physicalgeographical6', 'new_data_average/10_runs_geant_physicalgeographical7', 'new_data_average/10_runs_geant_physicalgeographical8', 'new_data_average/10_runs_geant_physicalgeographical9']
    file_list2 = ['new_data_average/10_runs_geant_virtualgeographical0', 'new_data_average/10_runs_geant_virtualgeographical1', 'new_data_average/10_runs_geant_virtualgeographical2', 'new_data_average/10_runs_geant_virtualgeographical3', 'new_data_average/10_runs_geant_virtualgeographical4', 'new_data_average/10_runs_geant_virtualgeographical5', 'new_data_average/10_runs_geant_virtualgeographical6', 'new_data_average/10_runs_geant_virtualgeographical7', 'new_data_average/10_runs_geant_virtualgeographical8', 'new_data_average/10_runs_geant_virtualgeographical9']
    file_list3 = ['new_data_average/10_runs_geant_virtual_isolationgeographical0', 'new_data_average/10_runs_geant_virtual_isolationgeographical1', 'new_data_average/10_runs_geant_virtual_isolationgeographical2', 'new_data_average/10_runs_geant_virtual_isolationgeographical3', 'new_data_average/10_runs_geant_virtual_isolationgeographical4', 'new_data_average/10_runs_geant_virtual_isolationgeographical5', 'new_data_average/10_runs_geant_virtual_isolationgeographical6', 'new_data_average/10_runs_geant_virtual_isolationgeographical7', 'new_data_average/10_runs_geant_virtual_isolationgeographical8', 'new_data_average/10_runs_geant_virtual_isolationgeographical9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    number_of_failed_nodes = 6
    mean_repair_time = 5
    time_points = np.linspace(0, number_of_failed_nodes * mean_repair_time * 1.5, 100)
    
    plotsSurvivability.plot_average_performance_curves_combined(time_points, dict1, time_points, dict2, time_points, dict3, 'new_fig_geant_geo_performance', n)



def avg_cumul_util():
    # uninett attack
    file_list1 = ['new_data_average/10_runs_uninett_physical_util_attack0', 'new_data_average/10_runs_uninett_physical_util_attack1', 'new_data_average/10_runs_uninett_physical_util_attack2', 'new_data_average/10_runs_uninett_physical_util_attack3', 'new_data_average/10_runs_uninett_physical_util_attack4', 'new_data_average/10_runs_uninett_physical_util_attack5', 'new_data_average/10_runs_uninett_physical_util_attack6', 'new_data_average/10_runs_uninett_physical_util_attack7', 'new_data_average/10_runs_uninett_physical_util_attack8', 'new_data_average/10_runs_uninett_physical_util_attack9']
    file_list2 = ['new_data_average/10_runs_uninett_virtual_util_attack0', 'new_data_average/10_runs_uninett_virtual_util_attack1', 'new_data_average/10_runs_uninett_virtual_util_attack2', 'new_data_average/10_runs_uninett_virtual_util_attack3', 'new_data_average/10_runs_uninett_virtual_util_attack4', 'new_data_average/10_runs_uninett_virtual_util_attack5', 'new_data_average/10_runs_uninett_virtual_util_attack6', 'new_data_average/10_runs_uninett_virtual_util_attack7', 'new_data_average/10_runs_uninett_virtual_util_attack8', 'new_data_average/10_runs_uninett_virtual_util_attack9']
    file_list3 = ['new_data_average/10_runs_uninett_virtual_isol_util_attack0', 'new_data_average/10_runs_uninett_virtual_isol_util_attack1', 'new_data_average/10_runs_uninett_virtual_isol_util_attack2', 'new_data_average/10_runs_uninett_virtual_isol_util_attack3', 'new_data_average/10_runs_uninett_virtual_isol_util_attack4', 'new_data_average/10_runs_uninett_virtual_isol_util_attack5', 'new_data_average/10_runs_uninett_virtual_isol_util_attack6', 'new_data_average/10_runs_uninett_virtual_isol_util_attack7', 'new_data_average/10_runs_uninett_virtual_isol_util_attack8', 'new_data_average/10_runs_uninett_virtual_isol_util_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)
    
    plotsSurvivability.plot_average_utilization_cdf(dict1, dict2, dict3, 'new_fig_uninett_attack_util')


    # geant attack
    file_list1 = ['new_data_average/10_runs_geant_physical_util_attack0', 'new_data_average/10_runs_geant_physical_util_attack1', 'new_data_average/10_runs_geant_physical_util_attack2', 'new_data_average/10_runs_geant_physical_util_attack3', 'new_data_average/10_runs_geant_physical_util_attack4', 'new_data_average/10_runs_geant_physical_util_attack5', 'new_data_average/10_runs_geant_physical_util_attack6', 'new_data_average/10_runs_geant_physical_util_attack7', 'new_data_average/10_runs_geant_physical_util_attack8', 'new_data_average/10_runs_geant_physical_util_attack9']
    file_list2 = ['new_data_average/10_runs_geant_virtual_util_attack0', 'new_data_average/10_runs_geant_virtual_util_attack1', 'new_data_average/10_runs_geant_virtual_util_attack2', 'new_data_average/10_runs_geant_virtual_util_attack3', 'new_data_average/10_runs_geant_virtual_util_attack4', 'new_data_average/10_runs_geant_virtual_util_attack5', 'new_data_average/10_runs_geant_virtual_util_attack6', 'new_data_average/10_runs_geant_virtual_util_attack7', 'new_data_average/10_runs_geant_virtual_util_attack8', 'new_data_average/10_runs_geant_virtual_util_attack9']
    file_list3 = ['new_data_average/10_runs_geant_virtual_isol_util_attack0', 'new_data_average/10_runs_geant_virtual_isol_util_attack1', 'new_data_average/10_runs_geant_virtual_isol_util_attack2', 'new_data_average/10_runs_geant_virtual_isol_util_attack3', 'new_data_average/10_runs_geant_virtual_isol_util_attack4', 'new_data_average/10_runs_geant_virtual_isol_util_attack5', 'new_data_average/10_runs_geant_virtual_isol_util_attack6', 'new_data_average/10_runs_geant_virtual_isol_util_attack7', 'new_data_average/10_runs_geant_virtual_isol_util_attack8', 'new_data_average/10_runs_geant_virtual_isol_util_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)
    
    plotsSurvivability.plot_average_utilization_cdf(dict1, dict2, dict3, 'new_fig_geant_attack_util')


    # uninett geo
    file_list1 = ['new_data_average/10_runs_uninett_physical_util_geographical0', 'new_data_average/10_runs_uninett_physical_util_geographical1', 'new_data_average/10_runs_uninett_physical_util_geographical2', 'new_data_average/10_runs_uninett_physical_util_geographical3', 'new_data_average/10_runs_uninett_physical_util_geographical4', 'new_data_average/10_runs_uninett_physical_util_geographical5', 'new_data_average/10_runs_uninett_physical_util_geographical6', 'new_data_average/10_runs_uninett_physical_util_geographical7', 'new_data_average/10_runs_uninett_physical_util_geographical8', 'new_data_average/10_runs_uninett_physical_util_geographical9']
    file_list2 = ['new_data_average/10_runs_uninett_virtual_util_geographical0', 'new_data_average/10_runs_uninett_virtual_util_geographical1', 'new_data_average/10_runs_uninett_virtual_util_geographical2', 'new_data_average/10_runs_uninett_virtual_util_geographical3', 'new_data_average/10_runs_uninett_virtual_util_geographical4', 'new_data_average/10_runs_uninett_virtual_util_geographical5', 'new_data_average/10_runs_uninett_virtual_util_geographical6', 'new_data_average/10_runs_uninett_virtual_util_geographical7', 'new_data_average/10_runs_uninett_virtual_util_geographical8', 'new_data_average/10_runs_uninett_virtual_util_geographical9']
    file_list3 = ['new_data_average/10_runs_uninett_virtual_isol_util_geographical0', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical1', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical2', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical3', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical4', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical5', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical6', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical7', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical8', 'new_data_average/10_runs_uninett_virtual_isol_util_geographical9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)
    
    plotsSurvivability.plot_average_utilization_cdf(dict1, dict2, dict3, 'new_fig_uninett_geo_util')


    # geant geo
    file_list1 = ['new_data_average/10_runs_geant_physical_util_geographical0', 'new_data_average/10_runs_geant_physical_util_geographical1', 'new_data_average/10_runs_geant_physical_util_geographical2', 'new_data_average/10_runs_geant_physical_util_geographical3', 'new_data_average/10_runs_geant_physical_util_geographical4', 'new_data_average/10_runs_geant_physical_util_geographical5', 'new_data_average/10_runs_geant_physical_util_geographical6', 'new_data_average/10_runs_geant_physical_util_geographical7', 'new_data_average/10_runs_geant_physical_util_geographical8', 'new_data_average/10_runs_geant_physical_util_geographical9']
    file_list2 = ['new_data_average/10_runs_geant_virtual_util_geographical0', 'new_data_average/10_runs_geant_virtual_util_geographical1', 'new_data_average/10_runs_geant_virtual_util_geographical2', 'new_data_average/10_runs_geant_virtual_util_geographical3', 'new_data_average/10_runs_geant_virtual_util_geographical4', 'new_data_average/10_runs_geant_virtual_util_geographical5', 'new_data_average/10_runs_geant_virtual_util_geographical6', 'new_data_average/10_runs_geant_virtual_util_geographical7', 'new_data_average/10_runs_geant_virtual_util_geographical8', 'new_data_average/10_runs_geant_virtual_util_geographical9']
    file_list3 = ['new_data_average/10_runs_geant_virtual_isol_util_geographical0', 'new_data_average/10_runs_geant_virtual_isol_util_geographical1', 'new_data_average/10_runs_geant_virtual_isol_util_geographical2', 'new_data_average/10_runs_geant_virtual_isol_util_geographical3', 'new_data_average/10_runs_geant_virtual_isol_util_geographical4', 'new_data_average/10_runs_geant_virtual_isol_util_geographical5', 'new_data_average/10_runs_geant_virtual_isol_util_geographical6', 'new_data_average/10_runs_geant_virtual_isol_util_geographical7', 'new_data_average/10_runs_geant_virtual_isol_util_geographical8', 'new_data_average/10_runs_geant_virtual_isol_util_geographical9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)
    
    plotsSurvivability.plot_average_utilization_cdf(dict1, dict2, dict3, 'new_fig_geant_geo_util')



def avg_number_nfs():
    def dict_to_dict(dict):
        new_dict = {}
        for i in range(len(dict[0])):
            new_dict[i] = dict[0][i]
        return new_dict

    # uninett attack
    file_list1 = ['new_data_average/10_runs_uninett_physical_nfs_attack9']
    file_list2 = ['new_data_average/10_runs_uninett_virtual_nfs_attack9']
    file_list3 = ['new_data_average/10_runs_uninett_virtual_isol_nfs_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    dict1 = dict_to_dict(dict1)
    dict2 = dict_to_dict(dict2)
    dict3 = dict_to_dict(dict3)
    
    plotsSurvivability.plot_average_amount_of_nfs(dict1, dict2, dict3,'new_fig_uninett_attack_nfs')


    # geant attack
    file_list1 = ['new_data_average/10_runs_geant_physical_nfs_attack9']
    file_list2 = ['new_data_average/10_runs_geant_virtual_nfs_attack9']
    file_list3 = ['new_data_average/10_runs_geant_virtual_isol_nfs_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    dict1 = dict_to_dict(dict1)
    dict2 = dict_to_dict(dict2)
    dict3 = dict_to_dict(dict3)

    plotsSurvivability.plot_average_amount_of_nfs(dict1, dict2, dict3,'new_fig_geant_attack_nfs')


    # uninett geo
    file_list1 = ['new_data_average/10_runs_uninett_physical_nfs_geographical9']
    file_list2 = ['new_data_average/10_runs_uninett_virtual_nfs_geographical9']
    file_list3 = ['new_data_average/10_runs_uninett_virtual_isol_nfs_geographical9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    dict1 = dict_to_dict(dict1)
    dict2 = dict_to_dict(dict2)
    dict3 = dict_to_dict(dict3)
    
    plotsSurvivability.plot_average_amount_of_nfs(dict1, dict2, dict3,'new_fig_uninett_geo_nfs')


    # geant geo
    file_list1 = ['new_data_average/10_runs_geant_physical_nfs_geographical9']
    file_list2 = ['new_data_average/10_runs_geant_virtual_nfs_geographical9']
    file_list3 = ['new_data_average/10_runs_geant_virtual_isol_nfs_geographical9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)

    dict1 = dict_to_dict(dict1)
    dict2 = dict_to_dict(dict2)
    dict3 = dict_to_dict(dict3)
    
    plotsSurvivability.plot_average_amount_of_nfs(dict1, dict2, dict3,'new_fig_geant_geo_nfs')



def avg_init_perf():
    # uninett attack
    n = 10
    file_list1 = ['new_data_average/10_runs_uninett_physical_init_perf_attack0', 'new_data_average/10_runs_uninett_physical_init_perf_attack1', 'new_data_average/10_runs_uninett_physical_init_perf_attack2', 'new_data_average/10_runs_uninett_physical_init_perf_attack3', 'new_data_average/10_runs_uninett_physical_init_perf_attack4', 'new_data_average/10_runs_uninett_physical_init_perf_attack5', 'new_data_average/10_runs_uninett_physical_init_perf_attack6', 'new_data_average/10_runs_uninett_physical_init_perf_attack7', 'new_data_average/10_runs_uninett_physical_init_perf_attack8', 'new_data_average/10_runs_uninett_physical_init_perf_attack9']
    file_list2 = ['new_data_average/10_runs_uninett_virtual_init_perf_attack0', 'new_data_average/10_runs_uninett_virtual_init_perf_attack1', 'new_data_average/10_runs_uninett_virtual_init_perf_attack2', 'new_data_average/10_runs_uninett_virtual_init_perf_attack3', 'new_data_average/10_runs_uninett_virtual_init_perf_attack4', 'new_data_average/10_runs_uninett_virtual_init_perf_attack5', 'new_data_average/10_runs_uninett_virtual_init_perf_attack6', 'new_data_average/10_runs_uninett_virtual_init_perf_attack7', 'new_data_average/10_runs_uninett_virtual_init_perf_attack8', 'new_data_average/10_runs_uninett_virtual_init_perf_attack9']
    file_list3 = ['new_data_average/10_runs_uninett_virtual_isol_init_perf_attack0', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack1', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack2', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack3', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack4', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack5', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack6', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack7', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack8', 'new_data_average/10_runs_uninett_virtual_isol_init_perf_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)
    
    plotsSurvivability.plot_average_initial_performance(dict1, dict2, dict3, n, 'new_fig_uninett_attack_init_perf')


    # geant attack
    n = 10
    file_list1 = ['new_data_average/10_runs_geant_physical_init_perf_attack0', 'new_data_average/10_runs_geant_physical_init_perf_attack1', 'new_data_average/10_runs_geant_physical_init_perf_attack2', 'new_data_average/10_runs_geant_physical_init_perf_attack3', 'new_data_average/10_runs_geant_physical_init_perf_attack4', 'new_data_average/10_runs_geant_physical_init_perf_attack5', 'new_data_average/10_runs_geant_physical_init_perf_attack6', 'new_data_average/10_runs_geant_physical_init_perf_attack7', 'new_data_average/10_runs_geant_physical_init_perf_attack8', 'new_data_average/10_runs_geant_physical_init_perf_attack9']
    file_list2 = ['new_data_average/10_runs_geant_virtual_init_perf_attack0', 'new_data_average/10_runs_geant_virtual_init_perf_attack1', 'new_data_average/10_runs_geant_virtual_init_perf_attack2', 'new_data_average/10_runs_geant_virtual_init_perf_attack3', 'new_data_average/10_runs_geant_virtual_init_perf_attack4', 'new_data_average/10_runs_geant_virtual_init_perf_attack5', 'new_data_average/10_runs_geant_virtual_init_perf_attack6', 'new_data_average/10_runs_geant_virtual_init_perf_attack7', 'new_data_average/10_runs_geant_virtual_init_perf_attack8', 'new_data_average/10_runs_geant_virtual_init_perf_attack9']
    file_list3 = ['new_data_average/10_runs_geant_virtual_isol_init_perf_attack0', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack1', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack2', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack3', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack4', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack5', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack6', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack7', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack8', 'new_data_average/10_runs_geant_virtual_isol_init_perf_attack9']
    path = 'new_data_average/'
    dict1 = read_files_and_put_in_dict(file_list1)
    dict2 = read_files_and_put_in_dict(file_list2)
    dict3 = read_files_and_put_in_dict(file_list3)
    
    plotsSurvivability.plot_average_initial_performance(dict1, dict2, dict3, n, 'new_fig_geant_attack_init_perf')



plot_avg_performance()
avg_cumul_util()
avg_number_nfs()
avg_init_perf()