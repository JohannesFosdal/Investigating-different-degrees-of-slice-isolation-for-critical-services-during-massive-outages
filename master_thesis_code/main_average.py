import mainSurvivability
import mainSurvivability_attack
import plotsSurvivability
import numpy as np

number_of_simulations = 10
#failure_type = 'geographical'
failure_type = 'attack'
filename = '10_runs_geant'    


def mainA():
    clusPR = {}
    virt_clusPR = {}
    virt_isolation = {}
    phys_util = {}
    virt_util = {}
    virt_isol_util = {}
    phys_nfs = {}
    virt_nfs = {}
    virt_isol_nfs = {}
    initial_performance_original = {}
    initial_performance_virt = {}
    initial_performance_virt_isol = {}

    for i in range(number_of_simulations):
        temp_phys, temp_virt, temp_virt_isol, temp_phys_util, temp_virt_util, temp_virt_isol_util, temp_phys_nfs, temp_virt_nfs, temp_virt_isol_nfs, temp_initial_performance_original, temp_initial_performance_virt, temp_initial_performance_virt_isol = mainSurvivability_attack.main()
        
        clusPR[i] = temp_phys
        virt_clusPR[i] = temp_virt
        virt_isolation[i] = temp_virt_isol
        phys_util[i] = temp_phys_util
        virt_util[i] = temp_virt_util
        virt_isol_util[i] = temp_virt_isol_util
        phys_nfs[i] = temp_phys_nfs
        virt_nfs[i] = temp_virt_nfs
        virt_isol_nfs[i] = temp_virt_isol_nfs
        initial_performance_original[i] = temp_initial_performance_original
        initial_performance_virt[i] = temp_initial_performance_virt
        initial_performance_virt_isol[i] = temp_initial_performance_virt_isol
        
        file_path = 'new_data_average/' + filename + '_physical_' + 'attack' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nphysical\n')
            phys_str = '\n'.join(str(item) for item in temp_phys)
            file.write(phys_str)

        file_path = 'new_data_average/' + filename + '_virtual_' + 'attack' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nvirtual\n')
            virt_str = '\n'.join(str(item) for item in temp_virt)
            file.write(virt_str)

        file_path = 'new_data_average/' + filename + '_virtual_isolation_' + 'attack' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nvirtual_isolation\n')
            virt_isol_str = '\n'.join(str(item) for item in temp_virt_isol)
            file.write(virt_isol_str)



        file_path = 'new_data_average/' + filename + '_physical_util_' + 'attack' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nphysical_util\n')
            phys_util_str = '\n'.join(str(item) for item in temp_phys_util)
            file.write(phys_util_str)
        
        file_path = 'new_data_average/' + filename + '_virtual_util_' + 'attack' + str(i) 
        with open(file_path, 'w') as file:
            file.write('\nvirtual_util\n')
            virt_util_str = '\n'.join(str(item) for item in temp_virt_util)
            file.write(virt_util_str)

        file_path = 'new_data_average/' + filename + '_virtual_isol_util_' + 'attack' + str(i)       
        with open(file_path, 'w') as file:
            file.write('\nvirtual_isol_util\n')
            virt_isol_util_str = '\n'.join(str(item) for item in temp_virt_isol_util)
            file.write(virt_isol_util_str)




        file_path = 'new_data_average/' + filename + '_physical_init_perf_' + 'attack' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nphysical_initial_performance\n')
            phys_init_perf_str = '\n'.join(str(item) for item in temp_initial_performance_original)
            file.write(phys_init_perf_str)
        
        file_path = 'new_data_average/' + filename + '_virtual_init_perf_' + 'attack' + str(i)    
        with open(file_path, 'w') as file:
            file.write('\nvirtual_initial_performance\n')
            virt_init_perf_str = '\n'.join(str(item) for item in temp_initial_performance_virt)
            file.write(virt_init_perf_str)

        file_path = 'new_data_average/' + filename + '_virtual_isol_init_perf_' + 'attack' + str(i)       
        with open(file_path, 'w') as file:
            file.write('\nvirtual_isol_initial_performance\n')
            virt_isol_init_perf_str = '\n'.join(str(item) for item in temp_initial_performance_virt_isol)
            file.write(virt_isol_init_perf_str)

    


    file_path = 'new_data_average/' + filename + '_physical_nfs_' + 'attack' + str(i)
    with open(file_path, 'w') as file:
        file.write('\nphysical_nfs\n')
        phys_nfs_str = '\n'.join(str(item) for item in phys_nfs.values())
        file.write(phys_nfs_str)
    
    file_path = 'new_data_average/' + filename + '_virtual_nfs_' + 'attack' + str(i) 
    with open(file_path, 'w') as file:
        file.write('\nvirtual_nfs\n')
        virt_nfs_str = '\n'.join(str(item) for item in virt_nfs.values())
        file.write(virt_nfs_str)

    file_path = 'new_data_average/' + filename + '_virtual_isol_nfs_' + 'attack' + str(i)        
    with open(file_path, 'w') as file:
        file.write('\nvirtual_isol_nfs\n')
        virt_isol_nfs_str = '\n'.join(str(item) for item in virt_isol_nfs.values())
        file.write(virt_isol_nfs_str)



    mean_repair_time = 0.1
    time_points = np.linspace(0, 65 * mean_repair_time * 2.5, 100)

    plotsSurvivability.plot_average_performance_curves_combined(time_points, clusPR, time_points, virt_clusPR, time_points, virt_isolation, filename+'average_performance_curve_attack', n = number_of_simulations)

    plotsSurvivability.plot_average_utilization_cdf(phys_util, virt_util, virt_isol_util, filename+'average_cumulative_utilization_attack')

    plotsSurvivability.plot_average_amount_of_nfs(phys_nfs, virt_nfs, virt_isol_nfs, filename+'average_amount_nfs_attack')

    plotsSurvivability.plot_average_initial_performance(initial_performance_original, initial_performance_virt, initial_performance_virt_isol, number_of_simulations, filename+'average_initial_performance')

def main():
    clusPR = {}
    virt_clusPR = {}
    virt_isolation = {}
    phys_util = {}
    virt_util = {}
    virt_isol_util = {}
    phys_nfs = {}
    virt_nfs = {}
    virt_isol_nfs = {}

    for i in range(number_of_simulations):
        temp_phys, temp_virt, temp_virt_isol, temp_phys_util, temp_virt_util, temp_virt_isol_util, temp_phys_nfs, temp_virt_nfs, temp_virt_isol_nfs, len_failed_nodes = mainSurvivability.main()

        clusPR[i] = temp_phys
        virt_clusPR[i] = temp_virt
        virt_isolation[i] = temp_virt_isol
        phys_util[i] = temp_phys_util
        virt_util[i] = temp_virt_util
        virt_isol_util[i] = temp_virt_isol_util
        phys_nfs[i] = temp_phys_nfs
        virt_nfs[i] = temp_virt_nfs
        virt_isol_nfs[i] = temp_virt_isol_nfs
        number_of_failed_nodes = len_failed_nodes
        
        file_path = 'new_data_average/' + filename + '_physical' + 'geographical' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nphysical\n')
            phys_str = '\n'.join(str(item) for item in temp_phys)
            file.write(phys_str)

        file_path = 'new_data_average/' + filename + '_virtual' + 'geographical' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nvirtual\n')
            virt_str = '\n'.join(str(item) for item in temp_virt)
            file.write(virt_str)

        file_path = 'new_data_average/' + filename + '_virtual_isolation' + 'geographical' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nvirtual_isolation\n')
            virt_isol_str = '\n'.join(str(item) for item in temp_virt_isol)
            file.write(virt_isol_str)
        


        file_path = 'new_data_average/' + filename + '_physical_util_' + 'geographical' + str(i)
        with open(file_path, 'w') as file:
            file.write('\nphysical_util\n')
            phys_util_str = '\n'.join(str(item) for item in temp_phys_util)
            file.write(phys_util_str)
        
        file_path = 'new_data_average/' + filename + '_virtual_util_' + 'geographical' + str(i) 
        with open(file_path, 'w') as file:
            file.write('\nvirtual_util\n')
            virt_util_str = '\n'.join(str(item) for item in temp_virt_util)
            file.write(virt_util_str)

        file_path = 'new_data_average/' + filename + '_virtual_isol_util_' + 'geographical' + str(i)       
        with open(file_path, 'w') as file:
            file.write('\nvirtual_isol_util\n')
            virt_isol_util_str = '\n'.join(str(item) for item in temp_virt_isol_util)
            file.write(virt_isol_util_str)




    file_path = 'new_data_average/' + filename + '_physical_nfs_' + 'geographical' + str(i)
    with open(file_path, 'w') as file:
        file.write('\nphysical_nfs\n')
        phys_nfs_str = '\n'.join(str(item) for item in phys_nfs.values())
        file.write(phys_nfs_str)
    
    file_path = 'new_data_average/' + filename + '_virtual_nfs_' + 'geographical' + str(i) 
    with open(file_path, 'w') as file:
        file.write('\nvirtual_nfs\n')
        virt_nfs_str = '\n'.join(str(item) for item in virt_nfs.values())
        file.write(virt_nfs_str)

    file_path = 'new_data_average/' + filename + '_virtual_isol_nfs_' + 'geographical' + str(i)        
    with open(file_path, 'w') as file:
        file.write('\nvirtual_isol_nfs\n')
        virt_isol_nfs_str = '\n'.join(str(item) for item in virt_isol_nfs.values())
        file.write(virt_isol_nfs_str)



    mean_repair_time = 5
    time_points = np.linspace(0, number_of_failed_nodes * mean_repair_time * 1.5, 100)

    plotsSurvivability.plot_average_performance_curves_combined(time_points, clusPR, time_points, virt_clusPR, time_points, virt_isolation, filename+'average_performance_curve_geographical', n = number_of_simulations)

    plotsSurvivability.plot_average_utilization_cdf(phys_util, virt_util, virt_isol_util, filename+'average_cumulative_utilization_geographical')

    plotsSurvivability.plot_average_amount_of_nfs(phys_nfs, virt_nfs, virt_isol_nfs, filename+'average_amount_nfs_geographical')

if __name__ == "__main__":
    if failure_type == "" 'attack':
        mainA()
    if failure_type == 'geographical':
        main()