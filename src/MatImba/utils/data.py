#############################
# MatBench dataset
#############################
def get_key(dataset_name):
    key_map = {
        'matbench_steels': 'yield strength',
        'matbench_jdft2d': 'exfoliation_en',
        'matbench_phonons': 'last phdos peak',
        'matbench_expt_gap': 'gap expt',
        'matbench_dielectric': 'n',
        'matbench_expt_is_metal': 'is_metal',
        'matbench_glass': 'gfa',
        'matbench_log_gvrh': 'log10(G_VRH)',
        'matbench_log_kvrh': 'log10(K_VRH)',
        'matbench_perovskites': 'e_form',
        'matbench_mp_gap': 'gap pbe',
        'matbench_mp_is_metal': 'is_metal',
        'matbench_mp_e_form':'e_form' 
    }
    return key_map[dataset_name]
