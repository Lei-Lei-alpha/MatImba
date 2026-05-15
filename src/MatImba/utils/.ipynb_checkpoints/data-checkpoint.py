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

#############################
# Element data
#############################
ele_lst = ['H','He','Li','Be','B','C','N','O','F','Ne',
           'Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca',
           'Sc', 'Ti', 'V','Cr', 'Mn', 'Fe', 'Co', 'Ni',
           'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
           'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
           'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
           'I', 'Xe','Cs', 'Ba','La', 'Ce', 'Pr', 'Nd', 'Pm',
           'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
           'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
           'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
           'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
           'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
           'Rf', 'Db', 'Sg', 'Bh','Hs', 'Mt', 'Ds', 'Rg', 'Cn',
           'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

A_ele_lib = ['Y', 'Ho', 'Er', 'Dy', 'Tm', 'Tb', 'Lu', 'Sc',
             'Gd', 'Sm', 'La', 'Nd', 'Pr', 'Yb', 'Ac', 'Ca',
             'Zr', 'Sr', 'Pm', 'Eu', 'Ce', 'Ba', 'Ti', 'Th',
             'Hf', 'Li', 'Pu', 'Pa', 'U', 'Np', 'Nb', 'Mg',
             'Na', 'K', 'Rb', 'Cs', 'Ta']

#############################
# Metallic radii
#############################
# Metallic radii
# Laws, K., Miracle, D. & Ferry, M. A predictive structural model for bulk metallic glasses. Nat Commun 6, 8123 (2015). https://doi.org/10.1038/ncomms9123
# Xiong, J., Zhang, TY. & Shi, SQ. Machine learning prediction of elastic properties and glass-forming ability of bulk metallic glasses. MRS Communications 9, 576–585 (2019). https://doi.org/10.1557/mrc.2019.44

metallic_radii = {
    'Li': 1.52, 'Be': 1.16, 'B': 0.88, 'C': 0.77, 'Na': 1.80, 'Mg': 1.60,
    'Al': 1.43, 'Si': 1.10, 'K': 2.30, 'Ca': 1.97, 'Sc': 1.62, 'Ti': 1.47,
    'V': 1.34, 'Cr': 1.28, 'Mn': 1.27, 'Fe': 1.26, 'Co': 1.25, 'Ni': 1.24,
    'Cu': 1.28, 'Zn': 1.34, 'Ga': 1.34, 'Ge': 1.25, 'Rb': 2.44, 'Sr': 2.15,
    'Y': 1.80, 'Zr': 1.60, 'Nb': 1.46, 'Mo': 1.400, 'Tc': 1.36, 'Ru': 1.34,
    'Rh': 1.32, 'Pd': 1.37, 'Ag': 1.44, 'Cd': 1.57, 'In': 1.67, 'Sn': 1.45,
    'Cs': 2.64, 'Ba': 2.23, 'La': 1.87, 'Ce': 1.818, 'Pr': 1.824, 'Nd': 1.814,
    'Pm': 1.85, 'Sm': 1.804, 'Eu': 1.96, 'Gd': 1.804, 'Tb': 1.773, 'Dy': 1.781,
    'Ho': 1.762, 'Er': 1.761, 'Tm': 1.759, 'Yb': 1.76, 'Lu': 1.738, 'Hf': 1.59,
    'Ta': 1.46, 'W': 1.408, 'Re': 1.37, 'Os': 1.35, 'Ir': 1.36, 'Pt': 1.385,
    'Au': 1.44, 'Hg': 1.52, 'Tl': 1.72, 'Pb': 1.80, 'Bi': 1.63, 'Po': 1.68,
    'Th': 1.78, 'Pa': 1.68, 'U': 1.58, 'Np': 1.75, 'Pu': 1.75, 
}

ele_ef = {
    'F': 666.0036341972414, 'O': 371.0981003283245, 'Cl': 314.8180877085921, 'Y': 236.2686018992784, 'Ho': 233.4913162787811,
    'Er': 233.1787820598742, 'Dy': 232.9312731440491, 'Tm': 232.75332208908245, 'Tb': 231.824274140677, 'Lu': 230.215259226846,
    'Sc': 226.9548139788774, 'Gd': 225.71065099059905, 'Sm': 222.1887952247816, 'Br': 216.324453907916, 'La': 215.5132117770606,
    'Nd': 214.5315821543097, 'Pr': 209.83483285042968, 'Yb': 208.83464862206236, 'Eu': 198.5492473787128, 'Ac': 196.2381472896883,
    'Ca': 195.09873091400905, 'Zr': 193.86127027709492, 'Li': 191.2455139135044, 'Sr': 188.90771634197003, 'Pm': 188.6570997150367,
    'I': 181.4922668093261, 'Ce': 180.90184479546247, 'Ba': 175.51847425663328, 'Ti': 174.53923344910194, 'Hf': 172.90120122453075,
    'Th': 172.45456594656605, 'Pu': 141.00838186411409, 'Pa': 124.69883713386226, 'Ta': 120.49357606854916, 'V': 114.75017473339528,
    'N': 108.75345638897376, 'K': 107.2704997768868, 'Na': 105.9738248670352, 'U': 103.42999341210556, 'Nb': 98.47546272412724,
    'Np': 94.54933041553714, 'S': 91.59584693827756, 'Rb': 91.34840574307258, 'Cs': 85.9112198877055, 'Mg': 85.218515653841,
    'B': 66.3138324672077, 'C': 61.35956107981837, 'Be': 48.19390739533714, 'Ni': 48.15604768927904, 'Al': 36.265526724063584,
    'Pd': 32.05556979960369, 'Co': 31.78051722518792, 'Si': 31.5356633436058, 'Fe': 26.427438487507, 'Cr': 22.379216333642923,
    'Rh': 11.82637014056869, 'Ge': 9.059158108020448, 'W': 1.297743446287774, 'Zn': -6.51733951527932, 'P': -7.538397400250166,
    'Sn': -12.126945730180651, 'Cu': -18.73803382032988, 'Ag': -46.65967228760927, 'Mn': -54.18294542670125, 'Pt': -82.09481291082682,
    'Re': -96.1814668254471, 'Ga': -137.64939231984334, 'Tl': -137.73721211424157, 'Cd': -145.54347366604307, 'In': -145.82507083835796,
    'Mo': -152.04397778294287, 'Bi': -160.4532801785683, 'Ir': -163.72735091979388, 'Sb': -175.00762069471648, 'Pb': -186.45622515637197,
    'Se': -186.77148099945455, 'As': -187.0327206428425, 'Tc': -516.8845209646033, 'Os': -724.1530238152611, 'Ru': -45.743867490034205,
    'Hg': -127.6115673550747, 'Au': -140.5579552572642
}

mat_cost = {
    'H': 1.39, '2H (D)': 13400.0, 'He': 24.0, 'Li': 83.5, 'Be': 857.0,
    'B': 3.68, 'C': 0.12, 'N': 0.14, 'O': 0.15, 'F': 2.0, 'Ne': 240.0,
    'Na': 3.0, 'Mg': 2.32, 'Al': 1.79, 'Si': 1.7, 'P': 2.69, 'S': 0.09,
    'Cl': 0.08, 'Ar': 0.93, 'K': 12.85, 'Ca': 2.2800000000000002,
    'Sc': 3460.0, 'Ti': 11.399999999999999, 'V': 371.0, 'Cr': 9.4,
    'Mn': 1.82, 'Fe': 0.42, 'Co': 32.8, 'Ni': 13.9, 'Cu': 6.0, 'Zn': 2.55,
    'Ga': 148.0, 'Ge': 962.0, 'As': 1.1545, 'Se': 21.4, 'Br': 4.39,
    'Kr': 290.0, 'Rb': 15500.0, 'Sr': 6.605, 'Y': 31.0, 'Zr': 36.400000000000006,
    'Nb': 73.5, 'Mo': 40.1, 'Tc': 100000.0, '99mTc': 1900000000000.0,
    'Ru': 10500.0, 'Rh': 147000.0, 'Pd': 49500.0, 'Ag': 521.0, 'Cd': 2.73,
    'In': 167.0, 'Sn': 18.7, 'Sb': 5.79, 'Te': 63.5, 'I': 35.0, 'Xe': 1800.0,
    'Cs': 61800.0, 'Ba': 0.2605, 'La': 4.85, 'Ce': 4.640000000000001,
    'Pr': 103.0, 'Nd': 57.5, '147Pm': 460000.0, 'Sm': 13.9, 'Eu': 31.4,
    'Gd': 28.6, 'Tb': 658.0, 'Dy': 307.0, 'Ho': 57.1, 'Er': 26.4, 'Tm': 3000.0,
    'Yb': 17.1, 'Lu': 643.0, 'Hf': 900.0, 'Ta': 305.0, 'W': 35.3, 'Re': 3580.0,
    'Os': 12000.0, 'Ir': 55850.0, 'Pt': 27800.0, 'Au': 44800.0, 'Hg': 30.2,
    'Tl': 4200.0, 'Pb': 2.0, 'Bi': 6.36, '209Po': 49200000000000.0,
    '225Ac': 29000000000000.0, 'Th': 287.0, 'U': 101.0, 'Np': 660000.0,
    '239Pu': 6490000.0, '241Am': 728000.0, '243Am': 750000.0, '244Cm': 185000000.0,
    '248Cm': 160000000000.0, '249Bk': 185000000000.0, '249Cf': 185000000000.0,
    '252Cf': 60000000000.0
}

#############################
# Translations for all feature names
#############################
magpie_features = {
    '0-norm': None, '2-norm': None, '3-norm': None, '5-norm': None, '7-norm': None, '10-norm': None,
    'min Number': r"min{$N$}", 'max Number': r"max{$N$}", 'range Number': '$R(\mathrm{N})$', 'mean Number': r"$\bar{N}$",
    'avg_dev Number': r"$\hat{N}$", 'mode Number': r"$\mathring{N}$", 'min MendeleevNumber': "min{MN}",
    'max MendeleevNumber': "max{MN}", 'range MendeleevNumber': '$R(\mathrm{MN}$)', 'mean MendeleevNumber': r"$\bar{\mathrm{MN}}$",
    'avg_dev MendeleevNumber': r"$\hat{\mathrm{MN}}$", 'mode MendeleevNumber': "$\mathring{\mathrm{MN}}$",
    'min AtomicWeight': r"min{$A_r$}", 'max AtomicWeight': r"max{$A_r$}", 'range AtomicWeight': '$R(A_r)$',
    'mean AtomicWeight': r"$\bar{A_r}$", 'avg_dev AtomicWeight': r"$\hat{A_r}$", 'mode AtomicWeight': r"$\mathring{A_r}$",
    'min MeltingT': r"min{$T_{\mathrm{m}}$}", 'max MeltingT': r"max{$T_{\mathrm{m}}$}", 'range MeltingT': '$R(T_{\mathrm{m}})$',
    'mean MeltingT': r"$\bar{T}_{\mathrm{m}}$", 'avg_dev MeltingT': r"$\hat{T}_{\mathrm{m}}$",
    'mode MeltingT': "$\mathring{T}_{\mathrm{m}}$", 'max Column': "max{Col}", 'range Column': '$R$(Col)',
    'mean Column': r"$\bar{\mathrm{Col}}$", 'avg_dev Column': r"$\hat{\mathrm{Col}}$",
    'mode Column': r"$\mathring{\mathrm{Col}}$", 'max Row': "max{Row}", 'range Row': '$R$(row)',
    'mean Row': r"$\bar{\mathrm{Row}}$", 'avg_dev Row': r"$\hat{\mathrm{Row}}$", 'min CovalentRadius': r"min{$r_{\mathrm{cov}}$}",
    'range CovalentRadius': '$R(r)$', 'mean CovalentRadius': r"$\bar{r}_{\mathrm{cov}}$",
    'avg_dev CovalentRadius': r"$\hat{r}_{\mathrm{cov}}$", 'mode CovalentRadius': r"$\mathring{r}_{\mathrm{cov}}$",
    'min Electronegativity': r"min{$\chi$}", 'max Electronegativity': r"max{$\chi$}", 'range Electronegativity': '$R(\chi)$',
    'mean Electronegativity': r"$\bar{\chi}$", 'avg_dev Electronegativity': r"$\hat{\chi}$",
    'mode Electronegativity': r"$\mathring{\chi}$", 'min NsValence': r"min{$N^s_{\mathrm{v}}$}",
    'mean NsValence': r"$\bar{N}^s_{\mathrm{v}}$", 'avg_dev NsValence': r"$\hat{N}^s_{\mathrm{v}}$",
    'mean NpValence': r"$\bar{N}^p_{\mathrm{v}}$", 'min NdValence': r"min{$N^d_{\mathrm{v}}$}",
    'max NdValence': r"max{$N^d_{\mathrm{v}}$}", 'range NdValence': '$R(N^d_{\mathrm{v}})$', 'mean NdValence': r"$\bar{N}^d_{\mathrm{v}}$",
    'avg_dev NdValence': r"$\hat{N}^d_{\mathrm{v}}$", 'mode NdValence': r"$\mathring{N}^d_{\mathrm{v}}$",
    'max NValence': r"max{$N_{\mathrm{v}}$}", 'range NValence': '$R(N_{\mathrm{v}})$', 'mean NValence': r"$\bar{N}_{\mathrm{v}}$",
    'avg_dev NValence': r"$\hat{N}_{\mathrm{v}}$", 'mode NValence': r"Mo{$N_{\mathrm{v}}$}",
    'mean NsUnfilled': r"$\bar{N}^s_{\mathrm{uf}}$", 'avg_dev NsUnfilled': r"$\hat{N}^p_{\mathrm{uf}}$",
    'mean NpUnfilled': r"$\bar{N}^s_{\mathrm{uf}}$", 'avg_dev NpUnfilled': r"$\hat{N}^p_{\mathrm{uf}}$",
    'min NdUnfilled': r"min{$N^d_{\mathrm{uf}}$}", 'max NdUnfilled': r"max{$N^d_{\mathrm{uf}}$}",
    'range NdUnfilled': '$R(N^d_{\mathrm{uf}})$', 'mean NdUnfilled': r"$\bar{N}^d_{\mathrm{uf}}$",
    'avg_dev NdUnfilled': r"$\hat{N}^d_{\mathrm{uf}}$", 'mode NdUnfilled': r"$\mathring{N}^d_{\mathrm{uf}}$",
    'min NUnfilled': r"min{$N_{\mathrm{uf}}$}", 'range NUnfilled': '$R(N_{\mathrm{uf}})$', 'mean NUnfilled': r"$\bar{N}_{\mathrm{uf}}$",
    'avg_dev NUnfilled': r"$\hat{N}_{\mathrm{uf}}$", 'mode NUnfilled': r"$\mathring{N}_{\mathrm{uf}}$",
    'min GSvolume_pa': r"min{$\nu_{\mathrm{pa}}$}", 'max GSvolume_pa': r"max{$\nu_{\mathrm{pa}}$}",
    'range GSvolume_pa': '$R(\nu_{\mathrm{pa}})$', 'mean GSvolume_pa': r"$\bar{\nu}_{\mathrm{pa}}$",
    'avg_dev GSvolume_pa': r"$\hat{\nu}_{\mathrm{pa}}$", 'mode GSvolume_pa': r"$\mathring{\nu}_{\mathrm{pa}}$",
    'min GSmagmom': r"min{$\mu$}", 'max GSmagmom': r"max{$\mu$}", 'range GSmagmom': '$R(\mu)$',
    'mean GSmagmom': r"$\bar{\mu}$", 'avg_dev GSmagmom': r"$\hat{\mu}$", 'mode GSmagmom': r"$\mathring{\mu}$",
    'max SpaceGroupNumber': "max{SG}", 'range SpaceGroupNumber': '$R(SG)$', 'mean SpaceGroupNumber': r"$\bar{\mathrm{SG}}$",
    'avg_dev SpaceGroupNumber': r"$\bar{\mathrm{SG}}$", 'mode SpaceGroupNumber': r"$\mathring{\mathrm{SG}}$",
    'avg s valence electrons': r"$\bar{V}_{\mathrm{s}}$", 'avg p valence electrons': r"$\bar{V}_{\mathrm{p}get}$",
    'avg d valence electrons': r"$\bar{V}_{\mathrm{d}}$", 'max ionic char': r"max{$C_{\mathrm{Ion}}$}",
    'avg ionic char': r"$\bar{C}_{\mathrm{Ion}}$", 'E_HM min': r"min{$\Delta H_{\mathrm{b}}$}",
    'E_HM max': r"max{$\Delta H_{\mathrm{b}}$}", 'E_HM mu': r"$\bar{\Delta H}_{\mathrm{b}}$",
    'E_HM dev': r"$\hat{\Delta H}_{\mathrm{b}}$"
}