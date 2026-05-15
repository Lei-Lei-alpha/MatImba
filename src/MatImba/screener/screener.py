import os
import json
import torch
import random
import itertools
import numpy as np
import pandas as pd
from more_itertools import chunked
from pymatgen.core import Composition
from ..utils import auto_featurise, metallic_radii, mat_cost

python_env = os.environ['_'].split('/')[-1]
if python_env == 'jupyter':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class compound_generater:
    def __init__(self, predictors, elements = None, nonstoichiometry = 0,
                 spacing = 0.1, mat_cost = mat_cost, constraints = None):
        
        self.predictors = predictors
        self.elements = elements
        self.nonstoichiometry = nonstoichiometry
        self.spacing = spacing
        self.mat_cost = mat_cost
        self.spacing = spacing
        self.ele_fracs = {}
        self.constraints = constraints
        
        if self.elements is not None:
            if self.constraints is not None:
                self._set_constraint(constraints)
            else:
                self._split_eles()
                for ele in self.elements:
                    self.ele_fracs.update({ele: np.arange(0, 1.01, self.spacing)})
        self.results = None
    
    
    def _split_eles(self):
        A_ele_lib = ['Y', 'Ho', 'Er', 'Dy', 'Tm', 'Tb', 'Lu', 'Sc',
                     'Gd', 'Sm', 'La', 'Nd', 'Pr', 'Yb', 'Ac', 'Ca',
                     'Zr', 'Sr', 'Pm', 'Eu', 'Ce', 'Ba', 'Ti', 'Th',
                     'Hf', 'Li', 'Pu', 'Pa', 'U', 'Np', 'Nb', 'Mg',
                     'Na', 'K', 'Rb', 'Cs', 'Ta']
        self.a_elements = [element for element in self.elements if element in A_ele_lib]
        self.b_elements = [element for element in self.elements if element not in A_ele_lib]
               
    
    def _set_constraint(self, constraints):
        if "excludes" in constraints and constraints["excludes"] is not None:
            self.elements = set([ele for ele in self.elements if ele not in constraints["excludes"]])
        
        self._split_eles()
        for ele in self.a_elements:
            self.ele_fracs.update({ele: np.arange(0, 1.01, self.spacing)})
            
        for ele in self.b_elements:
            self.ele_fracs.update({ele: np.arange(0, 2.01, self.spacing)})
        
        if "contents" in constraints and constraints["excludes"] is not None:
            for ele in list(constraints["contents"].keys()):
                self.ele_fracs.update({ele: np.arange(constraints["contents"][ele][0],
                                                      constraints["contents"][ele][1] + 0.01, self.spacing)})
                    
        if "RA/RB" in constraints:
            self.size_ratio_lim = [float(_) for _ in constraints["RA/RB"]]
        else:
            self.size_ratio_lim = None
            
        self.a_num = constraints["A_nums"]
        self.b_num = constraints["B_nums"]
                      
    def _generate_compounds(self, exclusive = True):
        """
        exclusive: bool, only keep compounds with total number of elements equals to a_num + b_num
        """
        print("-"*53)
        print("Get combinations ...")
        a_combs = itertools.combinations(self.a_elements, self.a_num)
        b_combs = itertools.combinations(self.b_elements, self.b_num)
        a_b_combs = list(set(itertools.product(a_combs, b_combs)))
        ini_a_frac_combs = [np.asarray(list(itertools.product(*[self.ele_fracs[ele] for ele in a_b_comb[0]]))) for a_b_comb in a_b_combs]
        a_frac_combs =[[frac_arr for frac_arr in ini_a_frac_comb if frac_arr.sum() == 1] for ini_a_frac_comb in ini_a_frac_combs]
        ini_b_frac_combs = [np.asarray(list(itertools.product(*[self.ele_fracs[ele] for ele in a_b_comb[1]]))) for a_b_comb in a_b_combs]
        b_frac_combs =[[frac_arr for frac_arr in ini_b_frac_comb if frac_arr.sum() == 2 - self.nonstoichiometry] for ini_b_frac_comb in ini_b_frac_combs]

        print("-"*53)
        print("Formatting formula ...")
        n_ele = self.a_num + self.b_num
        formula_lst, RA_RB_lst = [], []
        for i in tqdm(range(len(a_b_combs))):
            ab_comb = np.asarray([ele for comb in a_b_combs[i] for ele in comb])
            a_b_fracs = list(itertools.product(a_frac_combs[i], b_frac_combs[i]))
            a_b_fracs = np.asarray([np.hstack(a_b_frac) for a_b_frac in a_b_fracs])
            
            if self.size_ratio_lim:
                R_Ms = np.asarray([metallic_radii[ele] for ele in ab_comb])
                RA_RB = (2 - self.nonstoichiometry) * R_Ms[:self.a_num].dot(a_b_fracs.T[:self.a_num]) / R_Ms[-self.b_num:].dot(a_b_fracs.T[-self.b_num:])
                a_b_fracs = a_b_fracs[np.where((self.size_ratio_lim[0] < RA_RB) & (self.size_ratio_lim[1] > RA_RB))]
                RA_RB = RA_RB[np.where((self.size_ratio_lim[0] < RA_RB) & (self.size_ratio_lim[1] > RA_RB))]

            if exclusive:
                for j in range(len(a_b_fracs)):
                    compound = Composition("".join(f'{el}{amt}' for el, amt in zip(ab_comb, a_b_fracs[j])))
                    if len(compound) == n_ele:
                        RA_RB_lst.append(RA_RB[j])
                        formula_lst.append(compound.to_pretty_string())
            else:
                RA_RB_lst += RA_RB.tolist()
                formula_lst += [Composition("".join(f'{el}{amt}' for el, amt in zip(ab_comb, contents))).to_pretty_string() for contents in a_b_fracs]    
        
        del ini_a_frac_combs, a_frac_combs, ini_b_frac_combs, b_frac_combs
        del a_b_fracs, RA_RB, a_b_combs
        formula_df = pd.DataFrame({"Formula": formula_lst, 'RA/RB': RA_RB_lst}).drop_duplicates(keep = 'last')
        formula_df = formula_df.dropna().reset_index(drop = True)
        print(f"Total compounds: {len(formula_df)}")
        return formula_df
        
    def calc_cost(self, formula):
        comp = Composition(formula)
        weight_dict = comp.to_weight_dict
        all_el_cost = [self.mat_cost[ele] * weight_dict[ele] for ele in weight_dict.keys()]
        return np.sum(all_el_cost)

    def HtoM2wtfrac(self, df):
        wt_fracs = np.zeros((len(df), 1))
        for i in range(len(df)):
            formula, HtoM = df.loc[i, ["Formula", "HtoM"]].values
            comp = Composition(formula)
            els = [el.symbol for el in comp.elements]
            MWs = [el._atomic_mass for el in comp.elements]
            stoich = [comp.get_atomic_fraction(el) for el in comp.elements]
            assert np.isclose(np.sum(stoich), 1.0, rtol = 1e-05) # normalised so there is 1 metal atom
            newcomp = Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
            wt_fracs[i] = newcomp.get_wt_fraction('H') * 100
            # df.loc[i, "Hwtpercent"] = newcomp.get_wt_fraction('H') * 100
        return wt_fracs

    def cost_and_wtfrac(self, df):
        for i in range(len(df)):
            formula, HtoM = df.loc[i, ["Formula", "HtoM"]].values
            HtoM = 0 if HtoM < 0 else HtoM 
            comp = Composition(formula)
            els = [el.symbol for el in comp.elements]
            MWs = [el._atomic_mass for el in comp.elements]
            stoich = [comp.get_atomic_fraction(el) for el in comp.elements]
            assert np.isclose(np.sum(stoich), 1.0, rtol = 1e-05) # normalised so there is 1 metal atom
            newcomp = Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
            wt_frac = newcomp.get_wt_fraction('H') * 100
            
            weight_dict = comp.to_weight_dict
            all_el_cost = [self.mat_cost[ele] * weight_dict[ele] for ele in weight_dict.keys()]
            mat_cost = np.sum(all_el_cost)
            
            df.loc[i, ["Hwtpercent", "matcost"]] = [wt_frac, mat_cost]
        return df
        
    
    def pred_formula(self, formula, H_WperC = True):
        if not isinstance(formula, pd.DataFrame):
            formula = pd.DataFrame({'Formula': formula})
        print("Generating features ...")
        pred_fea = auto_featurise(formula)
        pred_fea = pred_fea.values
        predictions = {}
        for model_name, model in self.predictors.items():
            print("-" * 53)
            print(f"Predicting {model_name} ...")
            predictions.update({model_name: model.predict(pred_fea)})

        predictions = pd.DataFrame(predictions)
        predictions = pd.concat([formula, predictions], axis = 1)

        if H_WperC:
            print("-" * 53)
            print("Calculating H wt percent and materials costs ...")
            predictions = self.cost_and_wtfrac(predictions)
        return predictions
            
    
    def screen_compounds(self, criteria = None, batch_max = 80000, filename = None):
        print("\n")
        print("="*53)
        print("Generating compound formula ...")
        formula_df = self._generate_compounds()

        print("="*53)
        if len(formula_df) > batch_max:
            idx_lst = formula_df.index.tolist()
            chunk_idx = list(chunked(idx_lst, batch_max))
            num_chunks = len(chunk_idx)
            chunk_preds = []
            for i, idx in enumerate(chunk_idx):
                print(f"Predicting for chunk {i+1}/{num_chunks} of compounds ...")
                print("="*53)
                chunk_df = formula_df.loc[idx].reset_index(drop = True)
                chunk_preds.append(self.pred_formula(chunk_df))
            results_df = pd.concat(chunk_preds, ignore_index = True)
            
        else:
            print("="*53)
            results_df = self.pred_formula(formula_df)

        self.results = results_df
            
        print("Screening ...")
        print("="*53)
        
        if criteria is not None:
            if "lt" in criteria and "gt" not in criteria:
                self.results = results_df[results_df[label] <= criteria["lt"]]
            elif "gt" in criteria and "lt" not in criteria:
                self.results = results_df[results_df[label] >= criteria["gt"]]
            else:
                self.results = results_df[(results_df[label] <= criteria["lt"]) & (results_df[label] >= criteria["gt"])]
        else:
            self.results = results_df

        if filename:
            self.results.to_csv(filename, index_col = None)
            print(f"Results saved to {filename}!" )
        print("Done!")