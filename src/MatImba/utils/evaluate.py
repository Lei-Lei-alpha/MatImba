import sys
from .losses import *
from torch.optim import *
from torch.optim.lr_scheduler import *

import numpy as np
import pandas as pd
from mendeleev import element

from pymatgen.core import Composition
from MatImba.utils import auto_featurise, metallic_radii, mat_cost

def get_obj(obj_name):
    """
    Return object from the object name
    """
    from ..models import gbr_ensemble
    from ..models.resnet import ResNet
    from ..models.megnet import MEGNet
    return getattr(sys.modules[__name__], obj_name)

def load_model(ckpt, **params):
    saved_states = torch.load(ckpt)
    model = get_obj(saved_states["model"]["name"])(**params)
    model.load_state_dict(saved_states["model"]["states"])
    return model

def calc_peq0(dH, dS, temperature):
    return np.exp((dS - 1000 * dH / (temperature + 273.15)) / 8.314)

def get_eles(formula):
    all_eles = list(Composition(formula).as_dict().keys())
    return ''.join(all_eles)

def max_mp(formula):
    elements = list(Composition(formula).as_dict().keys())
    ele_mp = np.array([element(ele).melting_point for ele in elements])
    return ele_mp.max()
    
def max_mp_diff(formula):
    elements = list(Composition(formula).as_dict().keys())
    ele_mp = np.array([element(ele).melting_point for ele in elements])
    return ele_mp.max() - ele_mp.min()

def cost_and_wtfrac(df, mat_cost = mat_cost):
    for ind in df.index:
        formula, HtoM = df.loc[ind, ["Formula", "HtoM"]].values
        HtoM = 0 if HtoM < 0 else HtoM 
        comp = Composition(formula)
        els = [el.symbol for el in comp.elements]
        MWs = [el._atomic_mass for el in comp.elements]
        stoich = [comp.get_atomic_fraction(el) for el in comp.elements]
        assert np.isclose(np.sum(stoich), 1.0, rtol = 1e-05) # normalised so there is 1 metal atom
        newcomp = Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
        wt_frac = newcomp.get_wt_fraction('H') * 100
        
        weight_dict = comp.to_weight_dict
        all_el_cost = [mat_cost[ele] * weight_dict[ele] for ele in weight_dict.keys()]
            
        matertial_costs = np.sum(all_el_cost)
        
        df.loc[ind, ["Hwtpercent", "matcost"]] = [wt_frac, matertial_costs]
    return df
    
def get_mat_cost(formula, mat_cost = mat_cost):
    comp = Composition(formula)
    weight_dict = comp.to_weight_dict
    matertial_costs = np.sum([mat_cost[ele] * weight_dict[ele] for ele in weight_dict.keys()])
    return matertial_costs

def wtfrac(formula, HtoM):
    HtoM = 0 if HtoM < 0 else HtoM
    comp = Composition(formula)
    els = [el.symbol for el in comp.elements]
    MWs = [el._atomic_mass for el in comp.elements]
    stoich = [comp.get_atomic_fraction(el) for el in comp.elements]
    assert np.isclose(np.sum(stoich), 1.0, rtol = 1e-05) # normalised so there is 1 metal atom
    newcomp = Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
    wt_frac = newcomp.get_wt_fraction('H') * 100
    return wt_frac

def predict_compositions(formula, lds_models = True):

    fea_file = "ml_data/ab2_ml_fea_19112024.csv"
    data_file = "ml_data/ab2_final_19112024.csv"
    
    target_names = {
        "wt_pct": "Hydrogen_Weight_Percent",
        "dH": 'Heat_of_Formation_kJperMolH2',
        'HtoM': 'HtoM',
        'dS': 'Entropy_of_Formation_JperMolH2perK',
        'lnpeq': 'LnEquilibrium_Pressure_25C',
        'slope': 'Slope'
    }
    
    predictors = {}
    if lds_models:
        for key, name in target_names.items():
            model = gbr_ensemble()
            model.load_data(fea_file, data_file, target_name = name)
            model.load(f"trained_models/{key}/lds.pkl")
            predictors.update({key: model})
    else:
        for key, name in target_names.items():
            model = gbr_ensemble()
            model.load_data(fea_file, data_file, target_name = name)
            model.load(f"trained_models/{key}/control.pkl")
            predictors.update({key: model})
        
    if not isinstance(formula, pd.DataFrame):
        formula = pd.DataFrame({'Formula': formula})
    print("Generating features ...")
    pred_fea = auto_featurise(formula)
    pred_fea = pred_fea.values
    predictions = {}
    for model_name, model in predictors.items():
        print("-" * 53)
        print(f"Predicting {model_name} ...")
        predictions.update({model_name: model.predict(pred_fea)})

    predictions = pd.DataFrame(predictions)
    predictions = pd.concat([formula, predictions], axis = 1)

    print("-" * 53)
    print("Calculating H wt percent and materials costs ...")
    # predictions = cost_and_wtfrac(predictions)
    # predictions["Hwtpercent"] = predictions[['Formula', 'HtoM']].apply(wtfrac)
    # predictions["Hwtpercent"] = predictions.apply(lambda x: wtfrac(x.Formula, x.HtoM), axis = 1)
    # predictions["matcost"] = predictions['Formula'].apply(get_mat_cost)
    # predictions['ele_comp'] = predictions['Formula'].apply(get_eles)
    # predictions['max_mp'] = predictions['Formula'].apply(max_mp)
    # predictions['max_mp_diff'] = predictions['Formula'].apply(max_mp_diff)
    return predictions

def exclude_elements(dataframe, element, formula_col = "formula", drop_index = False):
    mask = [element not in Composition(row).as_dict() for row in dataframe.loc[:, formula_col]]
    result_df = dataframe[mask].reset_index(drop = True) if drop_index else dataframe[mask]
    return result_df


class dil_analysier():
    def __init__(self, models, names = None,):
        if isinstance(models, list) or isinstance(models, tuple):
            self.nmodels = len(models)
            self.models = models
        else:
            self.nmodels = 1
            self.models = [models]
        self.names = [f"model {i}" for i in range(self.nmodels)] if names is None else names
        self.results = {name: None for name in self.names}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for model in self.models:
            model.to(self.device)
    
    def evaluate(self, data_loader):
        criterion_maes = nn.L1Loss(reduction = "none")
        for i, model in enumerate(self.models):
            labels_all, losses_all, densities_all = [], [], []
            model.eval()
            with torch.no_grad():
                for idx, (inputs, targets, weights, densities) in enumerate(data_loader):
                    labels_all.extend(targets.numpy())
                    densities_all.extend(densities.numpy())
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                    
                    outputs = model(inputs)
                    loss_all = criterion_maes(outputs, targets)
                    losses_all.extend(loss_all.cpu().numpy())
            
            labels_all = np.array(labels_all).reshape(-1)
            losses_all = np.array(losses_all).reshape(-1)
            densities_all = np.array(densities_all).reshape(-1)

            pearson_cor = np.corrcoef(1 / densities_all, y = losses_all)
            awareness = 1 - np.abs(pearson_cor[0][-1])

            hist, bin_edges = np.histogram(labels_all, bins = "fd")
            bin_width = (bin_edges[-1] - bin_edges[0]) / (len(bin_edges) - 1)
            x = (bin_edges[:-1] + bin_edges[1:])/2
            y_locs = np.fmin(np.digitize(labels_all, bin_edges), len(x))
            binned_AEs = np.zeros(len(x))
            binned_density = np.zeros(len(x))
                
            for j in range(len(x)):
                locs = np.where(y_locs == j + 1)
                binned_AEs[j] = losses_all[locs].mean() if losses_all[locs].size != 0 else np.nan
                binned_density[j] = densities_all[locs].mean() if densities_all[locs].size != 0 else np.nan
            
            x = x[~np.isnan(binned_AEs)]
            counts = hist[~np.isnan(binned_AEs)]
            binned_AEs = binned_AEs[~np.isnan(binned_AEs)]
            binned_density = binned_density[~np.isnan(binned_density)]
            self.results.update({self.names[i]: {"MAEs": binned_AEs, "MAE": losses_all.mean(), "awareness": awareness}})
        self.results.update({"bin": x, "bin_width": bin_width, "counts": counts, "density": binned_density})
        
    def plot_one(self, name, target_name = None, filename = None, **kwargs):
        fig, ax = plt.subplots(figsize=(4, 3))
        axtwin = ax.twinx()
        x = self.results["name"]["bin"]
        binned_AEs = self.results["name"]["MAE"]
        counts = self.results["name"]["counts"]
        awareness = self.results["name"]["awareness"]
        bin_width = self.results["name"]["bin_width"]
        
        ax.plot(x, binned_AEs, c = "#4d4d4d", marker = "s", markerfacecolor = "#f0f0f0",
                        label = f"DIL awareness: %.2f"%awareness)
        
        axtwin.bar(x, counts, width = 0.98 * bin_width, edgecolor = '#053061', color = '#4393c3', alpha = 0.85)

        if target_name is None:
            target_name = "Targets"
        ax.set_xlabel(target_name)
        axtwin.set_ylabel(r"Testset counts")
        axtwin.tick_params(axis = 'y', colors = '#1f77b4')
        axtwin.yaxis.label.set_color('#1f77b4')
        ax.spines['right'].set_color('#1f77b4')
        axtwin.spines['right'].set_color('#1f77b4')
        ax.set_zorder(axtwin.get_zorder()+1)
        ax.patch.set_visible(False)
        
        # ax.legend()
        if filename is not None:
            plt.savefig(filename, dpi = 600, **kwargs)
        plt.show()