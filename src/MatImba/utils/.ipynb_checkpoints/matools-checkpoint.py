import sys
import numpy as np
import pandas as pd
import numpy.typing as npt
from pymatgen.core import Composition
from mendeleev import element
from typing import List, Union, Dict, Optional, Any, Tuple

from MatImba.utils import auto_featurise, mat_cost

# Attempt to import the model required for predict_compositions
# This assumes matools.py is in the same directory as evaluate.py (inside a package)
try:
    from ..models import gbr_ensemble
except ImportError:
    print("Warning: Could not import gbr_ensemble from ..models. predict_compositions may fail.")

def at_num_sorted(strlist: List[str]) -> List[str]:
    """
    Sorts elements in a list of chemical formula strings by their atomic number.

    Args:
        strlist (List[str]): A list of chemical formula strings (e.g., ['TiZr', 'HfTa']).

    Returns:
        List[str]: A list of strings where elements within each string are sorted by atomic number.
                   Hydrogen is excluded from the sorting.
    """
    return ["".join([str(e) for e in sorted([el for el in Composition(c).elements if str(el) !='H'], key=lambda x: x.number)]) 
               for c in strlist]

def formu_convert(formula: str) -> str:
    """
    Converts a chemical formula to its pretty string representation using Pymatgen.

    Args:
        formula (str): The input chemical formula.

    Returns:
        str: The formatted 'pretty' formula (e.g., 'Fe2O3').
    """
    return Composition(formula).to_pretty_string()


def calc_peq0(
    dS: Union[float, npt.ArrayLike], dH: Union[float, npt.ArrayLike],
    temperature: float = 298.15, logarithm: bool = False
) -> float:
    """
    Calculates the equilibrium pressure (peq) based on enthalpy and entropy.
    
    Args:
        dH (float): Enthalpy of formation (kJ/mol H2).
        dS (float): Entropy of formation (J/mol H2/K).
        temperature (float): Temperature in Celsius.

    Returns:
        float: The calculated equilibrium pressure (exp(lnpeq)).
    """
    lnpeq0 = (dS - 1000 * dH / temperature) / 8.314
    return lnpeq0 if logarithm else np.exp(lnpeq0)

def get_eles(formula: str) -> str:
    """
    Extracts all elements from a formula and joins them into a single string.

    Args:
        formula (str): Chemical formula.

    Returns:
        str: Concatenated element symbols (e.g., 'TiMn').
    """
    all_eles = list(Composition(formula).as_dict().keys())
    return ''.join(all_eles)

def max_mp(formula: str) -> float:
    """
    Finds the maximum melting point among the constituent elements of a formula.

    Args:
        formula (str): Chemical formula.

    Returns:
        float: The maximum melting point in Kelvin.
    """
    elements = list(Composition(formula).as_dict().keys())
    ele_mp = np.array([element(ele).melting_point for ele in elements])
    return ele_mp.max()

def max_mp_diff(formula: str) -> float:
    """
    Calculates the difference between the maximum and minimum melting points of the constituent elements.

    Args:
        formula (str): Chemical formula.

    Returns:
        float: The difference (Max MP - Min MP) in Kelvin.
    """
    elements = list(Composition(formula).as_dict().keys())
    ele_mp = np.array([element(ele).melting_point for ele in elements])
    return ele_mp.max() - ele_mp.min()

def wtfrac(formula: str, HtoM: float) -> float:
    """
    Calculates the Hydrogen weight fraction given a base formula and H/M ratio.

    Args:
        formula (str): Base chemical formula (without Hydrogen).
        HtoM (float): Hydrogen to Metal atomic ratio.

    Returns:
        float: Hydrogen weight percentage (0-100).
    """
    HtoM = 0 if HtoM < 0 else HtoM
    comp = Composition(formula)
    els = [el.symbol for el in comp.elements]
    # MWs = [el._atomic_mass for el in comp.elements] # Unused
    stoich = [comp.get_atomic_fraction(el) for el in comp.elements]
    assert np.isclose(np.sum(stoich), 1.0, rtol = 1e-05) # normalised so there is 1 metal atom
    newcomp = Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
    wt_frac = newcomp.get_wt_fraction('H') * 100
    return wt_frac

def HtoM2wtfrac(HtoM: float, comp: Composition) -> float:
    """
    Converts Hydrogen-to-Metal ratio to weight fraction for a Pymatgen composition.

    Args:
        HtoM (float): Hydrogen to Metal ratio.
        comp (Composition): Pymatgen Composition object (metal host).

    Returns:
        float: Hydrogen weight fraction.
    """
    els = [el.symbol for el in list(comp._data.keys())]
    # MWs = [el._atomic_mass for el in list(comp._data.keys())] # Unused
    stoich = [comp.get_atomic_fraction(el) for el in list(comp._data.keys())]
    assert np.isclose(np.sum(stoich), 1.0, rtol = 1e-05) # normalized so there is 1 metal atom
    newcomp = Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
    
    return newcomp.get_wt_fraction('H')


def wtfrac2HtoM(wtfrac: float, comp: Composition) -> float:
    """
    Convert Hydrogen weight fraction to H/M ratio for a given Pymatgen composition object.

    Args:
        wtfrac (float): Hydrogen weight fraction (0.0 to 1.0, check usage).
        comp (Composition): Pymatgen Composition object.

    Returns:
        float: Hydrogen to Metal (H/M) atomic ratio.
    """
    MWs = [el.atomic_mass for el in comp.elements] 
    stoich = [comp.get_atomic_fraction(el) for el in comp.elements]
    
    assert np.isclose(np.sum(stoich),1.0,rtol=1e-05)
    molefrac = wtfrac*np.sum(np.array(MWs)*np.array(stoich))/(1.008*(1-wtfrac))
    HtoM = molefrac/np.sum(stoich)
    return HtoM


def cweighted_elementalH_formE(comp: Composition, elem_table: pd.DataFrame) -> List[float]:
    """
    Calculates composition-weighted elemental hydride formation energy statistics.

    Args:
        comp (Composition): Pymatgen Composition object.
        elem_table (pd.DataFrame): DataFrame containing 'Species' and 'Ef' (Formation Energy) columns.

    Returns:
        List[float]: A list containing [Min Ef, Max Ef, Weighted Sum Ef, Std Dev Ef].
    """
    c = comp.as_dict()
    atlist = c.keys()-'H'
    tot = sum([c[key] for key in atlist])
    frac = [c[key]/tot for key in atlist]

    formElist = [float(elem_table.loc[elem_table['Species']==key]['Ef'])\
                 for key in atlist]
    formEcweighted = np.array(formElist)*frac
    
    return [min(formElist), max(formElist), 
            np.sum(formEcweighted), np.std(formEcweighted)]

def get_mat_cost(formula: str, mat_cost: Dict[str, float] = mat_cost) -> float:
    """
    Calculates the material cost for a specific formula based on weight fractions.

    Args:
        formula (str): Chemical formula.
        mat_cost (Dict[str, float]): Dictionary of element costs.

    Returns:
        float: The calculated material cost.
    """
    comp = Composition(formula)
    weight_dict = comp.to_weight_dict
    matertial_costs = np.sum([mat_cost[ele] * weight_dict[ele] for ele in weight_dict.keys()])
    return matertial_costs

def cost_and_wtfrac(df: pd.DataFrame, mat_cost: Dict[str, float] = mat_cost) -> pd.DataFrame:
    """
    Calculates the Hydrogen weight percentage and total material cost for a DataFrame of materials.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Formula' and 'HtoM' (Hydrogen to Metal ratio) columns.
        mat_cost (Dict[str, float], optional): Dictionary mapping element symbols to cost per unit weight. 
                                               Defaults to imported `mat_cost`.

    Returns:
        pd.DataFrame: The input DataFrame with added 'Hwtpercent' and 'matcost' columns.
    """
    for ind in df.index:
        formula, HtoM = df.loc[ind, ["Formula", "HtoM"]].values
        HtoM = 0 if HtoM < 0 else HtoM 
        comp = Composition(formula)
        els = [el.symbol for el in comp.elements]
        # MWs = [el._atomic_mass for el in comp.elements] # Unused variable
        stoich = [comp.get_atomic_fraction(el) for el in comp.elements]
        assert np.isclose(np.sum(stoich), 1.0, rtol = 1e-05) # normalised so there is 1 metal atom
        newcomp = Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
        wt_frac = newcomp.get_wt_fraction('H') * 100
        
        weight_dict = comp.to_weight_dict
        all_el_cost = [mat_cost[ele] * weight_dict[ele] for ele in weight_dict.keys()]
            
        matertial_costs = np.sum(all_el_cost)
        
        df.loc[ind, ["Hwtpercent", "matcost"]] = [wt_frac, matertial_costs]
    return df

def exclude_elements(dataframe: pd.DataFrame, element: str, formula_col: str = "formula", drop_index: bool = False) -> pd.DataFrame:
    """
    Filters a DataFrame to exclude rows containing a specific element in the formula.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        element (str): Element symbol to exclude (e.g., 'Cd').
        formula_col (str): Name of the column containing chemical formulas.
        drop_index (bool): Whether to reset and drop the old index.

    Returns:
        pd.DataFrame: A filtered DataFrame without the specified element.
    """
    mask = [element not in Composition(row).as_dict() for row in dataframe.loc[:, formula_col]]
    result_df = dataframe[mask].reset_index(drop = True) if drop_index else dataframe[mask]
    return result_df

def predict_compositions(formula: Union[str, List[str], pd.DataFrame], lds_models: bool = True) -> pd.DataFrame:
    """
    Runs ML predictions on given formulas using pre-trained Gradient Boosting ensembles.

    Args:
        formula (Union[str, List[str], pd.DataFrame]): Input formula(s). Can be a single string, list, or DataFrame.
        lds_models (bool): If True, loads LDS (Label Distribution Smoothing) models. If False, loads control models.

    Returns:
        pd.DataFrame: DataFrame containing the input formulas and predictions for properties 
                      (wt_pct, dH, HtoM, dS, lnpeq, slope).
    """
    # Local import to avoid circular dependency if models import matools
    from ..models import gbr_ensemble

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
    # Commented out blocks retained as per original request history
    # predictions = cost_and_wtfrac(predictions)
    return predictions