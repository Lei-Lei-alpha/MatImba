import os
import numpy as np
import pandas as pd
from .data import ele_ef
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer


def cweighted_elementalH_formE(comp, ele_ef = ele_ef):
    """
    comp : pymatgen.core.composition.Composition
    
    elem_table : DataFrame
        - contains one column with 'Species' element string and 'Ef' property column  
    """
    c = comp.as_dict()
    atlist = c.keys()-'H'
    tot = sum([c[key] for key in atlist])
    frac = [c[key]/tot for key in atlist]

    formElist = [float(ele_ef[key]) for key in atlist]
    formEcweighted = np.array(formElist)*frac
    
    return [min(formElist), max(formElist), 
            np.sum(formEcweighted), np.std(formEcweighted)]


def featurise(composition_df, composition_col = 'formula', elem_prop = True,
              irrelevant_features = None, filename = None, outdir = None):
    
    origcols = set(composition_df.columns)
    conversion_featurizer = StrToComposition(target_col_id = "composition_obj")
    conversion_featurizer.set_chunksize(5000)
    df = conversion_featurizer.featurize_dataframe(composition_df, composition_col)
    
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), 
                                          cf.ElementProperty.from_preset("magpie"),
                                          cf.IonProperty(fast=True)])
    
    feature_calculators.set_chunksize(1000)
    # feature_labels = feature_calculators.feature_labels()
    feature_df = feature_calculators.featurize_dataframe(df, col_id = "composition_obj")
    
    feature_df.columns = [col.replace('MagpieData','') for col in feature_df.columns]
    feature_df.columns = [col.replace('average','mu') for col in feature_df.columns]
    feature_df.columns = [col.replace('maximum','max') for col in feature_df.columns]
    feature_df.columns = [col.replace('minimum','min') for col in feature_df.columns]

    feature_df = feature_df.drop(list(origcols) + ['composition_obj'], axis=1, inplace = False)
    
    if elem_prop:
        elemH_formE = [cweighted_elementalH_formE(c) for c in df['composition_obj']]
        elemH_formE_df = pd.DataFrame(elemH_formE,columns=['E_HM min',
                                                           'E_HM max',
                                                           'E_HM mu',
                                                           'E_HM dev'])
        feature_df = pd.concat((feature_df, elemH_formE_df), axis=1, sort = False)
    
    feature_names = list(feature_df)
    feature_names = [feature_name.strip() for feature_name in feature_names]
    feature_df.columns = feature_names
    
    if irrelevant_features is not None:
        for feature in irrelevant_features:
            try:
                feature_df.drop(feature, axis = 1, inplace = True)
            except KeyError:
                print(f"Column {feature} not found, pass.")
                pass

    print("Shape of returned features: {0} rows, {1} columns".format(*feature_df.shape))
    
    if filename:
        if outdir is None:
            outdir = '.'
        
        dest_file = os.path.join(outdir, filename)
        feature_df.to_csv(dest_file, index = None)
        print(f"Feature data wrote to {dest_file}")
    
    return feature_df

def auto_featurise(formula, filename = None, outdir = None):
    delete_cols = [
        "mode GSbandgap", "mode NfValence", "min NpValence", "avg_dev GSbandgap",
        "mean GSbandgap", "range GSbandgap", "min NpUnfilled", "min NfValence",
        "mode NfUnfilled", "max GSbandgap", "mode NpUnfilled", "mode NpValence",
        "min NfUnfilled", "compound possible", "min GSbandgap", "min NsUnfilled",
        "max NsValence", "range NpUnfilled", "range NpValence", "min SpaceGroupNumber",
        "max NfUnfilled", "max NpValence", "max NpUnfilled", "mode NsUnfilled",
        "range NfValence", "min Row", "mean NfValence", "range NsUnfilled",
        "avg f valence electrons", "mean NfUnfilled", "max NfValence", "min Column",
        "max NUnfilled", "range NfUnfilled", "mode NsValence", "avg_dev NfValence",
        "mode Row", "min NValence", "max NsUnfilled", "avg_dev NpValence",
        "range NsValence", "avg_dev NfUnfilled", "max CovalentRadius"
    ]
    pred_fea = featurise(formula, composition_col = 'Formula', elem_prop = ele_ef,
                         irrelevant_features = delete_cols, filename = filename, outdir = outdir)
    return pred_fea