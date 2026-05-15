import os
import torch
import joblib
import shutil
import random
import numpy as np
import pandas as pd
from sklearn import linear_model
import pymatgen.core as pym_core
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    
def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
            


def set_return(o, names, vals):
    if isinstance (names, str):
        setattr(o, names, vals)
    else:
        [setattr(o, name, val) for name, val in zip(names, vals)]
    return o


def save_checkpoint(state, is_best, outdir, prefix=''):
    filename = os.path.join(outdir, f"{prefix}ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        print("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

            
def at_num_sorted(strlist):
    """
    strlist: list of chemical strings ['TiZrMnCrFeNi','HfZr']
    """
    return ["".join([str(e) for e in sorted([el for el in pym_core.Composition(c).elements if str(el) !='H'], key=lambda x: x.number)]) 
               for c in strlist]

def sk_linear_model(X,y):
    inds = np.argsort(X,axis=0).squeeze()
    fit = linear_model.LinearRegression().fit(X,y)
    pred = fit.predict(X)
    r2 = r2_score(pred,y)
    return fit, r2, pred, inds


def filter_by_predict_value(limlower, limupper, y, holdlower = True, holdupper = True):
    """
    Filters out data in y above and below critical values and optionally
    stores them as holdout data
    """

    if limlower is not None and limupper is None:
        holdout_indices = np.where(y<limlower)
        keep_indices    = np.where(y>limlower)
    elif limlower is None and limupper is not None:
        holdout_indices = np.where(y>limupper)
        keep_indices    = np.where(y<limupper)
    elif limlower is not None and limupper is not None:
        if holdlower and holdupper:
            holdout_indices = np.where((y<limlower) | (y>limupper))[0]
        elif holdlower and not holdupper:
            holdout_indices = np.where(y<limlower)
        elif not holdlower and holdupper:
            holdout_indices = np.where(y>limupper)
        else:
            holdout_indices = np.array([],dtype=int)

        keep_indices    = np.where((y>limlower) & (y<limupper))[0]
    else:
        keep_indices = slice(0,len(y))
        holdout_indices = np.array([],dtype=int)

    return keep_indices, holdout_indices

def compute_mean_relative_error(y_true, y_pred):
    return np.average(np.abs((y_pred-y_true)/y_true)*100)

def formu_convert(formula):
    return pym_core.Composition(formula).to_pretty_string()

def HtoM2wtfrac(HtoM, comp):
    els = [el.symbol for el in list(comp._data.keys())]
    MWs = [el._atomic_mass for el in list(comp._data.keys())]
    stoich = [comp.get_atomic_fraction(el) for el in list(comp._data.keys())]
    assert np.isclose(np.sum(stoich), 1.0, rtol=1e-05) # normalized so there is 1 metal atom
    newcomp = pym_core.Composition("".join(['%s%.10f'%(el, amt) for el, amt in zip(els, stoich)] + ['H%.10f'%HtoM]))
    
    return newcomp.get_wt_fraction('H')

def wtfrac2HtoM(wtfrac, comp):
    """
    Convert H/M ratio to a wt fraction for a given Pymatgen composition object
    """
    MWs = [pym_core.Element(el).atomic_mass for el in comp.as_dict().keys()]
    stoich = [comp.get_atomic_fraction(el) for el in comp.as_dict().keys()]
    #print(stoich, np.sum(stoich))
    assert np.isclose(np.sum(stoich),1.0,rtol=1e-05)
    molefrac = wtfrac*np.sum(np.array(MWs)*np.array(stoich))/(1.008*(1-wtfrac))
    HtoM = molefrac/np.sum(stoich)
    return HtoM

def cweighted_elementalH_formE(comp, elem_table):
    """
    comp : pymatgen.core.composition.Composition
    
    elem_table : DataFrame
        - contains one column with 'Species' element string and 'Ef' property column  
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

def internally_studentised_residual(X, Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    n = len(X)
    diff_mean_sqr = np.dot((X - mean_X), (X - mean_X))
    beta1 = np.dot((X - mean_X), (Y - mean_Y)) / diff_mean_sqr
    beta0 = mean_Y - beta1 * mean_X
    y_hat = beta0 + beta1 * X
    residuals = Y - y_hat
    h_ii = (X - mean_X) ** 2 / diff_mean_sqr + (1 / n)
    Var_e = np.sqrt(sum((Y - y_hat) ** 2)/(n-2))
    SE_regression = Var_e*((1-h_ii) ** 0.5)
    studentized_residuals = residuals/SE_regression
    return studentized_residuals

def featurise(composition_data, composition_col = 'formula', elem_prop = None, irrelevant_features = None, filename = None, outdir = None):
    origcols = set(composition_data.columns)
    conversion_featurizer = StrToComposition(target_col_id = "composition_obj")
    conversion_featurizer.set_chunksize(5000)
    df = conversion_featurizer.featurize_dataframe(composition_data, composition_col)
    
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), 
                                          cf.ElementProperty.from_preset("magpie"),
                                          cf.ValenceOrbital(props=['avg']), 
                                          cf.IonProperty(fast=True)])
    
    feature_calculators.set_chunksize(1000)
    # feature_labels = feature_calculators.feature_labels()
    feature_df = feature_calculators.featurize_dataframe(df, col_id = "composition_obj")
    
    feature_df.columns = [col.replace('MagpieData','') for col in feature_df.columns]
    feature_df.columns = [col.replace('average','mu') for col in feature_df.columns]
    feature_df.columns = [col.replace('maximum','max') for col in feature_df.columns]
    feature_df.columns = [col.replace('minimum','min') for col in feature_df.columns]

    feature_df = feature_df.drop(list(origcols) + ['composition_obj'], axis=1, inplace = False)
    
    if elem_prop is not None:
        elemH_formE = [cweighted_elementalH_formE(c,elem_prop)\
                   for c in df['composition_obj']]
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
        feature_df.to_csv(dest_file)
        print(f"Feature data wrote to {dest_file}")
    
    return feature_df
    