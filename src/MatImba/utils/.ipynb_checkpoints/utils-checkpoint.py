import os
import torch
import shutil
import random
import logging
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# Initialize logger for this module (inherits config from run_trainer.py)
logger = logging.getLogger(__name__)

def save_checkpoint(
    state, is_best, outdir, prefix = '', is_dil_aware_best = None,
    is_sera_best = None, is_r2_best = None
):
    filename = os.path.join(outdir, f"{prefix}.ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if is_dil_aware_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'dil_best.pth.tar'))
    if is_sera_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'sera_best.pth.tar'))
    if is_r2_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'r2_score_best.pth.tar'))


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """
    Displays progress using the unified logging system.
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        content = '\t'.join(entries)
        logger.info(content)

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

