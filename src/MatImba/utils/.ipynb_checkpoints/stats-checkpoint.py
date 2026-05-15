import math 
import numpy as np

def plot_histogram(r: np.array, bins: int = 10) -> None:
    """ """
    
    figsize = (10,5)
    
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    
    plt.hist(r, bins = bins, label='histogram', alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', direction='in', length=8, width=1, labelsize=15)
    # plt.title(f'Histogram with {bins} bins', fontsize=15)
    # plt.savefig(f'./histogram_{bins}-bins.png')

def sturge_optimal_bins(data: np.array) -> int:
    """ Sturge's rule for optimal bin selection
    Parameters: 
        data (np.array) - a one-dimensional array with data
    Returns:
        nbins (int) - number of bins
    """
    assert data.ndim == 1
    n = data.size
    width = 1.0 + np.log2(n)
    
    nbins = math.ceil((data.max() - data.min()) / width)
    nbins = max(1, nbins)
    
    return nbins
    
def doane_optimal_bins(data: np.array) -> int:
    """ Doane's rule for optimal bin selection
    Parameters: 
        data (np.array) - a one-dimensional array with data
    Returns:
        nbins (int) - number of bins
    """
    assert data.ndim == 1
    assert data.size > 2
    assert np.std(data) > 0
    
    n = data.size
    # coefficient
    sg1 = np.sqrt(6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0)))
    # skewness
    skew = np.mean(((data - np.mean(data)) / np.std(data))**3)
    # skewness correction
    Ke = np.log2(1.0 + np.absolute(skew)/sg1)
    
    width = 1.0 + np.log2(n) + Ke
    
    nbins = math.ceil((data.max() - data.min()) / width)
    nbins = max(1, nbins)
    
    return nbins


def rice_optimal_bins(data: np.array) -> int:
    """ The Rice rule for optimal bin selection
    Parameters: 
        data (np.array) - a one-dimensional array with data
    Returns:
        nbins (int) - number of bins
    """
    assert data.ndim == 1
    n = data.size
    width = 2 * n**(1./3)
    
    nbins = math.ceil((data.max() - data.min()) / width)
    nbins = max(1, nbins)
    
    return nbins

def scott_optimal_bins(data: np.array) -> int:
    """ The Scott rule for optimal bin selection
    Parameters: 
        data (np.array) - a one-dimensional array with data
    Returns:
        nbins (int) - number of bins
    """
    assert data.ndim == 1
    n = data.size
    width = 3.49 * np.std(data)/n**(1./3)
    
    nbins = math.ceil((data.max() - data.min()) / width)
    nbins = max(1, nbins)
    
    return nbins

def fd_optimal_bins(data: np.array) -> int:
    """ The Freedman-Diaconis rule for optimal bin selection
    Parameters: 
        data (np.array) - a one-dimensional array with data
    Returns:
        nbins (int) - number of bins
    """
    assert data.ndim == 1
    n = data.size
    
    p25, p75 = np.percentile(data, [25, 75])

    width = 2. * (p75 - p25)/n**(1./3)
    nbins = math.ceil((data.max() - data.min()) / width)
    nbins = max(1, nbins)
    
    return nbins