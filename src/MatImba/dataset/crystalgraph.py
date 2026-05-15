# crystalgraph.py
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader  # Change to torch.utils.data
# from tqdm import tqdm if os.environ.get('_', '').endswith('jupyter') else tqdm_notebook
from .datautils import set_return
from .imba import get_weights, estimate_density, calc_relevance
from ..utils.struct2graph import (
    SimpleCrystalConverter,
    FlattenGaussianDistanceConverter,
    GaussianDistanceConverter,
    AtomFeaturesExtractor,
)
from torch_geometric.data import Batch  # Add this import
# Check for jupyter environment to use the correct tqdm
python_env = os.environ['_'].split('/')[-1]
if python_env == 'jupyter':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def seed_worker(worker_id):
    """
    Robust worker seeding to ensure data augmentations/loading
    are deterministic per worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class BaseImbalancedDataset:
    """
    Base class for handling imbalanced datasets, computing densities, weights, and relevances.
    Supports loading from single file (with splitting) or pre-split dict of files.
    """
    def __init__(self, datafile: str | dict, target_name: str | None = None,
                 val_size: float = 0.1, test_size: float = 0.1,
                 batch_size: int = 64, random_seed: int = 99, 
                 smooth: str = 'kde', device: torch.device | None = None,
                 reweight_method: str = 'log_inv', relevance_method: str = 'weights_eps0',
                 eps_weight: float = 0.6, eps_relevance: float = 1e-4) -> None:
        """
        Parameters:
        - datafile: Path to pickle or dict {'train': path, 'val': path, 'test': path}.
        - target_name: Target column (infers last column if None).
        - val_size: Validation split fraction (if single file or no val file).
        - test_size: Test split fraction (if single file).
        - batch_size: Batch size for DataLoaders.
        - random_seed: Seed for shuffling/splitting.
        - smooth: Density estimation method ('kde', 'convolve', None).
        - device: Torch device (auto if None).
        - reweight_method: Method for get_weights.
        - relevance_method: 'weights_eps0' (get_weights with eps=0) or 'boxplot' (calc_relevance).
        - eps_weight: Eps for weights.
        - eps_relevance: Eps for relevances (if boxplot).
        """
        self.datafile = datafile
        self.target_name = target_name
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.smooth = smooth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.pin_memory = (self.device.type == 'cuda')
        self.reweight_method = reweight_method
        self.relevance_method = relevance_method
        self.eps_weight = eps_weight
        self.eps_relevance = eps_relevance

    def _load_dataframes(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[pd.DataFrame]]:
        """
        Load dataframes from file or dict, collect for stats.
        """
        train_df = val_df = test_df = None
        all_dfs_for_stats = []
        split_required = False

        if isinstance(self.datafile, str):
            all_df = pd.read_pickle(self.datafile)
            split_required = True
            all_dfs_for_stats = [all_df]
        elif isinstance(self.datafile, dict):
            if "train" not in self.datafile:
                raise ValueError("datafile dict must contain 'train' key.")
            train_df = pd.read_pickle(self.datafile["train"])
            all_dfs_for_stats.append(train_df)
            if "val" in self.datafile:
                val_df = pd.read_pickle(self.datafile["val"])
                all_dfs_for_stats.append(val_df)
            if "test" in self.datafile:
                test_df = pd.read_pickle(self.datafile["test"])
                all_dfs_for_stats.append(test_df)
        else:
            raise TypeError("datafile must be str or dict.")

        return train_df, val_df, test_df, all_dfs_for_stats, split_required

    def _compute_global_stats(self, all_dfs_for_stats: list[pd.DataFrame], lds_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute densities, weights, relevances on concatenated labels.
        """
        if self.target_name is None:
            self.target_name = all_dfs_for_stats[0].columns[-1]

        all_labels = np.concatenate([df[self.target_name].values for df in all_dfs_for_stats])
        all_densities = estimate_density(all_labels, smooth=self.smooth, **lds_params)
        all_weights = get_weights(all_densities, method=self.reweight_method, eps=self.eps_weight)

        if self.relevance_method == 'boxplot':
            all_relevances = calc_relevance(all_labels, eps=self.eps_relevance, sort=False, plot=False)
        elif self.relevance_method == 'weights_eps0':
            all_relevances = get_weights(all_densities, method=self.reweight_method, eps=0.0)
        else:
            raise ValueError(f"Unknown relevance_method '{self.relevance_method}'.")

        return all_labels, all_weights, all_relevances, all_densities

    def _set_seeds(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.random_seed)

    def _split_single_file(self, all_df: pd.DataFrame, all_labels: np.ndarray, all_weights: np.ndarray,
                           all_relevances: np.ndarray, all_densities: np.ndarray) -> tuple[list, list, list]:
        all_data_list = list(zip(all_df['structure'], all_labels, all_weights, all_relevances, all_densities))
        self._set_seeds()
        random.shuffle(all_data_list)

        n_total = len(all_data_list)
        n_val = int(n_total * self.val_size)
        n_test = int(n_total * self.test_size)
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError("Training set size is zero or negative. Adjust split ratios.")

        train_data = all_data_list[:n_train]
        val_data = all_data_list[n_train:n_train + n_val]
        test_data = all_data_list[n_train + n_val:]

        return train_data, val_data, test_data

    def _assign_stats_to_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None, test_df: pd.DataFrame | None,
                                all_labels: np.ndarray, all_weights: np.ndarray, all_relevances: np.ndarray,
                                all_densities: np.ndarray) -> tuple[list, list, list]:
        n_train = len(train_df)
        n_val = len(val_df) if val_df is not None else 0
        n_test = len(test_df) if test_df is not None else 0

        idx_val_start = n_train
        idx_test_start = n_train + n_val

        train_labels = all_labels[:n_train]
        train_weights = all_weights[:n_train]
        train_relevances = all_relevances[:n_train]
        train_densities = all_densities[:n_train]

        val_labels = val_weights = val_relevances = val_densities = None
        if val_df is not None:
            val_labels = all_labels[idx_val_start:idx_test_start]
            val_weights = all_weights[idx_val_start:idx_test_start]
            val_relevances = all_relevances[idx_val_start:idx_test_start]
            val_densities = all_densities[idx_val_start:idx_test_start]

        test_labels = test_weights = test_relevances = test_densities = None
        if test_df is not None:
            test_labels = all_labels[idx_test_start:idx_test_start + n_test]
            test_weights = all_weights[idx_test_start:idx_test_start + n_test]
            test_relevances = all_relevances[idx_test_start:idx_test_start + n_test]
            test_densities = all_densities[idx_test_start:idx_test_start + n_test]

        train_data = list(zip(train_df['structure'], train_labels, train_weights, train_relevances, train_densities))
        val_data = list(zip(val_df['structure'], val_labels, val_weights, val_relevances, val_densities)) if val_df is not None else []
        test_data = list(zip(test_df['structure'], test_labels, test_weights, test_relevances, test_densities)) if test_df is not None else []

        return train_data, val_data, test_data

    def _split_val_from_train(self, train_data: list) -> tuple[list, list]:
        self._set_seeds()
        random.shuffle(train_data)
        n_val_split = int(len(train_data) * self.val_size)
        if n_val_split == 0 and self.val_size > 0 and len(train_data) > 0:
            n_val_split = 1
        n_train_split = len(train_data) - n_val_split
        if n_train_split <= 0:
            raise ValueError("Training set size zero after val split.")
        return train_data[:n_train_split], train_data[n_train_split:]


class CgcnnDataset(BaseImbalancedDataset):
    """
    Dataset for crystal structures, converting to PyG graphs with injected stats.
    """
    def __init__(self, datafile: str | dict, target_name: str | None = None,
                 bond_converter: GaussianDistanceConverter | FlattenGaussianDistanceConverter | None = None,
                 atom_converter: AtomFeaturesExtractor | None = None, cutoff: float = 5.0,
                 add_z_bond_coord: bool = False, **kwargs) -> None:
        super().__init__(datafile, target_name, **kwargs)
        self.converter = SimpleCrystalConverter(
            target_name=self.target_name or 'y',  # Placeholder, updated later
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=cutoff,
            add_z_bond_coord=add_z_bond_coord,
        )

    def _inject_stats(self, data_list: list) -> list:
        """
        Inject stats into structures using set_return.
        """
        injected = []
        for s, t, w, r, d in tqdm(data_list, desc="Injecting stats", leave=False):
            injected.append(set_return(s, ['y', 'omega', 'phi', 'rou'], [float(t), float(w), float(r), float(d)]))
        return injected

    def prepare_data(self, **lds_params) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """
        Prepare DataLoaders for train/val/test.

        Parameters:
        - **lds_params: Params for estimate_density (e.g., kernel, kernel_size).

        Returns:
        - trainloader, valloader (or None), testloader (or None).
        """
        train_df, val_df, test_df, all_dfs_for_stats, split_required = self._load_dataframes()
        all_labels, all_weights, all_relevances, all_densities = self._compute_global_stats(all_dfs_for_stats, lds_params)

        if split_required:
            train_data, val_data, test_data = self._split_single_file(train_df, all_labels, all_weights, all_relevances, all_densities)  # all_df is train_df here
            trainset = self._inject_stats(train_data)
            valset = self._inject_stats(val_data) if val_data else []
            testset = self._inject_stats(test_data) if test_data else []
        else:
            train_data, val_data, test_data = self._assign_stats_to_splits(train_df, val_df, test_df, all_labels, all_weights, all_relevances, all_densities)
            if val_df is None and self.val_size > 0:
                print(f"No val file. Splitting {self.val_size*100:.0f}% from train.")
                train_data, val_data = self._split_val_from_train(train_data)
            trainset = self._inject_stats(train_data)
            valset = self._inject_stats(val_data) if val_data else []
            testset = self._inject_stats(test_data) if test_data else []

        print(f"Converting Train Set... (Pin Memory: {self.pin_memory})")
        train_graphs = [self.converter.convert(s) for s in tqdm(trainset, desc="Converting", leave=False)]

        val_graphs = []
        if valset:
            print("Converting Val Set...")
            val_graphs = [self.converter.convert(s) for s in tqdm(valset, desc="Converting", leave=False)]

        test_graphs = []
        if testset:
            print("Converting Test Set...")
            test_graphs = [self.converter.convert(s) for s in tqdm(testset, desc="Converting", leave=False)]

        def collate_fn(batch):
            return Batch.from_data_list(batch)

        trainloader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True, 
                                 collate_fn=collate_fn, pin_memory=self.pin_memory)
        
        valloader = DataLoader(val_graphs, batch_size=self.batch_size, shuffle=False, 
                               collate_fn=collate_fn, pin_memory=self.pin_memory) if val_graphs else None
        
        testloader = DataLoader(test_graphs, batch_size=self.batch_size, shuffle=False, 
                                collate_fn=collate_fn, pin_memory=self.pin_memory) if test_graphs else None

        print("DataLoaders created.")
        return trainloader, valloader, testloader