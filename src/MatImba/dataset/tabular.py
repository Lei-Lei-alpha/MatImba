import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .imba import get_weights, estimate_density
from torch.utils.data import DataLoader, TensorDataset, random_split

class TabDataset():
    def __init__(self, features_file: str, labels_file: str, target_col: str = None,
                 test_size: float = 0.15, batch_size: int = 32, data_range: "list | tuple" = None,
                 random_seed: int = 99, smooth: str = "convolve") -> None:
        
        assert smooth in {"kde", "convolve"}
        
        self.features_file = features_file
        self.labels_file = labels_file
        self.target_col = target_col
        self.test_size = test_size
        self.batch_size = batch_size
        self.data_range = data_range
        self.random_seed = random_seed
        self.smooth = smooth

    def _read_data_file(self):
        train_fea = pd.read_csv(self.features_file)
        train_df = pd.read_csv(self.labels_file)
        
        if self.data_range is not None:
            train_fea = train_fea[train_df[self.target_col].between(*self.data_range)]
            train_df = train_df[train_df[self.target_col].between(*self.data_range)]
        
        # Extract input & outupts as numpy arrays
        not_na_idx = train_df[self.target_col].notnull()
        
        inputs_array = train_fea[not_na_idx].values
        targets_array = train_df[not_na_idx][self.target_col].values
        
        return inputs_array, targets_array
           

    def prepare_data(self, **lds_params) -> tuple[DataLoader, DataLoader]:
        inputs_array, targets_array = self._read_data_file()
        inputs = torch.from_numpy(inputs_array.astype(float))
        
        label_densities = estimate_density(targets_array, smooth = self.smooth, **lds_params)
        label_densities = torch.tensor(label_densities).unsqueeze(-1)
        targets = torch.tensor(targets_array).unsqueeze(-1)
        
        weights = get_weights(targets_array, eps = 0.5)
        weights = torch.tensor(weights).unsqueeze(-1)
        dataset = TensorDataset(inputs, targets, weights, label_densities)

        total_data = len(targets_array)
        lengths = [total_data - int(self.test_size * total_data), int(self.test_size * total_data)]

        train_ds, val_ds = random_split(dataset, lengths, generator = torch.Generator().manual_seed(self.random_seed))
        train_loader = DataLoader(train_ds, self.batch_size, num_workers = 0, shuffle = True)
        val_loader = DataLoader(val_ds, self.batch_size, num_workers = 0, shuffle = True)
        return train_loader, val_loader

    # def kfold_data(self, smooth = "convolve"):
        