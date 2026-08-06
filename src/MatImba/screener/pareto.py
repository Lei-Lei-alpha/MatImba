import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition

# Define Types
TargetMap = Dict[str, float]
CriteriaMap = Dict[str, float]

def get_eles(formula: str) -> str:
    """Extracts elements from a formula string."""
    all_eles = list(Composition(formula).as_dict().keys())
    return "".join(all_eles)

def exclude_elements(
    dataframe: pd.DataFrame, 
    element: str, 
    formula_col: str = "formula", 
    drop_index: bool = False
) -> pd.DataFrame:
    """Filters out rows containing a specific element."""
    mask = [element not in Composition(row).as_dict() for row in dataframe[formula_col]]
    result_df = dataframe[mask]
    if drop_index:
        return result_df.reset_index(drop=True)
    return result_df

MATH_NAMES = {
    "dH": r"$\Delta H$",
    "HtoM": "H/M",
    "Hwtpercent": "wt.%",
    "slope": r"$\frac{d\ln (P / P_0)}{dx}$",
    "lnpeq": r"$\ln (\frac{ P_{eq}}{P_{0}})$",
    "dS": r"$\Delta S$",
    "matcost": "Price",
}

UNITS = {
    "dH": r"[kJ mol$^{-1}$ H$_2$]",
    "HtoM": "",
    "Hwtpercent": "",
    "slope": "",
    "lnpeq": "",
    "dS": r"[J mol$^{−1}$ H$_2$ K$^{−1}$]",
    "matcost": r"[USD kg$^{-1}$]",
}

class ParetoOptimal:
    """
    Analyzes Pareto efficiency for materials screening.
    """
    def __init__(
        self,
        dataframe: Union[pd.DataFrame, str],
        label: Optional[str] = None,
        pareto_targets: Optional[TargetMap] = None,
        outdir: str = "results",
    ):
        if isinstance(dataframe, str):
            self.dataframe = pd.read_csv(dataframe, index_col=0)
        else:
            self.dataframe = dataframe.copy()
        # Ensure ele_comp exists
        if "ele_comp" not in self.dataframe.columns and "Formula" in self.dataframe.columns:
            self.dataframe["ele_comp"] = self.dataframe["Formula"].apply(get_eles)
        self.pareto_targets = pareto_targets or {
            "dH": 18, "HtoM": -1, "Hwtpercent": -1, "dS": -1,
            "lnpeq": -1, "slope": 1, "matcost": 1
        }
        self.label = label
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def group_optimal(self, df_group: Optional[pd.core.groupby.DataFrameGroupBy] = None) -> pd.DataFrame:
        """
        Finds Pareto optimal points within each group (e.g., composition system).
        Optimized to avoid repeated dataframe copying.
        """
        if df_group is None:
            if "ele_comp" not in self.dataframe.columns:
                raise ValueError("Column 'ele_comp' missing for grouping.")
            df_group = self.dataframe.groupby("ele_comp")

        results_dfs = []
        for _, group_df in df_group:
            # Use the optimized prepare_data on the slice
            data = self.prepare_data(group_df)
            efficient_mask, _ = self.is_pareto_efficient(data)
            
            # Extract original indices of efficient points
            # data[:, 0] contains the index values from prepare_data
            efficient_indices = data[efficient_mask][:, 0]
            results_dfs.append(group_df.loc[efficient_indices])

        if not results_dfs:
            return pd.DataFrame()
            
        return pd.concat(results_dfs)

    def prepare_data(self, dataframe: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prepares the cost matrix for Pareto analysis.
        Minimization is assumed: targets with -1/1 multipliers are adjusted so lower is better.
        Differences (|val - target|) are calculated for specific targets.
        """
        df = self.dataframe if dataframe is None else dataframe
        
        # Collect columns first to build array efficiently
        data_list = [df.index.values] # Store index in column 0 to track back
        
        for key, value in self.pareto_targets.items():
            if key not in df.columns:
                continue
                
            col_values = df[key].values
            
            # Transform to minimization problem
            if value == 1:  # Minimize (e.g., cost)
                data_list.append(col_values)
            elif value == -1: # Maximize -> Minimize negative
                data_list.append(-col_values)
            else: # Target value -> Minimize absolute difference
                data_list.append(np.abs(col_values - value))
                
        return np.vstack(data_list).T
    
    def is_pareto_efficient(self, data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the pareto-efficient points in the provided data matrix.
        Assumes the first column (index 0) is an identifier/index and ignores it for comparison.
        """
        # Use self.prepare_data() if no data provided
        costs = self.prepare_data() if data is None else data
        
        # Slice off the index column [:, 1:] for actual cost comparison
        # costs_val shape: (n_samples, n_metrics)
        costs_val = costs[:, 1:] 
        
        is_efficient = np.arange(costs_val.shape[0])
        n_points = costs_val.shape[0]
        next_point_index = 0
        
        while next_point_index < len(costs_val):
            # Compare current point against all remaining points
            # mask: True if point i is strictly better than point 'next_point_index' in all dimensions
            nondominated_point_mask = np.any(costs_val < costs_val[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            
            # Filter efficient array
            is_efficient = is_efficient[nondominated_point_mask]
            costs_val = costs_val[nondominated_point_mask]
            
            # Update index to check next remaining point
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        
        return is_efficient_mask, is_efficient
    
    def screen_compound(
        self, 
        dataframe: Optional[pd.DataFrame] = None, 
        criteria: Optional[CriteriaMap] = None, 
        short: bool = True, 
        savename: str = "pareto_comp"
    ) -> pd.DataFrame:
        
        df = self.dataframe if dataframe is None else dataframe.copy()
        criteria = criteria or {"lnpeq": 1.8}

        if short:
            # Keep only the entry with minimum dH per composition
            df = df.loc[df.groupby("ele_comp")["dH"].idxmin()]

        # Find Pareto optimal points
        data = self.prepare_data(df)
        efficient_mask, _ = self.is_pareto_efficient(data)
        
        # Retrieve original rows
        efficient_indices = data[efficient_mask][:, 0]
        pareto_eff_alloys = df.loc[efficient_indices]

        # Apply additional criteria
        for key, value in criteria.items():
            if key in pareto_eff_alloys.columns:
                pareto_eff_alloys = pareto_eff_alloys[pareto_eff_alloys[key] >= value]

        if savename:
            path = os.path.join(self.outdir, f"{savename}.csv")
            pareto_eff_alloys.to_csv(path)
            
        return pareto_eff_alloys
    

def plot_optimal(
    optm_df: pd.DataFrame,
    dataframe: Optional[pd.DataFrame] = None,
    pareto_analyser: Optional[ParetoOptimal] = None,
    properties: List[str] = ["slope", "dH"],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4.5, 3.1),
    seed: int = 0,
    cmap: str = "inferno",
    math_names: Optional[Dict[str, str]] = None,
    units: Optional[Dict[str, str]] = None,
    plot_frontier: bool = False,
    highlight: bool = True,
    highlight_color: str = "#01665e",
    hide_axis_labels: bool = False,
    filename: Optional[str] = None,
) -> None:
    """
    Plots the screening landscape with Pareto frontier.
    """
    assert dataframe is not None or pareto_analyser is not None, "Provide dataframe or analyser."
    
    if dataframe is None:
        # We asserted pareto_analyser is not None, so this is safe
        dataframe = pareto_analyser.dataframe # type: ignore 
    if pareto_analyser is None:
        pareto_analyser = ParetoOptimal(dataframe)

    # Use defaults if not provided
    math_names = math_names or MATH_NAMES
    units = units or UNITS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # Prepare background data (hexbin)
    inds = dataframe.index.values.copy()
    np.random.seed(seed)
    np.random.shuffle(inds)
    
    plot_data = []
    axis_labels = []
    
    for prop in properties:
        target = pareto_analyser.pareto_targets.get(prop, -1)
        original_vals = dataframe[prop].values
        shuffled_vals = np.take(original_vals, inds)
        
        unit_str = units.get(prop, "")
        name_str = math_names.get(prop, prop)

        if target in [-1, 1]:
            plot_data.append(shuffled_vals)
            label = f"{name_str} {unit_str}".strip()
        else:
            # Distance to target
            plot_data.append(np.abs(shuffled_vals - target))
            label = f"|{name_str} - {target}| {unit_str}".strip()
            
        axis_labels.append(label)

    # Hexbin Plot
    sc = ax.hexbin(
        plot_data[0], 
        np.abs(plot_data[1]),
        cmap=cmap, 
        mincnt=1, 
        gridsize=25, 
        bins="log"
    )
    
    cb = plt.colorbar(sc, ax=ax)
    cb.ax.set_title("#C", fontsize=10)

    # Prepare Frontier Data
    frontier_data = []
    for prop in properties:
        target = pareto_analyser.pareto_targets.get(prop, -1)
        vals = optm_df[prop].values
        
        if target in [-1, 1]:
            frontier_data.append(vals)
        else:
            frontier_data.append(np.abs(vals - target))
            
    # Sort by x-axis property for cleaner line plotting
    if len(frontier_data) >= 2:
        sort_ind = np.argsort(frontier_data[0])
        x_sorted = frontier_data[0][sort_ind]
        y_sorted = frontier_data[1][sort_ind]

        if plot_frontier:
            ax.plot(x_sorted, y_sorted, linestyle="-", linewidth=1.5, c="#4d4d4d", alpha=0.75)

        if highlight:
            ax.scatter(
                x_sorted, y_sorted,
                marker="o", c="#f5f5f5", edgecolor=highlight_color, s=50,
                linewidths=1.5, alpha=0.6, zorder=1000, label="Pareto optimal"
            )
            ax.legend(frameon=False, loc="upper left")

    if not hide_axis_labels:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    if filename:
        plt.savefig(filename, dpi=600)