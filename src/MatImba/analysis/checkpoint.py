"""Checkpoint re-evaluation (the only model-coupled part of the analysis
layer).

Rebuilds the MEGNet training environment from an experiment YAML config and a
saved checkpoint so its test-set predictions can be regenerated.  Torch and
model imports happen lazily so the rest of :mod:`MatImba.analysis` stays
importable without a working GNN stack.
"""

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def seed_worker(worker_id):
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed: int):
    """Sets seeds for all relevant libraries to ensure reproducibility.
    Uses warn_only=True to prevent crashes on non-deterministic GNN ops."""
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def evaluate_ckpt(ckpt_path: str, config_file: str, fold: int = 0, run_id: int = 0,
                  data_loc_override: Optional[str] = None):
    """Loads a model from a checkpoint and recreates the Trainer environment
    for evaluation.

    Args:
        ckpt_path: Path to the .pth.tar checkpoint file.
        config_file: Path to the experiment YAML config.
        fold: Fold index for data loading.
        run_id: Run index (offsets the data seed).
        data_loc_override: Override data location from config.

    Returns:
        CgcnnTrainer: trainer with the loaded model, ready for ``.predict()``.
    """
    import torch
    import yaml

    from ..dataset import CgcnnDataset
    from ..models import MEGNet
    from ..trainer import CgcnnTrainer
    from ..utils import (
        AtomFeaturesExtractor,
        FlattenGaussianDistanceConverter,
        GaussianDistanceConverter,
    )
    from ..utils.evaluate import get_obj

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    logger.info("Loading configuration from %s...", config_file)
    with open(config_file) as config:
        expt_config = yaml.full_load(config)

    # --- 1. Reconstruct Converters ---
    cutoff = expt_config["data"]["cutoff"]
    edge_embed_size = expt_config["data"]["edge_embed_size"]
    if expt_config["data"]["add_z_bond_coord"]:
        bond_converter = FlattenGaussianDistanceConverter(
            centers=np.linspace(0, cutoff, edge_embed_size)
        )
    else:
        bond_converter = GaussianDistanceConverter(
            centers=np.linspace(0, cutoff, edge_embed_size)
        )
    atom_converter = AtomFeaturesExtractor(expt_config["data"]["atom_features"])
    target_name = expt_config["data"]["target_name"]

    # --- 2. Prepare Data ---
    model_name_dir = f"fold_{fold}"
    base_data_loc = data_loc_override if data_loc_override else expt_config["data"]["data_loc"]
    datafiles = {
        "train": os.path.join(base_data_loc, model_name_dir, "train.pickle.gz"),
        "test": os.path.join(base_data_loc, model_name_dir, "test.pickle.gz"),
    }
    if not os.path.exists(datafiles["test"]):
        logger.warning("Test data not found at %s. Dataloaders might be empty.",
                       datafiles["test"])

    seed = expt_config["data"]["seed"] + run_id
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    data_set_creator = CgcnnDataset(
        datafile=datafiles, target_name=target_name,
        bond_converter=bond_converter, atom_converter=atom_converter,
        random_seed=seed,
    )
    train_loader, val_loader, test_loader = data_set_creator.prepare_data(
        reweight=expt_config["data"].get("reweight", "log_inv"),
        generator=g, worker_init_fn=seed_worker,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3. Initialize Model ---
    model = MEGNet(
        edge_input_shape=bond_converter.get_shape(),
        node_input_shape=atom_converter.get_shape(),
        state_input_shape=expt_config["model"]["state_input_shape"],
        device=device,
    )

    # --- 4. Load Weights ---
    logger.info("Loading checkpoint weights from %s...", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model" in checkpoint and "states" in checkpoint["model"]:
        model.load_state_dict(checkpoint["model"]["states"], strict=False)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # --- 5. Initialize Trainer ---
    loss_func = get_obj(expt_config["loss"]["loss"])()
    return CgcnnTrainer(
        model=model,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        loss_func=loss_func,
        name=f"eval_fold_{fold}",
        epoch_range=0,
        weighted_loss=expt_config["train"]["weighted_loss"],
        dil_inform=expt_config["train"]["dil_inform"],
        outdir=os.path.join(expt_config["save"]["basedir"], expt_config["save"]["outdir"]),
    )
