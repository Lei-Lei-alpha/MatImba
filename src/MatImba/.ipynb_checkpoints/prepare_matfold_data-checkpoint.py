import os
import argparse
import logging
import pandas as pd
import json
import shutil
import numpy as np
from MatFold import MatFold
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def setup_logging(log_file_path):
    """Sets up logging to console and file."""
    # Reset handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. File Handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Saving log to: {log_file_path}")

def load_and_combine_data(data_dir, source_fold=0):
    """
    Loads train and test data from an existing fold to reconstruct the full dataset.
    """
    fold_dir = os.path.join(data_dir, f'fold_{source_fold}')
    train_path = os.path.join(fold_dir, 'train.pickle.gz')
    test_path = os.path.join(fold_dir, 'test.pickle.gz')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        error_msg = f"Could not find data in {fold_dir}. Ensure train.pickle.gz and test.pickle.gz exist."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info(f"Loading data from {fold_dir}...")
    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)
    
    # Combine
    df_full = pd.concat([df_train, df_test]).reset_index(drop=True)
    logger.info(f"Combined dataset size: {len(df_full)} samples")
    
    return df_full

def prepare_matfold_splits(args):
    # Ensure Output Directory Exists First
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup Logging inside the output directory
    log_path = os.path.join(args.out_dir, "preparation.log")
    setup_logging(log_path)
    
    logger.info("--- Starting MatFold Split Preparation ---")
    logger.info(f"Arguments: {vars(args)}")

    # 1. Load Data
    df = load_and_combine_data(args.data_dir, source_fold=0)
    
    # 2. Prepare for MatFold
    if 'structureid' not in df.columns:
        logger.info("Generating 'structureid' column...")
        df['structureid'] = [f"id_{i}" for i in range(len(df))]
    
    logger.info("Extracting structures for MatFold...")
    bulk_dict = {}
    valid_indices = []
    
    # Iterate with TQDM but don't log every step
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        try:
            struct = row['structure']
            bulk_dict[row['structureid']] = struct.as_dict()
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"Skipping index {idx} due to error: {e}")
            
    df = df.loc[valid_indices].reset_index(drop=True)
    logger.info(f"Valid structures retained: {len(df)}")
    
    # Proxy DataFrame
    df_for_matfold = df[['structureid']].copy()

    # 3. Initialize MatFold
    logger.info(f"Initializing MatFold with split type: {args.split_type}")
    mf = MatFold(
        df=df_for_matfold,
        bulk_dict=bulk_dict,
        seed=args.seed
    )
    
    # 4. Generate Splits
    temp_json_dir = os.path.join(args.out_dir, "matfold_outputs")
    if os.path.exists(temp_json_dir):
        shutil.rmtree(temp_json_dir)
    os.makedirs(temp_json_dir, exist_ok=True)
    
    logger.info("Generating splits via MatFold...")
    mf.create_nested_splits(
        split_type=args.split_type,
        n_outer_splits=args.n_folds,
        n_inner_splits=1, 
        output_dir=temp_json_dir,
        verbose=False # Turn off internal print to keep log clean
    )
    
    # 5. Robust File Parsing (CSV/JSON Mode)
    files = os.listdir(temp_json_dir)
    split_alias = None
    
    # Detect Alias
    for f in files:
        if f.startswith("mf.") and "_outer.train.csv" in f:
            parts = f.split('.')
            if len(parts) >= 5:
                split_alias = parts[1]
                break
    
    # Fallback for JSON mode (if MatFold version differs)
    is_json_mode = False
    if split_alias is None:
        # check for json files inside subfolders
        for root, dirs, f_files in os.walk(temp_json_dir):
            if "split_0.json" in f_files:
                is_json_mode = True
                temp_json_dir = root # update root to where files are
                logger.info(f"Detected JSON mode in {root}")
                break

    if split_alias is None and not is_json_mode:
        msg = f"CRITICAL ERROR: Could not determine MatFold output format. Files found: {files}"
        logger.error(msg)
        raise FileNotFoundError(msg)
        
    if split_alias:
        logger.info(f"Detected MatFold CSV alias: '{split_alias}'")

    # 6. Process Each Fold
    for fold in range(args.n_folds):
        train_ids, test_ids = [], []
        
        if not is_json_mode:
            # CSV Mode
            train_csv_name = f"mf.{split_alias}.k{fold}_outer.train.csv"
            test_csv_name = f"mf.{split_alias}.k{fold}_outer.test.csv"
            
            train_csv_path = os.path.join(temp_json_dir, train_csv_name)
            test_csv_path = os.path.join(temp_json_dir, test_csv_name)
            
            if not os.path.exists(train_csv_path):
                logger.error(f"Missing fold file: {train_csv_name}")
                raise FileNotFoundError(f"Missing fold file: {train_csv_name}")

            df_fold_train = pd.read_csv(train_csv_path)
            df_fold_test = pd.read_csv(test_csv_path)
            
            # Robust ID extraction
            col_name = 'structureid' if 'structureid' in df_fold_train.columns else df_fold_train.columns[0]
            train_ids = df_fold_train[col_name].values
            test_ids = df_fold_test[col_name].values
            
        else:
            # JSON Mode
            json_path = os.path.join(temp_json_dir, f"split_{fold}.json")
            with open(json_path, 'r') as f:
                split_data = json.load(f)
            train_ids = split_data['train']
            test_ids = split_data['test']

        # Filter ORIGINAL DataFrame
        df_train_final = df[df['structureid'].isin(train_ids)].copy()
        df_test_final = df[df['structureid'].isin(test_ids)].copy()
        
        # Save to Output Dir
        fold_out_dir = os.path.join(args.out_dir, f"fold_{fold}")
        os.makedirs(fold_out_dir, exist_ok=True)
        
        train_out = os.path.join(fold_out_dir, "train.pickle.gz")
        test_out = os.path.join(fold_out_dir, "test.pickle.gz")
        
        df_train_final.to_pickle(train_out, compression='gzip')
        df_test_final.to_pickle(test_out, compression='gzip')
        
        logger.info(f"Saved Fold {fold}: Train={len(df_train_final)}, Test={len(df_test_final)} -> {fold_out_dir}")

    logger.info(f"--- SUCCESS: All folds saved to {args.out_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate MatFold splits for MEGNet training.")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to original data directory containing fold_0/train.pickle.gz")
    
    parser.add_argument("--out_dir", type=str, required=True, 
                        help="Path to output directory where new folds will be saved.")
    
    parser.add_argument("--split_type", type=str, default="spacegroup", 
                        choices=['index', 'elements', 'spacegroup', 'composition', 'crystalsys'],
                        help="MatFold split strategy.")
    
    parser.add_argument("--n_folds", type=int, default=5, 
                        help="Number of cross-validation folds.")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for MatFold.")

    args = parser.parse_args()
    
    # Run
    prepare_matfold_splits(args)