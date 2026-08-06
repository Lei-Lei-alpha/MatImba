import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.use_deterministic_algorithms(True, warn_only=True)

import argparse

from MatImba.analysis import PredictionSet as ml_pred
from MatImba.analysis import evaluate_ckpt

DATASETS = ['phonons', 'log_kvrh', 'log_gvrh', 'perovskites']

# suffix → (experiment subdir, config suffix)
# config: expt_configs/final/{dataset}.yaml           (control)
#         expt_configs/final/{dataset}_{suffix}.yaml  (others)
METHODS = {
    'control':    'control',
    'dir':        'dir',
    'bsam':       'bsam',
    'smooth_dila': 'smooth_dila',
}

# Checkpoint metric suffix:  '' → best (MAE), 'dil_' → robust, 'sera_' → SERA, 'r2_score_' → R2
METRIC_SUFFIX = {
    'mae':       '',
    'dil':       'dil_',
    'sera':      'sera_',
    'r2':        'r2_score_',
}


def compute_and_save(dataset, suffix, fold, run_id, metric, force, expt):
    """
    Loads a checkpoint, runs prediction on the test set, builds an ml_pred
    object (which pre-computes SERA, SER curve, binned MAE, etc.) and saves
    it as a compressed .npz file alongside the checkpoint.
    """
    run_dir    = os.path.join('experiments', expt, dataset, suffix)
    prefix     = f'fold_{fold}_run{run_id}'
    ckpt_path  = os.path.join(run_dir, f'{prefix}.ckpt.{METRIC_SUFFIX[metric]}best.pth.tar')
    cache_path = os.path.join(run_dir, f'{prefix}_ml_pred.npz')

    if suffix == 'control':
        config_file = os.path.join('expt_configs', expt, f'{dataset}.yaml')
    else:
        config_file = os.path.join('expt_configs', expt, f'{dataset}_{suffix}.yaml')

    # --- Skip if cached and not forcing ---
    if os.path.exists(cache_path) and not force:
        try:
            pred = ml_pred.load(cache_path)
            print(f'  [SKIP] {prefix} ({suffix}) — cached  R2={pred.r2_score:.4f}  SERA={pred.sera:.4f}')
            return
        except Exception as e:
            print(f'  [WARN] {cache_path} corrupt ({e}), recomputing.')

    # --- Check checkpoint ---
    if not os.path.exists(ckpt_path):
        print(f'  [MISS] {ckpt_path} — no checkpoint, skipping.')
        return

    # --- Evaluate ---
    print(f'  [EVAL] {prefix} ({suffix}, metric={metric}) …')
    try:
        trainer = evaluate_ckpt(
            ckpt_path=ckpt_path,
            config_file=config_file,
            fold=fold,
            run_id=run_id,
        )
        targets, preds, relevances, densities = trainer.predict(trainer.test_loader)
        pred = ml_pred(targets, preds, relevances, densities)
        pred.save(cache_path)
        print(f'  [DONE] {prefix} ({suffix})  R2={pred.r2_score:.4f}  SERA={pred.sera:.4f}  → {cache_path}')
    except Exception as e:
        print(f'  [ERR]  {prefix} ({suffix}): {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate ml_pred .npz files from trained checkpoints.'
    )
    parser.add_argument('--datasets',  nargs='+', default=DATASETS,
                        help='Dataset names (default: all four)')
    parser.add_argument('--methods',   nargs='+', default=list(METHODS.keys()),
                        choices=list(METHODS.keys()),
                        help='Methods to evaluate (default: all)')
    parser.add_argument('--folds',     nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--runs',      nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--metric',    default='dil', choices=list(METRIC_SUFFIX.keys()),
                        help='Checkpoint selection metric (default: dil)')
    parser.add_argument('--expt',      default='final',
                        help='Experiment folder under experiments/ and expt_configs/ (default: final)')
    parser.add_argument('--force',     action='store_true',
                        help='Recompute and overwrite existing .npz files')
    args = parser.parse_args()

    print(f'Experiment: {args.expt}')
    for dataset in args.datasets:
        print(f'\n{"="*60}\nDataset: {dataset}\n{"="*60}')
        for method in args.methods:
            suffix = METHODS[method]
            print(f'\n  Method: {method}')
            for fold in args.folds:
                for run_id in args.runs:
                    compute_and_save(dataset, suffix, fold, run_id, args.metric, args.force, args.expt)


if __name__ == '__main__':
    main()
