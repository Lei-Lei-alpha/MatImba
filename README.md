# MatImba: Distribution Imbalance-Aware Materials Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: CC0 1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

Official implementation of **"Unbiasing Materials Discovery: A Distribution Imbalance-Aware Framework for Robust Regression"**.

Materials property datasets are almost always imbalanced: predictive capacity concentrates on the dense bulk of the target distribution, and errors spike exactly where high-value materials live — in the sparse tails. MatImba is a package to work with that reality end-to-end:

1. **Describe** — quantify a dataset's distribution imbalance before training (`h`, Gini, `D_KL`, `W₁`; density ρ and tail relevance φ).
2. **Train** — a model-agnostic robust trainer with tail-error-resistant checkpoint selection and the low-cost **DILA** regulariser (Distance Correlation penalty between residuals and label density).
3. **Analyse** — from out-of-distribution error decomposition to real screening performance, including the **SERA–α coupling diagnosis** that tells you whether tail error is trainable away at all or fixed by the dataset geometry.

---

## 🛠️ Installation

Requires Python 3.10–3.12 (pinned by `torch_geometric`); a CUDA GPU is recommended for GNN training.

```bash
git clone https://github.com/Lei-Lei-alpha/MatImba.git
cd MatImba
pip install -e .
```

---

## 📖 The three-stage workflow

### 1. Describe: quantify dataset imbalance

```python
import pandas as pd
from MatImba.analysis import DatasetProfile, compare_profiles

y = pd.read_csv("matbench_data/log_gvrh.csv").iloc[:, 0].values
profile = DatasetProfile(y, name="log_gvrh")
print(profile.metrics())
# {'name': 'log_gvrh', 'n': 10987, 'h': 0.498, 'Gini': ..., 'D_KL': ..., 'W1': ..., 'tail_fraction': ...}
profile.plot()  # distribution + relevance phi overlay

# Compare several datasets in imbalance-metric space (PCA biplot)
result = compare_profiles([profile, other_profile, ...])
print(result["table"])
```

### 2. Train: robust training with any PyTorch model

The generic training loop — DILA loss, EMA-smoothed robust model selection, five checkpoint flavors, per-epoch metric logging — lives in `BaseRobustTrainer`. To use it with your own model, implement two hooks:

```python
from MatImba.base_trainer import BaseRobustTrainer, BatchFields
from MatImba.dataset.imba import estimate_density, get_weights, calc_relevance

class MyTrainer(BaseRobustTrainer):
    def forward_batch(self, batch):
        x, *_ = batch
        return self.model(x)

    def unpack_batch(self, batch):
        x, y, omega, rou, phi = batch
        return BatchFields(y=y, weights=omega, density=rou, relevance=phi)

trainer = MyTrainer(model=model, train_loader=train_loader, val_loader=val_loader,
                    dil_inform=True, dil_config={"lambda": 1.0, "base_metric": "huber"})
trainer.fit()
```

`density` (ρ) and `relevance` (φ) come from `estimate_density` / `calc_relevance` on your training labels — see the complete runnable example in [`examples/custom_trainer_example.py`](examples/custom_trainer_example.py) (plain MLP, CPU, under a minute).

DILA adds essentially no runtime or memory over the control objective (unlike sharpness-aware training), which makes it the default mitigation to try when Stage 3 shows your dataset's tail error is awareness-coupled.

For the paper's MEGNet graph-network pipeline, use the config-driven entry point (`CgcnnTrainer` subclasses the same base):

```bash
# equivalently: matimba-train --cd expt_configs/final --cf log_kvrh_smooth_dila.yaml
python src/MatImba/run_trainer.py --cd expt_configs/final --cf log_kvrh_smooth_dila.yaml
```

Config suffixes: `_smooth_dila.yaml` (DILA, proposed) · `_dir.yaml` (Deep Imbalanced Regression) · `_bsam.yaml` (Balanced Sharpness-Aware Minimisation) · `.yaml` (control).

### 3. Analyse: OOD error → screening performance → SERA–α diagnosis

Everything in `MatImba.analysis` is model-agnostic: it consumes the `*_test_predictions.csv` and `*_val_log.csv` files the trainer writes (or plain arrays).

```bash
# Full standard report: metric tables (ddof=1, significant-figure formatting),
# alpha-SERA coupling, screening metrics, summary figures
matimba-analyse --experiments experiments/final --out figs/final
```

Or from Python:

```python
from MatImba.analysis import (collect_experiments, load_trajectories,
                              coupling_table, summary_plot, discovery_metrics)

index, preds = collect_experiments("experiments/final")

# OOD error decomposition (works identically on CV and MatFold splits)
summary_plot(preds["log_gvrh"], target_name="log_gvrh")

# Screening: precision/recall/enrichment at an experimental budget
print(discovery_metrics(preds["log_gvrh"]["smooth_dila"], budget_ratio=1.0))

# The core diagnosis: is tail error coupled to awareness alpha?
coupling = coupling_table(load_trajectories(index))
print(coupling[["dataset", "method", "regime", "spearman"]])
```

The coupling `regime` is the actionable output:

| regime | meaning | implication |
|---|---|---|
| `linear` | SERA falls monotonically with α | awareness-based training (e.g. DILA) buys tail accuracy |
| `thresholded` | coupling only below a breakpoint | gains saturate once α is high |
| `decoupled` | no significant dependence | the tail-error floor is set by dataset geometry (`W₁`); collect data, don't tune losses |

Threshold conventions are package-wide constants in `MatImba.analysis`: `SERA_T0 = 0.5` (SERA integration lower bound), `TAIL_PHI = 0.8` (head/tail partition), `SCREEN_PHI = 0.75` (screening candidates).

### Reproducing the paper

```bash
python scripts/make_paper_figs.py --out figs/paper
```

regenerates every figure and table (Fig 1–4, the quantitative α–SERA analysis, and the SI metric/coupling tables) from the archived predictions in `experiments/`.

---

## 📏 Metrics

- **SERA** — squared error–relevance area: tail error integrated over relevance thresholds.
- **α (awareness)** — `1 − dCor(log-L1 error, 1/density)`: 1 means errors are independent of label density; 0 means errors track sparsity.
- **h, Gini, D_KL, W₁** — bounded dataset imbalance descriptors (statistical magnitude and transport cost).
- **vHTS metrics** — extrapolative precision, tail recall and enrichment factor under a screening budget.

## 📊 Datasets

Benchmarked across the **MatBench v0.1** suite with both standard 5-fold CV and **MatFold** structure-disjoint (OOD) splits: `log_kvrh`, `log_gvrh`, `perovskites`, `phonons` (modelled); 10 targets profiled in Stage 1.

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

`tests/test_base_trainer.py` trains a small MLP through the full robust loop on CPU — the proof that the trainer works beyond graph networks.

---

## 📝 Citation

```bibtex
@article{lei2026unbiasing,
  title={Unbiasing Materials Discovery: A Distribution Imbalance-Aware Framework for Robust Regression},
  author={Lei, Lei and Witman, Matthew D. and Stavila, Vitalie and Grant, David M. and Dornheim, Martin and Ling, Sanliang},
  journal={*********************},
  year={2026}
}
```

## 🤝 Acknowledgements

This work was supported by EPSRC (EP/V042556/1) and the Leverhulme Trust. Computing resources were provided by the University of Nottingham's Ada HPC and the Sulis supercomputer.

## 📜 License

Distributed under the CC0 1.0 Universal license. See `LICENSE` for details.
