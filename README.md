# MatImba: Distribution Imbalance-Aware Materials Discovery

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Unbiasing Materials Discovery: A Distribution Imbalance-Aware Framework for Robust Regression"**.

MatImba (Materials Imbalance) is a statistically rigorous framework designed to address the pervasive challenge of target imbalance in materials science datasets. By decoupling predictive error from empirical label density, MatImba enables robust regression even in the sparse, long-tailed extremities where high-value materials (e.g., ultra-hard alloys, high-temperature superconductors) typically reside.

---

## 🚀 Key Innovations

- **Distribution Imbalance Level ($h$):** A scale-invariant metric (normalised Pietra ratio) to rigorously quantify dataset skewness.
- **Geometric Topology ($W_1$):** Integration of Wasserstein distance to decouple statistical sparsity from geometric transport cost across property manifolds.
- **DILA Regularisation:** The **Distribution Imbalance Level Aware (DILA)** loss, which explicitly penalises the statistical dependence (Distance Correlation) between predictive residuals and local data density.
- **Accuracy-First Robust Selection:** A dynamic checkpoint selection criterion that prioritises global accuracy while enforcing structural awareness and tail-error stability.
- **Global Parity:** Systematically suppresses extreme error spikes in sparse regions (improving SERA) without degrading the Mean Absolute Error (MAE) of the well-represented bulk.

---

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-enabled GPU (recommended for training)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Lei-Lei-alpha/MatImba.git
   cd matimba
   ```

2. Install dependencies:
   ```bash
   pip install -r src/MatImba.egg-info/requires.txt
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

---

## 📖 Usage

### 1. Quantifying Dataset Imbalance
Use our diagnostic tools to evaluate your dataset's topology:
```python
from MatImba.utils.losses import calc_h, calc_w1

# target_values: array-like of regression targets
h_val = calc_h(target_values)
w1_val = calc_w1(target_values)
print(f"Imbalance Level (h): {h_val:.3f}, Wasserstein (W1): {w1_val:.3f}")
```

### 2. Training with DILA
The main entry point for training is `run_trainer.py`. We provide configuration files for MatBench datasets in `expt_configs/final/`.

```bash
# Train MEGNet on log_kvrh using DILA regularisation
python src/MatImba/run_trainer.py --config expt_configs/final/log_kvrh_smooth_dila.yaml
```

DILA is an loss objective function, you can easily use it in your own training scripts.

### 3. Comparison with Baselines
MatImba supports multiple imbalanced regression strategies:
- **Control:** Standard Empirical Risk Minimisation (MSE/L1).
- **DIR:** Deep Imbalanced Regression (Label/Feature Distribution Smoothing).
- **BSAM:** Balanced Sharpness-Aware Minimisation.

Example config suffix mappings:
- `_smooth_dila.yaml`: DILA (Proposed)
- `_dir.yaml`: Deep Imbalanced Regression
- `_bsam.yaml`: Balanced Sharpness-Aware Minimisation
- `.yaml`: Standard Control

### 4. Evaluation Metrics
We provide advanced metrics for tail-end robustness:
- **SERA:** Squared Error with respect to Relevance Area.
- **vHTS Simulation:** Virtual High-Throughput Screening (Precision/Recall for extreme values).

---

## 📊 Datasets

MatImba is benchmarked across the **MatBench v0.1** suite and **MatFold** structural extrapolation splits:
- `log_kvrh`: Bulk Modulus (Log)
- `log_gvrh`: Shear Modulus (Log)
- `perovskites`: Formation Energy
- `phonons`: Vibrational DOS Peak

---

## 📝 Citation

If you use this framework or the DILA loss in your research, please cite our paper:

```bibtex
@article{lei2026unbiasing,
  title={Unbiasing Materials Discovery: A Distribution Imbalance-Aware Framework for Robust Regression},
  author={Lei, Lei and Witman, Matthew D. and Stavila, Vitalie and Grant, David M. and Dornheim, Martin and Ling, Sanliang},
  journal={*********************},
  year={2026}
}
```

---

## 🤝 Acknowledgements
This work was supported by EPSRC (EP/V042556/1) and the Leverhulme Trust. Computing resources were provided by the University of Nottingham's Ada HPC and the Sulis supercomputer.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
