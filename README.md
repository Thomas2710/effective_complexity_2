# Effective Complexity

A research-oriented Python package for studying **representation learning**, **effective dimensionality**, and **teacher–student alignment** using synthetic data.

The framework is designed to be:

- Modular  
- Reproducible  
- Easy to extend with new datasets and models  

It supports:

- Synthetic Gaussian datasets with controllable dimensionality  
- Teacher–student training via KL divergence  
- MLP and ResMLP architectures  
- Evaluation using PCA, CCA, residual CCA, alignment error, and diversity  
- Multi-architecture and multi-seed sweeps  
- Automatic aggregation of results  

---

## 🚀 Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd effective_complexity_pkg
```

Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

Install in editable mode
```bash
pip install -e .
```
## Repository structure
project_root/
├── pyproject.toml
├── config.yaml
├── run_all_experiments.py
├── aggregate_results.py
├── effective_complexity/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── experiments/
│   └── utils/
└── experiments_out/


## Generate Datasets

You can generate synthetic datasets before training using:
```bash
python synthetic.py
```


or, if exposed through your runner:
```bash
python runner.py --generate
```

Typical options include:
```bash
python synthetic.py \
  --dataset gaussian_clusters \
  --num_samples 50000 \
  --dim 128 \
  --num_classes 10 \
  --output_dir data/
```

This will create dataset files that can later be loaded by training and evaluation scripts.

Training with Generated Data
python train.py --dataset data/gaussian_clusters.npz

End-to-End Example
## Generate data
```bash
python synthetic.py --dataset gaussian_clusters --num_samples 50000
```

## Train model
```bash
python train.py --dataset data/gaussian_clusters.npz
```

## Evaluate representations
```bash
python evaluate.py --checkpoint outputs/model.pt
```



## Running Experiments via User Scripts

In addition to command-line entry points, you can launch large experiment sweeps and post-process results using the provided user scripts.


Run All Experiments
```bash
python run_all_experiments.py
```

This script orchestrates multiple training and evaluation runs across different configurations (models, datasets, depths, seeds, etc.).

Typical use cases:

Architecture sweeps

Hyperparameter studies

Reproducibility experiments

Aggregate Results

After experiments finish, aggregate and summarize outputs:

```bash
python aggregate_results.py
```
This script collects metrics from experiment folders and produces consolidated result files (e.g., CSV/JSON) for downstream analysis and plotting.