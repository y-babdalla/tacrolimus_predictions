# DoseTAIlor: A Web-Based Platform for Personalised Tacrolimus Dose Optimisation Across Multi-Centre Populations Using Interpretable AI

**Authors**:  
Youssef Abdalla, Laura Gongas, Brais Muñiz Castro, Luis Ramudo Cela, Francisco Suárez, Mine Orlu, Luis Margusino-Framiñán, Abdul W. Basit, David Shorthouse, Alvaro Goyanes

**Preprint**: https://doi.org/10.21203/rs.3.rs-5907999/v1

**Software**: https://dosetailor.com/tacrolimusv1 

This repository provides end-to-end Python pipelines for:

- **Data preprocessing and feature engineering** (including imputation, feature scaling, sequence generation)
- **Training LSTM models** for tacrolimus level prediction (including vanilla, attention-based, and time-aware variants)
- **Cross-validation** and hyperparameter tuning (using Optuna)
- **Synthetic data generation and evaluation** (using SDV-based generative models)

---

## Overview

The code contained herein demonstrates how to:

1. **Preprocess tacrolimus data** (handling missing values, time columns, group-based imputation, etc.).  
2. **Train LSTM models** with different attention mechanisms (feature-level, time-level).  
3. **Perform cross-validation** and gather performance metrics (RMSE, MAE, etc.).  
4. **Generate synthetic datasets** (e.g. with Gaussian Copula, CTGAN, CopulaGAN, TVAE) and evaluate their similarity to real data.  

The ultimate goal is to predict tacrolimus levels under various clinical contexts, showcasing advanced deep learning architectures and data augmentation techniques.

---

## Repository Structure

A typical layout might look like this:

```
tacrolimus_predictions/
├── data/
│   └── tacrolimus_data.csv           # Example raw data
├── src/
│   ├── data/
│   │   ├── preprocess.py             # Data loading & preprocessing
│   │   └── config.py                 # Configuration for data columns, mappings, etc.
│   ├── models/
│   │   ├── lstm_models.py            # LSTM, Attention LSTM, Time-aware LSTM classes
│   │   ├── transfer_learning.py      # Transfer learning module (freezes LSTM layers)
│   │   └── ...
│   ├── train/
│   │   ├── trainer.py                # PyTorch Lightning training loop
│   │   └── cross_validation.py       # Cross-validation routines & hyperparameter tuning
│   └── synthetic/
│       ├── generate_synthetic.py     # Uses SDV to create synthetic datasets
│       └── evaluate_synthetic.py     # Evaluates synthetic data quality & coverage
├── scripts/
│   ├── run_training.py               # Example script to train an LSTM model
│   ├── run_crossval.py               # Example script to perform cross-validation
│   ├── run_synthetic.py              # Example script to generate/evaluate synthetic data
│   └── ...
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

- **`data/`** holds input data (or placeholders) for tacrolimus measurements, patient demographics, etc.  
- **`src/data/`** contains code for data cleaning, feature engineering, and configuration.  
- **`src/models/`** includes various LSTM architectures (including attention/time-decay variants) and transfer learning modules.  
- **`src/train/`** handles model training logic (e.g., PyTorch Lightning loops, cross-validation).  
- **`src/synthetic/`** handles SDV-based synthetic data generation and evaluation.  
- **`scripts/`** contains runnable scripts to train models, perform cross-validation, or generate synthetic data.  
- **`tests/`** includes unit tests for validating logic in the above modules.  

---

## Installation & Dependencies

1. **Install Python 3.11+** (earlier versions may work but are untested).
2. **Set up a virtual environment (recommended)**
3. **Install project dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

Below are some example commands to illustrate how you might run core functionalities. Adjust file paths, model parameters, and script arguments to match your setup.

1. **Train an LSTM model** (with or without attention/time-decay):
   ```bash
   python scripts/run_training.py \
       --model_type "time_aware" \
       --hidden_size 128 \
       --num_layers 2 \
       --learning_rate 1e-3 \
       --epochs 20
   ```
   This script reads in your data (from `data/`) and trains an LSTM model based on the arguments provided.

2. **Cross-validation**:
   ```bash
   python scripts/run_crossval.py \
       --folds 5 \
       --model_type "attention" \
       --study_name "my_optuna_study"
   ```
   This script performs (e.g.) K-fold cross-validation, optionally using Optuna for hyperparameter tuning.

3. **Generate synthetic data**:
   ```bash
   python scripts/run_synthetic.py \
       --model "gaussian" \
       --prefix "my_synth_experiment" \
       --num_rows 2000 \
       --evaluate
   ```
   This trains an SDV synthesiser on your real data, produces synthetic rows, and optionally evaluates them with diagnostic reports.

---

## Output Files

- **Model checkpoints** (e.g., `.ckpt` or `.pth` files) are often saved in a `checkpoints/` folder automatically by PyTorch Lightning.  
- **Evaluation metrics and logs** are typically written to a `logs/` or `results/` subfolder.  
- **Synthetic data** and diagnostic CSV files are saved under the specified output directory (e.g., `models/`, `synthetic_outputs/`).  

Consult the `scripts/` code for details on the exact paths and naming conventions.

---

## Contact and Citation

If you use or extend this codebase in academic or research contexts, please cite appropriately and consider acknowledging the authors. For help or questions, feel free to open an issue or reach out to the maintainers.
