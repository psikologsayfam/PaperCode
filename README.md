# GA‑Optimized Ensemble for Respiratory Outcome Prediction

This repository contains the complete cross‑validated implementation of a heterogeneous ensemble model whose weights are optimised by a genetic algorithm (GA). The code reproduces all results reported in the manuscript, including performance at clinical cutoffs, comparison with baseline models, and visualisations (ROC, calibration, feature importance).

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt

## Note
If you run the script outside Google Colab, remove the `google-colab` line from `requirements.txt` and modify the path loading section in the script (replace `drive.mount` and `os.chdir` with your local path).
## Data
The original dataset (0_2 merged_respiratory_modelling_final.csv) is not included due to confidentiality. Please contact the corresponding author for access, or place your own CSV with the same column structure in the working directory before running the script.

Expected columns: all numeric predictors, a binary Outcome column (Yes/No), and optionally a patientid column (will be dropped).

## Usage
Place the dataset in the same directory as the script.

## Run:

bash
python ga_ensemble_analysis.py
All console outputs (performance tables) and plots (ROC, calibration, feature importance, bar charts) will be displayed sequentially.

## Reproducibility
Random seeds are fixed (random_state=42).

All preprocessing steps (KNN imputation, scaling, SMOTE) are performed inside each CV fold to prevent data leakage.
