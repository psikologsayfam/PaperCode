# GA‑Optimized Ensemble for Respiratory Outcome Prediction

This repository contains the complete cross‑validated implementation of a heterogeneous ensemble model whose weights are optimised by a genetic algorithm (GA). The code reproduces all results reported in the manuscript, including performance at clinical cutoffs, comparison with baseline models, and visualisations (ROC, calibration, feature importance).

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt

#Note
If you run the script outside Google Colab, remove the `google-colab` line from `requirements.txt` and modify the path loading section in the script (replace `drive.mount` and `os.chdir` with your local path).
