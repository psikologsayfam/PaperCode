# ICU ventilation outcome prediction: strict two-hour analysis

This directory contains the source implementation of a retrospective HiRID v1.1.1 analysis using a landmark at two hours after ICU admission. It is maintained separately from the repository's original ensemble code. Only source files, a clinical-variable mapping and synthetic unit tests are published.

## Study protocol

1. Identify eligible patients from authorised local source records.
2. Reconstruct clinical measurements whose **measurement and electronic entry timestamps** both fall between ICU admission and the two-hour landmark.
3. Reconstruct the historical 44-feature representation, then the locked 46-predictor schema with an integrity-checked local cache.
4. Audit mortality observations in the absence of a verified death timestamp. A late recorded observation supports a lower bound on survival, not the time of death.
5. Evaluate six base classifiers using nested stratified cross-validation. Optimise genetic ensemble weights on training-only out-of-fold predictions, and compare with uniform weighting and logistic regression.

The reconstructed historical composite outcome includes mortality labels for which precise event times are unavailable. The analysis is **exploratory**, and the source cohort and outcome definitions require independent verification before clinical interpretation. No clinical claims or numerical performance results are asserted by this code release.

## Environment and data access

- Python 3.11 or later; install packages in `requirements.txt`.
- Obtain HiRID v1.1.1 through the provider's authorised data-access process. **No restricted clinical records are included.**
- Supply the original source tables within the private directory structure expected by the scripts.
- Set `NIV_RESEARCH_ROOT` to the secure experiment workspace. On the authorised Windows research computer the default is `D:\ML\research_campaign`.
- Give each execution a unique `NIV_RUN_ID` to prevent accidental overwrites and to connect dependent stages.

Generated manifests, diagnostics, runtime messages, patient-level predictions, saved models and all experiment outputs are written to the private workspace. They are never written to this repository. The `.gitignore` at repository root additionally excludes common data, model and output formats from source control.

## Install and run

From the repository root, in PowerShell:

```powershell
$env:NIV_RESEARCH_ROOT = 'D:\ML\research_campaign'
$env:NIV_RUN_ID = 'your-unique-run-id'
python -m pip install -r revision/requirements.txt
python -m unittest discover -s revision/tests -v
python revision/NIV_HIRID_R6/build_clean_raw_2h_cohort.py
python revision/NIV_HIRID_R6/build_original_44_features_strict_raw.py
python revision/NIV_HIRID_R6/build_original46_clean_cached.py
python revision/NIV_HIRID_R6/audit_death_only_post_landmark_vitals.py
python revision/NIV_HIRID_R6/fit_original46_ga_original_fitness.py
```

The scripts use a fixed schema and cohort-specific integrity checks. They **fail on missing or inconsistent sources** rather than generating substitute observations. Custom existing private stage folders can be provided through `NIV_CLEAN_BUILD_DIR`, `NIV_ORIGINAL44_BUILD_DIR` and `NIV_DEATH_AUDIT_DIR`.

## Source layout

| Path | Responsibility |
| --- | --- |
| `NIV_HIRID_R6/build_clean_raw_2h_cohort.py` | Source cohort reconstruction and strict two-hour gating |
| `NIV_HIRID_R6/build_original_44_features_strict_raw.py` | Historical feature-schema reconstruction |
| `NIV_HIRID_R6/build_original46_clean_cached.py` | Locked 46-predictor cache and SHA-256 manifest |
| `NIV_HIRID_R6/audit_death_only_post_landmark_vitals.py` | Post-landmark observation audit for unresolved death timing |
| `NIV_HIRID_R6/fit_original46_ga_original_fitness.py` | Nested cross-validation, GA weighting and comparators |
| `NIV_HIRID_R6/hirid_variable_map.json` | Clinical variable identifiers |
| `tests/test_model_helpers.py` | Synthetic numerical smoke tests |

## Verification scope

All included Python files pass syntax checks; the helper-level tests run on synthetic data. The entire workflow has **not** been executed here with the restricted HiRID source, so end-to-end reproducibility and reported results have not been independently verified in this release.
