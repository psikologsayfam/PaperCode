# ICU ventilation outcome modelling

Source code for research on intensive care unit respiratory outcomes using genetic-algorithm ensemble methods.

## Repository structure

- `final_code.py`: original ensemble analysis implementation.
- `preprocessing.py`: original preprocessing implementation.
- `requirements.txt`: dependencies for the original code.
- [`revision/`](revision/README.md): separate strict two-hour HiRID admission-landmark analysis, with source reconstruction, nested evaluation and synthetic tests.

The revision code is a distinct analysis pathway. It does **not** establish equivalence with the original implementation or any previously reported performance estimate.

## Clinical data and research outputs

Restricted patient-level inputs are **not distributed**. Obtain clinical datasets only through the relevant authorised provider and follow the applicable data-use conditions.

All original data, processed cohorts, run diagnostics, evaluation tables, model checkpoints and other experiment outputs belong in a separate private research workspace. The repository publishes code and documentation only.

## Reproducing the revision

See [revision/README.md](revision/README.md) for the required HiRID input layout, dependency installation, execution order and verification scope.

The available synthetic tests check selected implementation helpers. They are **not** an independently verified end-to-end reproduction using restricted clinical records. The historical composite mortality outcome also requires additional temporal validation before clinical interpretation.
