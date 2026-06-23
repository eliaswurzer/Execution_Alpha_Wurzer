# Execution Alpha of Intraday Liquidity Provision versus Market-on-Close

This repository contains the public research code for the master thesis
"Execution Alpha of Intraday Liquidity Provision versus Market-on-Close in the
S&P 500".

The code implements a DTAQ-based execution simulation pipeline: preprocessing,
point-in-time universe handling, limit-order fill models, strategy simulation,
alpha and tracking-error metrics, hypothesis tests, and figure/table rendering.

## Repository Contents

- `coding/analysis/`: core data loaders, microstructure features, strategy
  logic, simulation engine, metrics, fill models, runners, and reporting tools.
- `coding/preprocessing/`: raw DTAQ preprocessing and data-audit utilities.
- `coding/volume/`: closing- and intraday-volume database builder.
- `coding/statistical_tests/`: supplementary inference and reporting suite.
- `coding/tests/`: synthetic unit and integration tests.
- `reference/index_membership/`: public or technical reference files and schema
  examples.

## Architecture Overview

The repository is organized as a reproducible research pipeline. Each layer is
kept separate so that licensed data access, empirical simulation, statistical
testing, and thesis reporting can be inspected independently.

```text
Licensed inputs and run configuration
        |
        v
Preprocessing and reference universe construction
        |
        v
Volume database and microstructure feature construction
        |
        v
Strategy simulation and fill-model robustness
        |
        v
Alpha, tracking-error, and hypothesis-test outputs
        |
        v
Tables, figures, and audit summaries
```

### 1. Configuration and Inputs

Real-data runs are controlled through CLI arguments or environment variables,
not personal machine defaults. The main inputs are licensed Daily TAQ Trade and
NBBO files, point-in-time index membership data, and optional run-output
directories.

### 2. Preprocessing Layer

`coding/preprocessing/` converts raw Trade and NBBO inputs into analysis-ready
per-date, per-symbol files. It also contains data-availability and transition
audits used to verify that the empirical universe has complete trade and quote
coverage before simulation.

### 3. Reference Universe Layer

`reference/index_membership/` documents the expected membership schema and
contains public approximation files suitable for code inspection and lightweight
checks. The final empirical thesis run used licensed point-in-time constituent
inputs that are not redistributed here.

### 4. Volume Layer

`coding/volume/` builds the volume database used for daily volume, closing
auction volume, intraday bucket shares, and expected closing-volume estimates.
These outputs determine parent-order sizing and support the liquidity and
sample-construction diagnostics.

### 5. Core Analysis Layer

`coding/analysis/` contains the empirical engine:

- `data/` loads TAQ-derived inputs, membership intervals, listing labels, and
  stress-day calendars.
- `microstructure/` builds spread, order-flow imbalance, signing, and imbalance
  features.
- `strategies/` defines the Market-on-Close benchmark, static passive,
  time-adaptive, signal-conditioned, and value-aware strategy rules.
- `simulation/` replays the trade tape against posted limit orders and routes
  any residual quantity to the closing auction.
- `fill_model/` implements Cox, Kaplan-Meier, XGBoost, adverse-selection, and
  schedule-model robustness components.
- `metrics/` computes execution alpha, fill rates, tracking-error variance, and
  risk-adjusted execution alpha.
- `inference/` implements clustered inference, bootstrap diagnostics, and power
  calculations.
- `reporting/` renders thesis-style tables and figures from validated outputs.
- `runners/` provides command-line entry points for the hypothesis pipeline,
  robustness passes, audits, and reporting exports.

### 6. Statistical Testing Layer

`coding/statistical_tests/` contains supplementary statistical routines and
method checks used to assess the economic tests, multiple-testing controls,
out-of-sample calibration, and H3 risk-ranking diagnostics.

### 7. Test Layer

`coding/tests/` contains synthetic unit and integration tests. The public test
suite is designed to run without licensed market data.

## Data Availability

Licensed raw DTAQ files, processed Parquet files, commercial constituent
snapshots, trained model artifacts, DuckDB databases, thesis PDFs, and local
run outputs are intentionally excluded. See `DATA_AVAILABILITY.md`.

## Quick Start

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -B -m pytest coding/tests -q -m "not realdata"
```

Real-data runs require local licensed inputs configured through environment
variables or explicit CLI flags. See `REPRODUCIBILITY.md`.

## License

No open-source license is granted in this public submission repository. The
code is visible for thesis review and archival inspection only unless the
author grants separate permission.
