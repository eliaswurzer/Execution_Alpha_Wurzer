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
