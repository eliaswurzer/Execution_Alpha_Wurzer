# Data Availability

This repository does not redistribute licensed, commercial, or locally
generated research data.

Excluded materials include:

- raw Daily TAQ Trade and NBBO files,
- processed Parquet files derived from raw TAQ,
- DuckDB volume databases,
- trained Cox, Kaplan-Meier, XGBoost, and value-model artifacts,
- commercial point-in-time index constituent snapshots,
- commercial symbol crosswalks and audit outputs,
- final thesis PDFs, office forms, logs, notebook drafts, and run artifacts.

The included `reference/index_membership` files provide public schema examples
and technical approximations for code inspection. They are not a substitute for
licensed point-in-time membership data when reproducing the final empirical
results.

Users who have access to the required licensed inputs can place them outside
the repository and point the pipeline to them via the environment variables
documented in `REPRODUCIBILITY.md`.
