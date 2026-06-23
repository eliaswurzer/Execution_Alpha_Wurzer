# Point-in-Time Index Membership References

This directory contains public or technical reference files used by the
research code. Licensed raw market data, commercial constituent snapshots, and
commercial crosswalks are not included.

## Included Files

- `public_sp500_20180102_symbols.csv` and related `public_sp500_*` files:
  technical public start-list approximations for preprocessing pilots.
- `sp500_membership_intervals_public_approx_2018_2019.csv`: a public
  approximation of S&P 500 interval membership for code inspection and
  lightweight checks.
- `sp500_ticker_events_2018_2019.csv`: public ticker-event support file.
- `expected_vc_identity_map.csv`: audited continuity map used by the expected
  closing-volume estimator.

## Not Included

The final thesis run used licensed data and commercially sourced membership
inputs that cannot be redistributed in this public repository. The following
materials are intentionally absent:

- raw Daily TAQ Trade and NBBO files,
- processed Parquet files,
- DuckDB volume databases,
- trained Cox, Kaplan-Meier, XGBoost, or value-model artifacts,
- commercial constituent snapshots,
- commercial symbol crosswalks and related audit outputs.

For full empirical reproduction, provide your own licensed point-in-time
membership file named `sp500_membership_intervals.csv` with columns:

```text
index_id,symbol,effective_from,effective_to,company_name,sector,listing_exchange,source,source_note
```

Only `index_id`, `symbol`, `effective_from`, and `effective_to` are required by
the loader. Dates must use ISO format (`YYYY-MM-DD`).

The public approximation is useful for understanding expected schema and code
paths, but it should not be treated as the authoritative thesis membership
ledger.
