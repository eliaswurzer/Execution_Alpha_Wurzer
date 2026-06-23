# Reproducibility Notes

The synthetic test suite runs without licensed market data:

```powershell
python -B -m compileall -q coding
python -B -m pytest coding/tests -q -m "not realdata"
```

Real-data reproduction requires locally available licensed inputs. Configure
paths through CLI flags or environment variables:

```powershell
$env:THESIS_RUN_ROOT = "C:\\path\\to\\workspace"
$env:THESIS_DATA_ROOT = "C:\\path\\to\\licensed_taq_processed"
$env:THESIS_ARTIFACTS_DIR = "C:\\path\\to\\artifacts"
$env:THESIS_VOLUME_DB = "C:\\path\\to\\volume\\dollar_volume.duckdb"
$env:THESIS_INDEX_MEMBERSHIP_DIR = "C:\\path\\to\\reference\\index_membership"
$env:THESIS_TAQ_PARQUET_2018 = "C:\\path\\to\\processed\\2018"
$env:THESIS_TAQ_PARQUET_2019 = "C:\\path\\to\\processed\\2019"
$env:THESIS_RAW_TAQ_ROOT = "C:\\path\\to\\raw_dtaq"
```

Full empirical runs should be configured through the runner-specific CLI flags
and the environment variables above. Use local licensed-data locations for all
market-data and constituent inputs before running the pipeline.

The public quick tests are synthetic and do not require licensed inputs.
