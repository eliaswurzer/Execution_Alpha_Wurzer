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
$env:THESIS_REFINITIV_CONSTITUENTS = "C:\\path\\to\\constituents_sp500.csv"
```

The template run configuration at
`coding/analysis/run_configs/final_2018_2019_value_aware.json` shows the
expected shape of the final-run config. Replace the neutral placeholder paths
with local licensed-data locations before running a full empirical pipeline.

The default quick tests deliberately exclude tests marked `realdata`. To run
real-data checks, set `THESIS_ENABLE_REALDATA_TESTS=1` after configuring the
data paths above.
