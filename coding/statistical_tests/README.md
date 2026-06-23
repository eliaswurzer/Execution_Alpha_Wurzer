# Statistical Tests

Central supplementary statistics package for the thesis.

It collects:

- multiple-testing adjustment utilities,
- fill-specification economic tests,
- paired alternative-minus-queue diagnostics,
- stratified out-of-sample fill-model calibration,
- a consolidated test registry,
- thesis-ready LaTeX snippets.

Default command from the workspace root:

```powershell
$env:PYTHONPATH='coding'
python -m statistical_tests.run_all
```

Diagnostic render without rebuilding the OOS event panel:

```powershell
$env:PYTHONPATH='coding'
python -m statistical_tests.run_all --skip-oos --diagnostic-skip-oos
```

Default outputs are written to:

```text
coding/artifacts/statistical_tests_20260618_final_v4/
```

The default input family is the final v4 run set: queue-aware tape replay,
strict tape replay, at-or-through tape replay, Cox, Kaplan-Meier, and XGBoost
survival. The runner refuses to write final outputs unless each configured run
is complete, has 371 validated shards, has zero critical simulation failures,
uses the current feature and trade policies, and exposes non-empty H1/H2/H3
artifacts.

The queue-aware tape replay remains the headline fill mechanism. Strict and
at-or-through tape replay are deterministic robustness bounds. Cox,
Kaplan-Meier, and XGBoost are model-based robustness specifications. The
model-based rows are interpreted as fill-hazard robustness, while tape replay
remains the primary tape-feasible economic evidence.

The confirmatory H2 decision surface is the pooled matched differential for
H2a and H2b. Per-bin max-t union rows are exploratory diagnostics. H3 is
reported as a descriptive risk-return trade-off analysis with block-bootstrap
uncertainty diagnostics, not as a separate confirmatory decision p-value.

OOS event-panel reuse is guarded by `oos_event_manifest.json`, which records a
cache policy, the headline H1 panel hash, sampling parameters, feature and
trade policies, and event-shard hashes. Changing these inputs forces the OOS
status file to be rebuilt unless `--force-oos` is already in use.
