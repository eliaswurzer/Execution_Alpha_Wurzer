# Statistical Methods Assessment and Inference Hardening

This document records an evaluation of the thesis statistical apparatus and the
hardening changes implemented on top of it. The question it answers is whether
the inference is sufficient to make precise, p-hacking-resistant claims and to
speak with confidence about statistical significance.

## Inventory of the two inference layers

The thesis runs inference in two layers.

The core layer in `coding/analysis/inference/` provides the primitives. The
two-way clustered standard error (`clustering.py`) follows Cameron, Gelbach and
Miller and clusters on symbol and on date with the symbol-date intersection
subtracted. The hypothesis tests (`tests.py`) implement the paired primary test
of S3-full against the Market-on-Close benchmark in the primary window, the
subgroup tests with a Holm step-down correction, and a rolling six-month
stability panel with a clustered confidence band.

The supplementary layer in `coding/statistical_tests/` consumes the persisted
hypothesis panels and produces the robustness and reproducibility artifacts:
the fill-specification robustness summary with Holm adjustment, the paired
alternative-minus-queue tests, the pooled H2 signal family, the exploratory
per-bin H2 union diagnostic, the out-of-sample fill-model calibration with a
Brier decomposition and an AUC diagnostic, the copy-ready LaTeX snippets, and a
manifest with SHA256 input hashing.

## What was already sound

The pre-registration discipline is in place: the primary hypotheses, the
primary window, and the directional alternatives are designated before the
panel is inspected, and exploratory analyses are separated from the
confirmatory set. The two-way clustered estimator is the correct choice for a
symbol-by-date panel with both cross-sectional and serial dependence. The
confirmatory-versus-exploratory split with Holm control within each emitted
family is the textbook protocol against type-I error inflation. The paired
designs on the primary symbol-date order surface remove cross-strategy common
variation without relying on batch-specific order identifiers. The
reproducibility infrastructure now combines input hashing, manifest recording,
feature-policy checks, canonical run gating, and fingerprinted stratified
out-of-sample sampling. The out-of-sample fill calibration is a genuine held-out
assessment.

## Gaps identified

1. The analytic p-values used a normal reference distribution with no
   distribution-free or few-cluster alternative. This is accurate at the
   full-sample scale of several hundred date clusters but can overstate
   significance when a subgroup leaves few clusters in one dimension.

2. The registered design is one-sided for H1, H2a, and H2b, but the code
   reported only two-sided p-values, so the reported number was not the
   pre-registered one.

3. There was no power or minimum-detectable-effect analysis, so a
   non-rejection could not be distinguished from an underpowered test.

4. The H3 risk-adjusted ranking (information ratio, tracking-error variance,
   RAEAR, rank order) was reported purely as point estimates with no sampling
   uncertainty.

5. The per-bin H2 differentials emitted ten uncorrected t-statistics per signal
   family. These rows are useful diagnostics, but they must not become a second
   confirmatory H2 decision surface beside the pooled matched differential.

6. There was no single consolidated registry making the full test surface and
   every correction auditable.

## What the hardening adds

Workstream A (headline significance). A reusable wild cluster bootstrap
(`analysis/inference/bootstrap.py`) for the mean-against-zero and paired
differential statistics, with Webb six-point weights by default and a two-way
multiplicative-weight construction, supplying a bootstrap p-value and a
studentized percentile-t confidence interval. A design-based minimum detectable
effect (`analysis/inference/power.py`). Both are wired into the
fill-specification summary so every headline row carries a bootstrap p-value, a
one-sided p-value, a confidence interval, and an MDE.

Workstream B (H3 inference). A block bootstrap by trading date
(`statistical_tests/h3_inference.py`) that recomputes the full H3 table on each
resample, emitting percentile confidence intervals for the information ratio,
the tracking-error variance, and the tracking-error standard deviation, the
bootstrap probability that the information-ratio ranking is preserved, the
pairwise ordering probabilities, and the probability that the RAEAR ranking
flips across the risk-aversion grid. The point-estimate methodology is
unchanged; this is an inferential supplement.

Workstream C (governance). One-sided p-values aligned with the registered
directional alternatives are computed alongside the existing two-sided values
in both the core tests and the supplementary suite, without mutating the
existing two-sided columns. The confirmatory H2 decision surface is the pooled
matched differential for H2a and H2b. The per-bin H2 family receives a Holm
correction across bins and a bootstrap max-t union diagnostic whose shared
cluster-weight draw respects cross-bin dependence. Benjamini-Hochberg
false-discovery-rate p-values are reported beside Holm for exploratory
families. A consolidated test registry enumerates every emitted test with its
role, family, correction, and decision p-value.

Workstream D (run lineage). The default configuration now points to the current
final v4 run family. The suite refuses to write final outputs unless every
configured input run is complete, has the expected 371 validated date shards,
has zero critical simulation failures, uses the current feature and trade
policies, and exposes non-empty H1/H2/H3 artifacts. The OOS event cache now
requires a manifest fingerprint tied to the headline H1 panel hash, sampling
parameters, feature and trade policies, and event-shard hashes.

## Reproducibility

All bootstraps are seeded from a fixed seed and the bootstrap settings
(replication counts, weight family, two-way flag, seed, alpha, power) are
recorded in the run manifest. New statistics are added as new columns and new
files. Final thesis outputs should be regenerated in the new dated v4 output
folder after all canonical input runs pass the gate; older folders remain
historical artifacts, not current final evidence.

## Deliberately deferred follow-ups

These require additional computation beyond the existing panels and were
scoped out of the present change.

A placebo falsification test that reruns the strategies on an off-window
interval (the 11:00 to 12:00 interval) and shows the effect vanishes. The
window is already defined in the configuration and the marker exists in the
core inference module, but no aggregated placebo run is wired in; it needs a
simulation pass.

Distributional and outlier diagnostics (skewness, kurtosis, and a
leave-one-cluster-out jackknife of the headline mean) to demonstrate the
headline is not driven by a few symbol-days.

## Thesis prose reconciled

The methodology prose in the current thesis source was reconciled with the
implemented inference (methodology sentences only; the results tables were left
unchanged by request).

The Inference section previously stated that "wild-cluster bootstrap inference
is not part of the standard evaluation design." It now describes the wild
cluster bootstrap as a distribution-free robustness check reported alongside the
asymptotic clustered estimator.

The H3 section previously stated that the rank-stability diagnostic is
"descriptive rather than a clustered pairwise test with confidence intervals."
It now records that the descriptive diagnostic is supplemented by a block
bootstrap over trading dates that attaches confidence intervals to the
information ratio and the tracking-error variance and quantifies the probability
that the ranking is preserved.

The Primary Hypothesis Designation section now states explicitly that for the
directional hypotheses the reported clustered p-values are two-sided and that
the registered one-sided p-value follows directly from them.

Still open in the thesis (results tables, deferred by request): the H1 primary
table reports the two-sided p-value (0.135 at t = -1.50); the column could be
labeled two-sided and the registered one-sided value (0.93, which does not
reject because the point estimate is negative) added in the note. The same
two-sided labeling applies to the exploratory subgroup tables.

## How to reproduce

Run the fast synthetic test suite:

```powershell
$env:PYTHONPATH='coding'
python -B -m pytest coding\tests -q -m "not realdata" -p no:cacheprovider --basetemp=coding\artifacts\pytest_tmp
```

Regenerate the suite against the existing validated panels:

```powershell
$env:PYTHONPATH='coding'
python -m statistical_tests.run_all --skip-oos --diagnostic-skip-oos
```

Pass `--skip-bootstrap` for a fast draft that keeps the analytic, one-sided, and
MDE columns but omits the bootstrap passes.
