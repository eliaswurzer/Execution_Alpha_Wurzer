from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import pytest

from analysis import config as cfg
from analysis.data import taq_loader
from analysis.runners import _common
from analysis.runners.calibrate_fill_model import run_calibration


pytestmark = pytest.mark.realdata


def _require_realdata_root() -> None:
    if os.environ.get("THESIS_ENABLE_REALDATA_TESTS") != "1":
        pytest.skip("set THESIS_ENABLE_REALDATA_TESTS=1 to run real-data smoke tests")
    root = cfg.TAQ_PARQUET_DIR[2018]
    required = [
        root / "20180102" / "trades" / "AAPL.parquet",
        root / "20180102" / "nbbo" / "AAPL.parquet",
        root / "20180102" / "qc" / "trade_qc_summary.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"real-data smoke root incomplete: {missing}")


def test_realdata_loads_qc_valid_aapl_msft_symbol_days() -> None:
    _require_realdata_root()
    day = dt.date(2018, 1, 2)

    assert taq_loader.trade_qc_policy_status(day) == (True, "ok")
    for sym in ["AAPL", "MSFT"]:
        trades, nbbo = taq_loader.load_symbol_day(day, sym)
        assert not trades.empty
        assert not nbbo.empty
        assert {"time", "price", "volume", "sale_condition"}.issubset(trades.columns)
        assert {"time", "best_bid", "best_offer", "mid"}.issubset(nbbo.columns)


def test_realdata_mini_calibration_writes_xgb_artifacts_and_validates_dry_run(artifact_dir) -> None:
    _require_realdata_root()
    day = dt.date(2018, 1, 2)
    out_dir = artifact_dir / "mini_calibration"

    run_calibration(
        ["AAPL", "MSFT"],
        start=day,
        end=day,
        out_dir=out_dir,
        workers=1,
        event_sample_per_symbol_day=300,
        as_sample_per_symbol_day=100,
        event_sample_every_seconds=30,
        min_coverage=0.95,
        fit_xgb_survival=True,
        xgb_device="cpu",
    )

    manifest = pd.read_json(out_dir / "calibration_manifest.json", typ="series")
    assert manifest["status"] == "complete"
    assert manifest["coverage"] == 1.0
    assert manifest["xgb_survival_status"] == "complete"
    assert list(out_dir.glob("xgb_tier_*.ubj"))
    assert _common.validate_run(
        ["S1_STATIC"],
        day,
        day,
        out_dir,
        symbols=["AAPL", "MSFT"],
        fill_specification="xgb",
    )
